from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import HTMLResponse
from pathlib import Path
from datetime import datetime
import PyPDF2
from docx import Document
import uvicorn
import os
import logging
import openai
from dotenv import load_dotenv
import json
from pdf2image import convert_from_path
import pytesseract

app = FastAPI()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()
OPENAI_API_KEY = os.getenv("apikey")
if not OPENAI_API_KEY:
    logger.error("OPENAI_API_KEY environment variable not set")
    raise EnvironmentError("OPENAI_API_KEY environment variable not set")

# Initialize OpenAI client
client = openai.OpenAI(api_key=OPENAI_API_KEY)

# In-memory storage for cases and files
cases = {}  # {case_id: created_at}
case_files = {}  # {case_id: [{file_name, file_path, upload_date, timeline, issues, grounds, parties, document_tag, key_events_tag}]}

# Create uploads directory
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

async def call_llm_api(content: str, file_name: str, document_tag: str, key_events_tag: str) -> dict:
    try:
        prompt = f"""
You are an expert legal document analyzer with deep knowledge of Indian law. Analyze the following legal document content and extract the requested information in JSON format. The document may have varied structures, so do not rely on specific patterns. Use your understanding of Indian legal context to identify:

1. **Parties**: Identify the plaintiff and defendant (e.g., from case titles like "Party1 v. Party2" or mentions of "plaintiff"/"defendant"). Use "Unknown Plaintiff" or "Unknown Defendant" if not found.
2. **Timeline**: Extract a unified timeline of events with dates in "Month DD, YYYY" format (e.g., "January 15, 2025") for specific dates or "YYYY" format (e.g., "2025") for year-only events. Ensure all dates are consistently formatted and represent the event's occurrence accurately. There might be some events that contain only month and year. Display that in the format Month, YYYY. For example, if you see something like "in the month of novemmber, 2023", then the output must be "November, 2023". Or if you see anything like "nov, 2025", then output must be "November, 2025". some example are - for "2015-Apr", output is "April, 2015". for "Oct, 2020", the output must be "October, 2020". "for 01-01-2020" or "01/01/2020", the output must be "January 01, 2020". Use these examples to extract correct dates. The dates MUST BE in sorted order from oldest to latest.
3. **Legal Issues**: For each party (plaintiff and defendant), identify legal issues, clearly stating:
   - What the party is doing wrong (specific actions or omissions).
   - Applicable Indian laws they might face (e.g., Indian Contract Act, 1872; Indian Penal Code, 1860; Trade Marks Act, 1999; or specific constitutional articles). Include section numbers where relevant.
   - Summarize each issue in a full sentence (e.g., "Plaintiff failed to deliver goods as per contract, violating Section 55 of the Indian Contract Act, 1872").
4. **Grounds for Winning**: Determine grounds on which each party could win, based on evidence or defenses mentioned in the document.
5. **Mismatches**: Identify any conflicting events in the timeline (e.g., multiple events on the same date with contradictory summaries). Additionally, analyze the timeline for any facts that conflict with the Indian Constitution (e.g., violations of fundamental rights under Articles 14, 19, or 21). If no constitutional conflicts are found, state "No facts identified that conflict with the Indian Constitution."

Consider the document tag ({document_tag}) and key events tag ({key_events_tag}) to contextualize your analysis, but do not include them in the output JSON unless explicitly relevant to the issues or grounds.

Return the response in JSON with the following structure:
{{
  "parties": {{"plaintiff": "string", "defendant": "string"}},
  "timeline": [{{"document_date": "string", "summary": "string"}}],
  "issues": {{"plaintiff": ["string"], "defendant": ["string"]}},
  "grounds": {{"plaintiff": ["string"], "defendant": ["string"]}},
  "mismatches": ["string"]
}}

Document content:
{content}
"""
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a legal document analysis assistant with expertise in Indian law. Provide structured JSON output as requested."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=1500,
            temperature=0.7,
            response_format={"type": "json_object"}
        )
        
        llm_output = response.choices[0].message.content
        parsed_output = json.loads(llm_output.strip())
        
        # Add is_important flag to each timeline event
        for event in parsed_output["timeline"]:
            event["is_important"] = False
        
        # Validate response structure
        required_keys = ["parties", "timeline", "issues", "grounds", "mismatches"]
        if not all(key in parsed_output for key in required_keys):
            logger.error(f"Invalid LLM response structure: {parsed_output}")
            raise HTTPException(status_code=500, detail="Invalid LLM response structure")
        
        return parsed_output
    except openai.OpenAIError as e:
        logger.error(f"OpenAI API error for {file_name}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"OpenAI API error: {str(e)}")
    except Exception as e:
        logger.error(f"LLM processing error for {file_name}: {str(e)}")
        return {
            "timeline": [{"document_date": datetime.now().strftime("%B %d, %Y"), "summary": f"Error processing {file_name}", "is_important": False}],
            "issues": {"plaintiff": ["Error identifying issues"], "defendant": ["Error identifying issues"]},
            "grounds": {"plaintiff": ["Error identifying grounds"], "defendant": ["Error identifying grounds"]},
            "mismatches": ["Error analyzing mismatches"],
            "parties": {"plaintiff": "Unknown", "defendant": "Unknown"}
        }

async def call_chatbot_llm(case_id: str, user_message: str) -> str:
    try:
        if case_id not in case_files:
            raise HTTPException(status_code=404, detail="Case not found")
        
        case_data = case_files[case_id]
        context = "Case Context:\n"
        for file_data in case_data:
            context += f"File: {file_data['file_name']}, Document Tag: {file_data['document_tag']}, Key Events Tag: {file_data['key_events_tag']}\n"
            context += f"Parties: {file_data['parties']['plaintiff']} vs {file_data['parties']['defendant']}\n"
            context += f"Timeline: {json.dumps(file_data['timeline'], indent=2)}\n"
            context += f"Issues: {json.dumps(file_data['issues'], indent=2)}\n"
            context += f"Grounds: {json.dumps(file_data['grounds'], indent=2)}\n"
            context += f"Mismatches: {json.dumps(file_data['mismatches'], indent=2)}\n\n"
        
        prompt = f"""
You are a legal assistant with expertise in Indian law. The user is asking about a case with the following details:
{context}

User's question: {user_message}

Provide a clear, concise, and accurate response based on the case data and your legal expertise. If the question is unrelated to the case, respond appropriately but keep the answer relevant to Indian legal context if possible.
"""
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a legal assistant with expertise in Indian law."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=500,
            temperature=0.7
        )
        
        return response.choices[0].message.content.strip()
    except openai.OpenAIError as e:
        logger.error(f"OpenAI API error in chatbot: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Chatbot API error: {str(e)}")
    except Exception as e:
        logger.error(f"Chatbot processing error: {str(e)}")
        return "Error processing your request. Please try again."

def extract_text(file_path: str, file_type: str) -> str:
    try:
        if file_type == "pdf":
            with open(file_path, "rb") as f:
                pdf = PyPDF2.PdfReader(f)
                if len(pdf.pages) == 0:
                    raise ValueError("PDF is empty or corrupted")
                text = ""
                for page in pdf.pages:
                    extracted = page.extract_text()
                    if extracted:
                        text += extracted
                if text.strip():
                    logger.info(f"Extracted {len(text)} characters from PDF using PyPDF2")
                    return text[:5000]

                # Fallback to OCR for scanned PDFs
                logger.info("No text extracted with PyPDF2, attempting OCR")
                images = convert_from_path(file_path)
                text_parts = []
                for i, image in enumerate(images):
                    text = pytesseract.image_to_string(image, lang='eng')
                    if text.strip():
                        text_parts.append(text)
                        logger.info(f"Extracted {len(text)} characters from page {i + 1} using OCR")
                    else:
                        logger.warning(f"No text extracted from page {i + 1} using OCR")
                text = "\n".join(text_parts)
                if not text.strip():
                    raise ValueError("No text could be extracted from PDF, even with OCR")
                logger.info(f"Extracted {len(text)} characters from PDF using OCR")
                return text[:5000]
        elif file_type == "docx":
            doc = Document(file_path)
            text = " ".join([para.text for para in doc.paragraphs if para.text.strip()])
            if not text.strip():
                raise ValueError("No text could be extracted from DOCX")
            return text[:5000]
        elif file_type == "txt":
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()
                if not text.strip():
                    raise ValueError("No text could be extracted from TXT")
                return text[:5000]
        raise ValueError(f"Unsupported file type: {file_type}")
    except Exception as e:
        logger.error(f"Error extracting text from {file_path}: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Error extracting text: {str(e)}")

@app.post("/create_case")
async def create_case(file: UploadFile = File(...), document_tag: str = Form(...), key_events_tag: str = Form(...)):
    file_ext = file.filename.split(".")[-1].lower()
    if file_ext not in ["pdf", "docx", "txt"]:
        raise HTTPException(status_code=400, detail="Invalid file type. Only PDF, DOCX, TXT allowed.")

    case_id = f"CASE_{datetime.now().strftime('%Y%m%d%H%M%S')}"
    cases[case_id] = datetime.now().isoformat()
    
    file_path = UPLOAD_DIR / f"{case_id}_{file.filename}"
    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")
    
    with open(file_path, "wb") as f:
        f.write(content)
    
    text = extract_text(file_path, file_ext)
    analysis = await call_llm_api(text, file.filename, document_tag, key_events_tag)
    
    case_files[case_id] = [{
        "file_name": file.filename,
        "file_path": str(file_path),
        "upload_date": datetime.now().isoformat(),
        "timeline": analysis["timeline"],
        "issues": analysis["issues"],
        "grounds": analysis["grounds"],
        "parties": analysis["parties"],
        "mismatches": analysis.get("mismatches", ["No mismatches identified"]),
        "document_tag": document_tag,
        "key_events_tag": key_events_tag
    }]
    
    logger.info(f"Case created: {case_id}, File: {file.filename}, Document Tag: {document_tag}, Key Events Tag: {key_events_tag}")
    return {"message": "Case created successfully", "case_id": case_id}

@app.post("/update_case")
async def update_case(case_id: str = Form(...), file: UploadFile = File(...), document_tag: str = Form(...), key_events_tag: str = Form(...)):
    file_ext = file.filename.split(".")[-1].lower()
    if file_ext not in ["pdf", "docx", "txt"]:
        raise HTTPException(status_code=400, detail="Invalid file type. Only PDF, DOCX, TXT allowed.")

    if case_id not in cases:
        raise HTTPException(status_code=404, detail="Case not found")

    file_path = UPLOAD_DIR / f"{case_id}_{file.filename}"
    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")
    
    with open(file_path, "wb") as f:
        f.write(content)
    
    text = extract_text(file_path, file_ext)
    analysis = await call_llm_api(text, file.filename, document_tag, key_events_tag)
    
    case_files[case_id].append({
        "file_name": file.filename,
        "file_path": str(file_path),
        "upload_date": datetime.now().isoformat(),
        "timeline": analysis["timeline"],
        "issues": analysis["issues"],
        "grounds": analysis["grounds"],
        "parties": analysis["parties"],
        "mismatches": analysis.get("mismatches", ["No mismatches identified"]),
        "document_tag": document_tag,
        "key_events_tag": key_events_tag
    })
    
    logger.info(f"Case updated: {case_id}, File: {file.filename}, Document Tag: {document_tag}, Key Events Tag: {key_events_tag}")
    return {"message": "Case updated successfully", "case_id": case_id}

@app.post("/chat/{case_id}")
async def chat(case_id: str, message: str = Form(...)):
    if case_id not in cases:
        raise HTTPException(status_code=404, detail="Case not found")
    
    response = await call_chatbot_llm(case_id, message)
    logger.info(f"Chat request for case {case_id}: {message}")
    return {"response": response}

@app.post("/toggle_important/{case_id}/{event_index}")
async def toggle_important(case_id: str, event_index: int):
    if case_id not in cases:
        raise HTTPException(status_code=404, detail="Case not found")
    
    combined_timeline = []
    for file_data in case_files.get(case_id, []):
        combined_timeline.extend(file_data["timeline"])
    
    if event_index < 0 or event_index >= len(combined_timeline):
        raise HTTPException(status_code=400, detail="Invalid event index")
    
    combined_timeline[event_index]["is_important"] = not combined_timeline[event_index].get("is_important", False)
    
    # Update the original case_files timeline
    current_index = 0
    for file_data in case_files[case_id]:
        for event in file_data["timeline"]:
            if current_index == event_index:
                event["is_important"] = combined_timeline[event_index]["is_important"]
                break
            current_index += 1
    
    logger.info(f"Toggled importance for event {event_index} in case {case_id}: {combined_timeline[event_index]['is_important']}")
    return {"message": "Event importance toggled successfully", "is_important": combined_timeline[event_index]["is_important"]}

@app.get("/timeline/{case_id}")
async def get_timeline(case_id: str):
    if case_id not in cases:
        logger.error(f"Case not found: {case_id}")
        raise HTTPException(status_code=404, detail="Case not found")
    
    logger.info(f"Fetching timeline for case: {case_id}, Files: {len(case_files.get(case_id, []))}")
    combined_timeline = []
    plaintiff_issues = []
    defendant_issues = []
    plaintiff_grounds = []
    defendant_grounds = []
    mismatches = []
    parties = {"plaintiff": "Unknown Plaintiff", "defendant": "Unknown Defendant"}
    document_tags = []
    key_events_tags = []
    
    for file_data in case_files.get(case_id, []):
        combined_timeline.extend(file_data["timeline"])
        plaintiff_issues.extend(file_data["issues"]["plaintiff"])
        defendant_issues.extend(file_data["issues"]["defendant"])
        plaintiff_grounds.extend(file_data["grounds"]["plaintiff"])
        defendant_grounds.extend(file_data["grounds"]["defendant"])
        mismatches.extend(file_data["mismatches"])
        parties.update(file_data["parties"])
        document_tags.append(file_data["document_tag"])
        key_events_tags.append(file_data["key_events_tag"])
    
    def parse_date(date_str):
        try:
            return datetime.strptime(date_str, "%B %d, %Y")
        except ValueError:
            try:
                return datetime.strptime(date_str, "%B, %Y")
            except ValueError:
                try:
                    return datetime.strptime(date_str, "%Y")
                except ValueError:
                    logger.warning(f"Invalid date format: {date_str}, using current date as fallback")
                    return datetime.now()
    
    combined_timeline = sorted(combined_timeline, key=lambda x: parse_date(x["document_date"]))
    
    return {
        "case_id": case_id,
        "parties": parties,
        "timeline": combined_timeline,
        "plaintiff_issues": list(set(plaintiff_issues)) if plaintiff_issues else [f"No issues identified for {parties['plaintiff']}."],
        "defendant_issues": list(set(defendant_issues)) if defendant_issues else [f"No issues identified for {parties['defendant']}."],
        "analysis": {
            "mismatches": list(set(mismatches)) if mismatches else ["No facts identified that conflict with the Indian Constitution."],
            "plaintiff_grounds": list(set(plaintiff_grounds)) if plaintiff_grounds else [f"No strong grounds identified for {parties['plaintiff']}."],
            "defendant_grounds": list(set(defendant_grounds)) if defendant_grounds else [f"No strong grounds identified for {parties['defendant']}."],
            "document_tags": list(set(document_tags)) if document_tags else ["No document tags available"],
            "key_events_tags": list(set(key_events_tags)) if key_events_tags else ["No key events tags available"]
        }
    }

@app.get("/cases")
async def get_cases():
    return {"cases": list(cases.keys())}

@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    html_content = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Case Timeline Builder</title>
    <script src="https://cdn.jsdelivr.net/npm/react@18.2.0/umd/react.development.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/react-dom@18.2.0/umd/react-dom.development.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/@babel/standalone@7.22.5/babel.min.js"></script>
    <script src="https://cdn.tailwindcss.com"></script>
    <style>
        .loading-overlay {
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: rgba(0, 0, 0, 0.5);
            display: flex;
            justify-content: center;
            align-items: center;
            z-index: 1000;
        }
        .loading-gif {
            width: 50px;
            height: 50px;
        }
        .chat-container {
            height: 400px;
            overflow-y: auto;
            border: 1px solid #ccc;
            padding: 10px;
            margin-top: 10px;
        }
        .chat-message {
            margin-bottom: 10px;
        }
        .user-message {
            background-color: #e0f7fa;
            padding: 8px;
            border-radius: 5px;
            text-align: right;
        }
        .bot-message {
            background-color: #f3f4f6;
            padding: 8px;
            border-radius: 5px;
        }
        .important-event {
            background-color: #ffedd5;
            padding: 8px;
            border-radius: 5px;
            font-weight: bold;
        }
    </style>
</head>
<body>
    <div id="root"></div>
    <script type="text/babel">
        function App() {
            const [file, setFile] = React.useState(null);
            const [caseId, setCaseId] = React.useState('');
            const [timeline, setTimeline] = React.useState([]);
            const [plaintiffIssues, setPlaintiffIssues] = React.useState([]);
            const [defendantIssues, setDefendantIssues] = React.useState([]);
            const [analysis, setAnalysis] = React.useState(null);
            const [parties, setParties] = React.useState({ plaintiff: 'Unknown', defendant: 'Unknown' });
            const [message, setMessage] = React.useState('');
            const [action, setAction] = React.useState(null);
            const [availableCases, setAvailableCases] = React.useState([]);
            const [loading, setLoading] = React.useState(false);
            const [documentTag, setDocumentTag] = React.useState('');
            const [keyEventsTag, setKeyEventsTag] = React.useState('');
            const [chatMessages, setChatMessages] = React.useState([]);
            const [chatInput, setChatInput] = React.useState('');
            const chatContainerRef = React.useRef(null);

            const handleFileChange = (e) => {
                setFile(e.target.files[0]);
            };

            const fetchCases = async () => {
                try {
                    const response = await fetch('/cases');
                    const data = await response.json();
                    setAvailableCases(data.cases || []);
                } catch (error) {
                    console.error('Fetch Cases Error:', error);
                }
            };

            React.useEffect(() => {
                fetchCases();
            }, []);

            React.useEffect(() => {
                if (chatContainerRef.current) {
                    chatContainerRef.current.scrollTop = chatContainerRef.current.scrollHeight;
                }
            }, [chatMessages]);

            const handleSubmit = async (e, endpoint) => {
                e.preventDefault();
                if (!file) {
                    setMessage('Please select a file');
                    return;
                }
                if (!documentTag) {
                    setMessage('Please enter a document tag');
                    return;
                }
                if (!keyEventsTag) {
                    setMessage('Please enter a key events tag');
                    return;
                }
                if (endpoint === 'update_case' && !caseId) {
                    setMessage('Please select a case ID');
                    return;
                }

                const formData = new FormData();
                formData.append('file', file);
                formData.append('document_tag', documentTag);
                formData.append('key_events_tag', keyEventsTag);
                if (endpoint === 'update_case') {
                    formData.append('case_id', caseId);
                }

                setLoading(true);
                try {
                    const response = await fetch(`/${endpoint}`, {
                        method: 'POST',
                        body: formData
                    });
                    const data = await response.json();
                    if (response.ok) {
                        setMessage(data.message);
                        setCaseId(data.case_id);
                        setFile(null);
                        setDocumentTag('');
                        setKeyEventsTag('');
                        fetchCases();
                    } else {
                        setMessage(data.detail || 'An error occurred');
                    }
                } catch (error) {
                    setMessage('Error: ' + error.message);
                } finally {
                    setLoading(false);
                }
            };

            const fetchTimeline = async () => {
                if (!caseId) {
                    setMessage('Please select a case ID');
                    return;
                }

                setLoading(true);
                try {
                    const response = await fetch(`/timeline/${caseId}`);
                    if (!response.ok) {
                        const errorData = await response.json();
                        throw new Error(errorData.detail || `HTTP error! Status: ${response.status}`);
                    }
                    const data = await response.json();
                    console.log('Timeline Data:', data);
                    setTimeline(data.timeline || []);
                    setPlaintiffIssues(data.plaintiff_issues || []);
                    setDefendantIssues(data.defendant_issues || []);
                    setAnalysis(data.analysis || {});
                    setParties(data.parties || { plaintiff: 'Unknown', defendant: 'Unknown' });
                    setMessage('');
                } catch (error) {
                    console.error('Fetch Timeline Error:', error);
                    setMessage(`Error fetching timeline: ${error.message}`);
                    setTimeline([]);
                    setPlaintiffIssues([]);
                    setDefendantIssues([]);
                    setAnalysis({});
                    setParties({ plaintiff: 'Unknown', defendant: 'Unknown' });
                } finally {
                    setLoading(false);
                }
            };

            const handleChatSubmit = async (e) => {
                e.preventDefault();
                if (!caseId) {
                    setMessage('Please select a case ID to chat');
                    return;
                }
                if (!chatInput.trim()) {
                    setMessage('Please enter a chat message');
                    return;
                }

                const userMessage = chatInput.trim();
                setChatMessages([...chatMessages, { type: 'user', text: userMessage }]);
                setChatInput('');

                setLoading(true);
                try {
                    const response = await fetch(`/chat/${caseId}`, {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
                        body: new URLSearchParams({ 'message': userMessage })
                    });
                    const data = await response.json();
                    if (response.ok) {
                        setChatMessages(prev => [...prev, { type: 'bot', text: data.response }]);
                        setMessage('');
                    } else {
                        setMessage(data.detail || 'Chat error occurred');
                    }
                } catch (error) {
                    setMessage('Chat Error: ' + error.message);
                } finally {
                    setLoading(false);
                }
            };

            const toggleImportant = async (eventIndex) => {
                if (!caseId) {
                    setMessage('Please select a case ID');
                    return;
                }

                setLoading(true);
                try {
                    const response = await fetch(`/toggle_important/${caseId}/${eventIndex}`, {
                        method: 'POST'
                    });
                    const data = await response.json();
                    if (response.ok) {
                        setTimeline(prevTimeline => {
                            const newTimeline = [...prevTimeline];
                            newTimeline[eventIndex].is_important = data.is_important;
                            return newTimeline;
                        });
                        setMessage('Event importance updated');
                    } else {
                        setMessage(data.detail || 'Error toggling importance');
                    }
                } catch (error) {
                    setMessage('Error toggling importance: ' + error.message);
                } finally {
                    setLoading(false);
                }
            };

            return (
                <div className="container mx-auto p-4">
                    {loading && (
                        <div className="loading-overlay">
                            <img
                                src="https://cdnjs.cloudflare.com/ajax/libs/galleriffic/2.0.2/css/loader.gif"
                                alt="Loading..."
                                className="loading-gif"
                            />
                        </div>
                    )}
                    <h1 className="text-2xl font-bold mb-4">Case Timeline Builder</h1>
                    
                    <div className="mb-4">
                        <button 
                            className="bg-blue-500 text-white px-4 py-2 mr-2 rounded"
                            onClick={() => setAction('create')}
                            disabled={loading}
                        >
                            Create New Case
                        </button>
                        <button 
                            className="bg-green-500 text-white px-4 py-2 mr-2 rounded"
                            onClick={() => setAction('update')}
                            disabled={loading}
                        >
                            Update Existing Case
                        </button>
                        <button 
                            className="bg-purple-500 text-white px-4 py-2 rounded"
                            onClick={() => setAction('timeline')}
                            disabled={loading}
                        >
                            View Timeline
                        </button>
                        <button 
                            className="bg-yellow-500 text-white px-4 py-2 ml-2 rounded"
                            onClick={() => setAction('chat')}
                            disabled={loading}
                        >
                            Chat About Case
                        </button>
                    </div>

                    {action && (action === 'create' || action === 'update') && (
                        <div>
                            {action === 'update' && (
                                <div className="mb-2">
                                    <label className="block">Select Case ID:</label>
                                    <select 
                                        value={caseId}
                                        onChange={(e) => setCaseId(e.target.value)}
                                        className="border p-2 w-full"
                                        disabled={loading}
                                    >
                                        <option value="">Select a case</option>
                                        {availableCases.map((id) => (
                                            <option key={id} value={id}>{id}</option>
                                        ))}
                                    </select>
                                </div>
                            )}
                            <div className="mb-2">
                                <label className="block">Document Tag (e.g., agreement, chargesheet):</label>
                                <input 
                                    type="text" 
                                    value={documentTag}
                                    onChange={(e) => setDocumentTag(e.target.value)}
                                    className="border p-2 w-full"
                                    placeholder="Enter document type"
                                    disabled={loading}
                                />
                            </div>
                            <div className="mb-2">
                                <label className="block">Key Events Tag (e.g., infringement, murder):</label>
                                <input 
                                    type="text" 
                                    value={keyEventsTag}
                                    onChange={(e) => setKeyEventsTag(e.target.value)}
                                    className="border p-2 w-full"
                                    placeholder="Enter key event type"
                                    disabled={loading}
                                />
                            </div>
                            <div className="mb-2">
                                <label className="block">Upload File (PDF, DOCX, TXT):</label>
                                <input 
                                    type="file" 
                                    accept=".pdf,.docx,.txt"
                                    onChange={handleFileChange}
                                    className="border p-2 w-full"
                                    disabled={loading}
                                />
                            </div>
                            <div className="mb-2">
                                <button 
                                    onClick={(e) => handleSubmit(e, action === 'create' ? 'create_case' : 'update_case')}
                                    className="bg-blue-500 text-white px-4 py-2 rounded"
                                    disabled={loading}
                                >
                                    Upload
                                </button>
                            </div>
                        </div>
                    )}

                    {action === 'timeline' && (
                        <div>
                            <div className="mb-2">
                                <label className="block">Select Case ID:</label>
                                <select 
                                    value={caseId}
                                    onChange={(e) => setCaseId(e.target.value)}
                                    className="border p-2 w-full"
                                    disabled={loading}
                                >
                                    <option value="">Select a case</option>
                                    {availableCases.map((id) => (
                                        <option key={id} value={id}>{id}</option>
                                    ))}
                                </select>
                            </div>
                            <button 
                                onClick={fetchTimeline}
                                className="bg-purple-500 text-white px-4 py-2 rounded"
                                disabled={loading}
                            >
                                Fetch Timeline
                            </button>
                        </div>
                    )}

                    {action === 'chat' && (
                        <div>
                            <div className="mb-2">
                                <label className="block">Select Case ID:</label>
                                <select 
                                    value={caseId}
                                    onChange={(e) => setCaseId(e.target.value)}
                                    className="border p-2 w-full"
                                    disabled={loading}
                                >
                                    <option value="">Select a case</option>
                                    {availableCases.map((id) => (
                                        <option key={id} value={id}>{id}</option>
                                    ))}
                                </select>
                            </div>
                            <div className="mb-2">
                                <label className="block">Chat About Case:</label>
                                <div className="chat-container" ref={chatContainerRef}>
                                    {chatMessages.map((msg, index) => (
                                        <div
                                            key={index}
                                            className={`chat-message ${msg.type === 'user' ? 'user-message' : 'bot-message'}`}
                                        >
                                            {msg.text}
                                        </div>
                                    ))}
                                </div>
                                <div className="mt-2 flex">
                                    <input
                                        type="text"
                                        value={chatInput}
                                        onChange={(e) => setChatInput(e.target.value)}
                                        className="border p-2 flex-grow"
                                        placeholder="Type your message..."
                                        disabled={loading}
                                        onKeyPress={(e) => e.key === 'Enter' && handleChatSubmit(e)}
                                    />
                                    <button
                                        onClick={handleChatSubmit}
                                        className="bg-blue-500 text-white px-4 py-2 ml-2 rounded"
                                        disabled={loading}
                                    >
                                        Send
                                    </button>
                                </div>
                            </div>
                        </div>
                    )}

                    {message && (
                        <div className="mt-4 p-2 bg-red-100 text-red-700">
                            {message}
                        </div>
                    )}

                    {caseId && (
                        <div className="mt-4">
                            <p className="text-lg font-semibold">Case ID: {caseId}</p>
                            <p className="text-lg">Plaintiff: {parties.plaintiff}</p>
                            <p className="text-lg">Defendant: {parties.defendant}</p>
                        </div>
                    )}

                    <div className="mt-4">
                        {timeline.length > 0 ? (
                            <>
                                <h2 className="text-xl font-bold mb-2">Case Timeline</h2>
                                <div className="relative border-l-4 border-blue-500 pl-4">
                                    {timeline.map((event, index) => (
                                        <div key={index} className={`mb-4 ${event.is_important ? 'important-event' : ''}`}>
                                            <div className="absolute w-3 h-3 bg-blue-500 rounded-full -left-1.5 border border-blue-500"></div>
                                            <p className="font-bold">{event.document_date}</p>
                                            <p className="text-sm">Summary: {event.summary}</p>
                                            <button
                                                onClick={() => toggleImportant(index)}
                                                className={`text-sm px-2 py-1 rounded mt-1 ${event.is_important ? 'bg-red-500 text-white' : 'bg-green-500 text-white'}`}
                                                disabled={loading}
                                            >
                                                {event.is_important ? 'Unmark Important' : 'Mark Important'}
                                            </button>
                                        </div>
                                    ))}
                                </div>

                                {(plaintiffIssues.length > 0 || defendantIssues.length > 0) && (
                                    <>
                                        <h2 className="text-xl font-bold mt-4 mb-2">Case Issues</h2>
                                        <h3 className="text-lg font-semibold">Plaintiff Issues ({parties.plaintiff})</h3>
                                        <ul className="list-disc pl-5">
                                            {plaintiffIssues.map((issue, index) => (
                                                <li key={index}>{issue}</li>
                                            ))}
                                        </ul>
                                        <h3 className="text-lg font-semibold mt-2">Defendant Issues ({parties.defendant})</h3>
                                        <ul className="list-disc pl-5">
                                            {defendantIssues.map((issue, index) => (
                                                <li key={index}>{issue}</li>
                                            ))}
                                        </ul>
                                    </>
                                )}

                                {analysis && (
                                    <>
                                        <h2 className="text-xl font-bold mt-4 mb-2">Case Analysis</h2>
                                        <h3 className="text-lg font-semibold">Mismatches</h3>
                                        <ul className="list-disc pl-5">
                                            {(analysis.mismatches || []).map((mismatch, index) => (
                                                <li key={index}>{mismatch}</li>
                                            ))}
                                        </ul>
                                        <h3 className="text-lg font-semibold">Plaintiff Grounds ({parties.plaintiff})</h3>
                                        <ul className="list-disc pl-5">
                                            {(analysis.plaintiff_grounds || []).map((ground, index) => (
                                                <li key={index}>{ground}</li>
                                            ))}
                                        </ul>
                                        <h3 className="text-lg font-semibold">Defendant Grounds ({parties.defendant})</h3>
                                        <ul className="list-disc pl-5">
                                            {(analysis.defendant_grounds || []).map((ground, index) => (
                                                <li key={index}>{ground}</li>
                                            ))}
                                        </ul>
                                        <h2 className="text-xl font-bold mt-4 mb-2">Document Tags</h2>
                                        <ul className="list-disc pl-5">
                                            {(analysis.document_tags || ['No document tags available']).map((tag, index) => (
                                                <li key={index}>{tag}</li>
                                            ))}
                                        </ul>
                                        <h2 className="text-xl font-bold mt-4 mb-2">Key Events Tags</h2>
                                        <ul className="list-disc pl-5">
                                            {(analysis.key_events_tags || ['No key events tags available']).map((tag, index) => (
                                                <li key={index}>{tag}</li>
                                            ))}
                                        </ul>
                                    </>
                                )}
                            </>
                        ) : (
                            <p>No timeline data available for this case.</p>
                        )}
                    </div>
                </div>
            );
        }

        ReactDOM.render(<App />, document.getElementById('root'));
    </script>
</body>
</html>
"""
    return HTMLResponse(content=html_content)

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)