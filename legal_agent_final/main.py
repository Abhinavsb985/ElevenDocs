import os
import re
import base64
import logging
from flask import Flask, request, jsonify
from flask_cors import CORS

# --- LangChain / VertexAI ---
from langchain_google_vertexai import ChatVertexAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.agents import tool, AgentExecutor, create_react_agent
from langchain import hub
from langchain.memory import ConversationBufferMemory

# --- Google Cloud Clients ---
from google.cloud import vision
from google.cloud import documentai

# ================== INIT ==================
app = Flask(__name__)
CORS(app)
logging.basicConfig(level=logging.DEBUG)

# ---------- Vertex AI init (requires ADC credentials and project/location) ----------
try:
    # Prefer explicit init to avoid relying on gcloud defaults
    from google.cloud import aiplatform

    VERTEX_PROJECT = os.getenv("GOOGLE_PROJECT_ID") or os.getenv("VERTEXAI_PROJECT_ID")
    VERTEX_LOCATION = os.getenv("VERTEXAI_LOCATION", "us-central1")
    if not VERTEX_PROJECT:
        raise RuntimeError(
            "Missing GOOGLE_PROJECT_ID or VERTEXAI_PROJECT_ID env var for Vertex AI."
        )

    aiplatform.init(project=VERTEX_PROJECT, location=VERTEX_LOCATION)
except Exception as e:
    # Fail fast with clear message
    raise RuntimeError(
        f"Vertex AI initialization failed: {e}. Ensure GOOGLE_APPLICATION_CREDENTIALS points to a valid service account JSON and set GOOGLE_PROJECT_ID/ VERTEXAI_PROJECT_ID."
    )

# VertexAI Model
llm = ChatVertexAI(model_name="gemini-2.0-flash", temperature=0.5)

# Memory
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Last doc memory
last_document_text = None
last_document_summary = None

# Agent policy
AGENT_POLICY = (
    "You are a friendly legal AI assistant. "
    "If a file is uploaded, summarize it. "
    "If text is pasted, summarize it. "
    "If the user asks about the last document, answer from it. "
    "Always end answers with: 'Disclaimer: I am an AI assistant, not a lawyer.'"
)

# ================== PROMPTS ==================
summarization_prompt = ChatPromptTemplate.from_template("""
You are a legal AI assistant. Explain the document clearly in plain English.

- Start: 'Here’s what this document means in simple terms:'
- Summarize the purpose.
- List obligations, risks, key dates.
- End with one-line conclusion.
{document_text}
""")

summarization_chain = summarization_prompt | llm | StrOutputParser()

qa_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful AI legal assistant. Answer ONLY from the document."),
    ("human", "Document:\n{document_text}\n\nQuestion:\n{question}")
])
qa_chain = qa_prompt | llm | StrOutputParser()

# ================== HELPERS ==================
def extract_text_from_image(image_bytes: bytes) -> str:
    """OCR for images with Vision API."""
    client = vision.ImageAnnotatorClient()
    image = vision.Image(content=image_bytes)
    response = client.document_text_detection(image=image)

    if response.error.message:
        raise Exception(response.error.message)

    return response.full_text_annotation.text


def extract_text_from_pdf(pdf_bytes: bytes) -> str:
    """OCR for PDFs with Document AI. Falls back to PyPDF2 if Document AI fails."""
    try:
        client = documentai.DocumentProcessorServiceClient()
        project_id = os.getenv("GOOGLE_PROJECT_ID")
        processor_id = os.getenv("DOCUMENT_AI_PROCESSOR_ID")  # must exist in GCP
        location = "us"
        name = f"projects/{project_id}/locations/{location}/processors/{processor_id}"
        raw_document = documentai.RawDocument(content=pdf_bytes, mime_type="application/pdf")
        request = documentai.ProcessRequest(name=name, raw_document=raw_document)
        result = client.process_document(request=request)
        text = result.document.text
        if text:
            return text
    except Exception as e:
        logging.error(f"Document AI PDF OCR failed: {e}")
        # Fallback to PyPDF2
        try:
            from PyPDF2 import PdfReader
            import io
            pdf_file = io.BytesIO(pdf_bytes)
            reader = PdfReader(pdf_file)
            text = ""
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
            return text
        except Exception as e2:
            logging.error(f"PyPDF2 fallback failed: {e2}")
            return ""


def clean_text(text: str) -> str:
    """Remove borders, symbols, normalize spaces."""
    text = re.sub(r'[-=*_~]{3,}', ' ', text)
    text = re.sub(r'[^\w\s.,;:()%-]', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()


# ================== TOOLS ==================
@tool
def process_and_summarize_document(input_dict: dict) -> str:
    """Process an uploaded document (PDF/Image/TXT)."""
    global last_document_text, last_document_summary
    try:
        file_data = input_dict.get("file_data")
        file_type = input_dict.get("file_type", "txt")
        decoded = base64.b64decode(file_data)

        if file_type == "pdf":
            text = extract_text_from_pdf(decoded)
        elif file_type in ["png", "jpeg", "jpg"]:
            text = extract_text_from_image(decoded)
        else:
            text = decoded.decode("utf-8")

        text = clean_text(text)

        if not text:
            return "❌ Couldn’t extract text."

        summary = summarization_chain.invoke({"document_text": text})
        last_document_text, last_document_summary = text, summary
        return summary

    except Exception as e:
        return f"⚠️ Error processing document: {str(e)}"


@tool
def summarize_pasted_document(text: str) -> str:
    """Summarize pasted text in simple language."""
    global last_document_text, last_document_summary
    summary = summarization_chain.invoke({"document_text": text})
    last_document_text, last_document_summary = text, summary
    return summary


@tool
def qa_on_last_document(question: str) -> str:
    """Answer questions about the last uploaded or pasted document."""
    if not last_document_text:
        return "⚠️ No document available yet."
    return qa_chain.invoke({"document_text": last_document_text, "question": question})


@tool
def answer_general_question(question: str) -> str:
    """Answer general legal questions in a friendly way."""
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a supportive AI legal assistant. "
                   "Keep answers clear and end with a disclaimer."),
        ("human", "{question}")
    ])
    chain = prompt | llm | StrOutputParser()
    return chain.invoke({"question": question})


# ================== AGENT ==================
tools = [
    process_and_summarize_document,
    summarize_pasted_document,
    qa_on_last_document,
    answer_general_question
]

agent_prompt = hub.pull("hwchase17/react")
agent = create_react_agent(llm, tools, agent_prompt)

agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    handle_parsing_errors=True,
    memory=memory
)

# ================== API ==================
def run_agent(user_input, file_data_b64=None, file_type=None):
    try:
        if file_data_b64 and file_type:
            return process_and_summarize_document.invoke({
                "input_dict": {"file_data": file_data_b64, "file_type": file_type}
            })
        else:
            response = agent_executor.invoke({"input": f"{AGENT_POLICY}\n\nUser: {user_input}"})
            return response.get("output", "⚠️ No output.")
    except Exception as e:
        return f"⚠️ Error: {str(e)}"


@app.route("/invoke", methods=["POST"])
def invoke():
    try:
        if request.is_json:
            data = request.get_json()
            user_input = data.get("text")
            if not user_input:
                return jsonify({"error": "Provide 'text'"}), 400
            return jsonify({"response": run_agent(user_input)})

        file = request.files.get('file_data')
        if not file:
            return jsonify({"error": "No file uploaded"}), 400

        user_input = request.form.get("input", "")
        file_type = request.form.get("file_type", file.filename.split('.')[-1].lower())
        file_b64 = base64.b64encode(file.read()).decode("utf-8")

        return jsonify({"output": run_agent(user_input, file_data_b64=file_b64, file_type=file_type)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/", methods=["GET"])
def home():
    return jsonify({"status": "✅ Legal AI API running", "endpoints": ["/invoke"]})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8002)), debug=True)
