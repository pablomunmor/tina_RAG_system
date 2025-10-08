import os
import json
import datetime
from flask import Flask, request, jsonify, render_template, session
from werkzeug.utils import secure_filename

# LangChain imports
from langchain_community.document_loaders import TextLoader, PyPDFLoader, UnstructuredWordDocumentLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain_huggingface import HuggingFacePipeline, HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings, OpenAI

# Configuration
from dotenv import load_dotenv
load_dotenv()

# Fallback to environment variables
HUGGING_FACE_HUB_API_TOKEN = os.getenv('HUGGING_FACE_HUB_API_TOKEN')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

# Storage configuration for persistent data
# On platforms like Render, set STORAGE_DIR to a mounted disk path (e.g., /var/data)
STORAGE_DIR = os.getenv('STORAGE_DIR')

if STORAGE_DIR:
    # Use persistent storage for uploads and the vector store
    UPLOAD_FOLDER = os.path.join(STORAGE_DIR, 'uploaded_files')
    VECTOR_STORE_PATH = os.path.join(STORAGE_DIR, "faiss_index")
else:
    # Use local folders for local development
    UPLOAD_FOLDER = 'uploaded_files'
    VECTOR_STORE_PATH = "faiss_index"

# Constants
LOGS_FOLDER = 'conversation_logs'
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'docx'}

# App Setup
app = Flask(__name__)
app.secret_key = 'clementina_health_2024'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure required folders exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(LOGS_FOLDER, exist_ok=True)
if STORAGE_DIR:
    os.makedirs(VECTOR_STORE_PATH, exist_ok=True)

# Clementina's Personality Template
CLEMENTINA_TEMPLATE = """You are Tina (short for Clementina), a compassionate and knowledgeable maternal health assistant. You have the warmth of a trusted midwife combined with evidence-based medical knowledge.


Your personality and rules:
- Be warm, gentle, and encouraging. Use a soft, caring tone.
- Summarize and explain information in your own words. Do not quote directly from the source material.
- **CRITICAL RULE: Your main response MUST be under 100 characters.**
- **CRITICAL RULE: Directly answer the user's question. Do not provide related, but irrelevant information.**
- If a user's name is provided but it sounds like a health topic (e.g., "Sore Nipples"), gently ignore the name and answer the question directly.

Communication style:
- Keep responses brief and conversational (1-2 sentences maximum).
- Focus on the most important, practical information first.
- After answering, if it seems helpful, ask a single, relevant follow-up question. For example, if the user asks about feeding, you could ask, "Would you like some tips on how to tell if the baby is latched on correctly?"

IMPORTANT MEDICAL DISCLAIMER: Always remind users that your guidance is educational and they should consult their healthcare provider for personalized medical advice, especially for urgent concerns.

Knowledge Base Context:
{context}

Question: {question}


Tina's caring response:"""


def log_conversation(user_message, bot_response, sources, feedback=None, user_name=None):
    """Log conversations for quality improvement and analysis."""
    log_entry = {
        "timestamp": datetime.datetime.now().isoformat(),
        "user_name": user_name,
        "user_message": user_message,
        "bot_response": bot_response if isinstance(bot_response, str) else " ".join(bot_response),
        "sources": sources,
        "feedback": feedback,
        "session_id": request.remote_addr
    }
    
    try:
        date_str = datetime.datetime.now().strftime("%Y-%m-%d")
        log_file = os.path.join(LOGS_FOLDER, f"conversations_{date_str}.jsonl")
        
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(log_entry) + '\n')
    except Exception as e:
        print(f"Error logging conversation: {e}")

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_llm_and_embeddings(provider: str):
    """Returns the appropriate LLM and embeddings based on the provider."""
    if provider == 'openai':
        if not OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY not set. Please add it to .env or set as environment variable.")
        embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
        llm = OpenAI(temperature=0.3, openai_api_key=OPENAI_API_KEY)
        return llm, embeddings
    elif provider == 'local':
        if not HUGGING_FACE_HUB_API_TOKEN:
            print("⚠️  Warning: HUGGING_FACE_HUB_API_TOKEN not set. Using local model without API.")
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        llm = HuggingFacePipeline.from_model_id(
            model_id="google/flan-t5-large",
            task="text2text-generation",
            pipeline_kwargs={"max_new_tokens": 300, "temperature": 0.3},
        )
        return llm, embeddings
    else:
        raise ValueError(f"Unsupported provider: {provider}. Choose 'openai' or 'local'.")

def create_index(file_path: str, provider: str = 'local'):
    """Loads a document, splits it, creates embeddings, and stores them in FAISS."""
    print(f"Processing medical content: {file_path} with provider: {provider}")
    
    if file_path.lower().endswith('.pdf'):
        loader = PyPDFLoader(file_path)
    elif file_path.lower().endswith('.docx'):
        loader = UnstructuredWordDocumentLoader(file_path)
    else:
        loader = TextLoader(file_path)
        
    documents = loader.load()
    if not documents:
        raise ValueError("Could not load any documents from the file.")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=50,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    texts = text_splitter.split_documents(documents)
    if not texts:
        raise ValueError("Could not split the document into text chunks.")
        
    print(f"Split medical document into {len(texts)} chunks for precise retrieval.")

    _, embeddings = get_llm_and_embeddings(provider)

    if os.path.exists(VECTOR_STORE_PATH):
        print("Merging with existing knowledge base...")
        # SECURITY NOTE: allow_dangerous_deserialization is set to True because we are loading
        # a FAISS index from a local file. This is safe in this context because the file is
        # generated by the application itself. In a production environment where the index
        # might come from an untrusted source, this should be handled with more care.
        db = FAISS.load_local(VECTOR_STORE_PATH, embeddings, allow_dangerous_deserialization=True)
        db.add_documents(texts)
    else:
        print("Creating new knowledge base...")
        db = FAISS.from_documents(texts, embeddings)
    
    db.save_local(VECTOR_STORE_PATH)
    print(f"Medical knowledge base updated and saved at '{VECTOR_STORE_PATH}'.")

def initialize_faiss_index():
    """Create an empty FAISS index if one doesn't exist."""
    if not os.path.exists(VECTOR_STORE_PATH) or not os.path.exists(os.path.join(VECTOR_STORE_PATH, "index.faiss")):
        print("Creating an empty FAISS index.")
        _, embeddings = get_llm_and_embeddings('local')
        # Create an empty index with some dummy data
        db = FAISS.from_texts(["Clementina is ready to help."], embeddings)
        db.save_local(VECTOR_STORE_PATH)
        print("Empty FAISS index created.")

def ask_question(query: str, provider: str = 'local', user_name: str = None, chat_history=None):
    """Answers a question using the RAG pipeline with Clementina's personality and conversational memory."""
    if chat_history is None:
        chat_history = []

    if not os.path.exists(VECTOR_STORE_PATH):
        return ["I don't have access to my knowledge base yet. Please ask an administrator to upload some medical content first so I can help you better."], []

    llm, embeddings = get_llm_and_embeddings(provider)
    db = FAISS.load_local(VECTOR_STORE_PATH, embeddings, allow_dangerous_deserialization=True)
    retriever = db.as_retriever(search_kwargs={"k": 2})

    # This prompt is to rephrase the follow-up question to be a standalone question
    condense_question_template = """Given the following conversation and a follow-up question, rephrase the follow-up question to be a standalone question, in its original language.

Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question:"""
    CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(condense_question_template)

    # This is the main prompt for answering the question
    qa_prompt = PromptTemplate(
        template=CLEMENTINA_TEMPLATE,
        input_variables=["context", "question"]
    )

    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        condense_question_prompt=CONDENSE_QUESTION_PROMPT,
        combine_docs_chain_kwargs={"prompt": qa_prompt},
        return_source_documents=True,
    )

    result = qa_chain.invoke({"question": query, "chat_history": chat_history})
    answer = result.get("answer", "I'm sorry, I couldn't find information to help with that question.")
    sources = [doc.metadata.get('source', 'N/A') for doc in result.get("source_documents", [])]
    
    health_keywords = ['pregnancy', 'baby', 'birth', 'pain', 'bleeding', 'symptom', 'medical', 'health', 'doctor', 'medication']
    if any(keyword in query.lower() for keyword in health_keywords):
        disclaimer = "\n\n💗 Remember: This information is for educational purposes only. Please consult with your healthcare provider for personalized medical advice, especially if you have specific concerns or symptoms."
        answer += disclaimer
    
    answer_parts = [p.strip() for p in answer.split("\n\n") if p.strip()]

    if user_name and not session.get('greeted'):
        answer_parts[0] = f"Hi {user_name}! {answer_parts[0]}"
        session['greeted'] = True

    return answer_parts, sources

# Flask Routes
@app.route('/')
def serve_index():
    return render_template('index.html')

@app.route('/set_name', methods=['POST'])
def set_name():
    """Store user's name in session."""
    data = request.json
    name = data.get('name', '').strip()
    if name:
        session['user_name'] = name
        session['greeted'] = False  # Reset greeted flag for new user
        session['chat_history'] = []  # Start a fresh chat history
        return jsonify({"success": True, "message": f"Nice to meet you, {name}!"})
    return jsonify({"success": False, "message": "Please provide a valid name."})

@app.route('/ask', methods=['POST'])
def handle_ask():
    data = request.json
    query = data.get('message')
    provider = data.get('provider', 'local')
    user_name = session.get('user_name')

    if not query:
        return jsonify({"error": "No message provided."}), 400

    chat_history = session.get('chat_history', [])

    try:
        answer_parts, sources = ask_question(query, provider, user_name, chat_history)
        
        # Update chat history
        bot_response = " ".join(answer_parts)
        chat_history.append((query, bot_response))
        session['chat_history'] = chat_history

        # Log the conversation
        log_conversation(query, answer_parts, sources, user_name=user_name)
        
        conversation_id = f"{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}_{abs(hash(query)) % 10000}"
        
        return jsonify({
            "answer_parts": answer_parts,
            "sources": list(set(sources)),
            "conversation_id": conversation_id
        })
    except Exception as e:
        print(f"Error in handle_ask: {e}")
        return jsonify({
            "answer_parts": ["I'm sorry, I'm having a little trouble right now. Please try again in a moment."],
            "sources": ["Clementina's Care"],
            "conversation_id": f"fallback_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        })

@app.route('/feedback', methods=['POST'])
def handle_feedback():
    data = request.json
    conversation_id = data.get('conversation_id')
    rating = data.get('rating')
    feedback_text = data.get('feedback', '')
    user_name = session.get('user_name')
    
    feedback_entry = {
        "timestamp": datetime.datetime.now().isoformat(),
        "conversation_id": conversation_id,
        "rating": rating,
        "feedback_text": feedback_text,
        "user_name": user_name,
        "session_id": request.remote_addr
    }
    
    try:
        date_str = datetime.datetime.now().strftime("%Y-%m-%d")
        feedback_file = os.path.join(LOGS_FOLDER, f"feedback_{date_str}.jsonl")
        
        with open(feedback_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(feedback_entry) + '\n')
        
        return jsonify({"success": "Thank you for your feedback! It helps me improve."})
    except Exception as e:
        print(f"Error logging feedback: {e}")
        return jsonify({"success": "Thank you for your feedback!"})

@app.route('/analytics', methods=['GET'])
def get_analytics():
    """Get basic analytics from conversation logs."""
    try:
        analytics_data = {
            "total_conversations": 0,
            "total_feedback": 0,
            "average_rating": 0,
            "recent_topics": [],
            "user_names": []
        }
        
        # Read recent conversation logs
        import glob
        conversation_files = glob.glob(os.path.join(LOGS_FOLDER, "conversations_*.jsonl"))
        feedback_files = glob.glob(os.path.join(LOGS_FOLDER, "feedback_*.jsonl"))
        
        # Process conversations
        for file_path in conversation_files[-7:]:  # Last 7 days
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        if line.strip():
                            conv = json.loads(line)
                            analytics_data["total_conversations"] += 1
                            if conv.get('user_name'):
                                analytics_data["user_names"].append(conv['user_name'])
            except Exception as e:
                print(f"Error reading {file_path}: {e}")
        
        # Process feedback
        ratings = []
        for file_path in feedback_files[-7:]:  # Last 7 days
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        if line.strip():
                            fb = json.loads(line)
                            analytics_data["total_feedback"] += 1
                            if fb.get('rating'):
                                ratings.append(fb['rating'])
            except Exception as e:
                print(f"Error reading {file_path}: {e}")
        
        if ratings:
            analytics_data["average_rating"] = round(sum(ratings) / len(ratings), 2)
        
        # Unique users
        analytics_data["unique_users"] = len(set(analytics_data["user_names"]))
        analytics_data["user_names"] = list(set(analytics_data["user_names"]))
        
        return jsonify(analytics_data)
        
    except Exception as e:
        print(f"Analytics error: {e}")
        return jsonify({"error": "Could not load analytics"}), 500

@app.route('/upload', methods=['POST'])
def handle_upload():
    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request."}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({"error": "No selected file."}), 400
        
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        provider = request.form.get('provider', 'local')

        try:
            create_index(filepath, provider)
            return jsonify({"success": f"Medical content '{filename}' has been reviewed and added to my knowledge base successfully."})
        except Exception as e:
            return jsonify({"error": f"Failed to process medical content: {e}"}), 500

    return jsonify({"error": "File type not supported. Please upload .txt, .pdf, or .docx files only."}), 400

if __name__ == '__main__':
    initialize_faiss_index()
    print("🌸 Clementina Health Assistant starting up...")
    print("💬 Conversational tone enabled")
    print("👤 Name collection in main input")
    print("📊 Analytics system active")
    print("📁 Conversation logging enabled")
    app.run(host='0.0.0.0', port=8080, debug=True)
