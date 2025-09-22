import os
import json
import datetime
from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename

# LangChain imports
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_huggingface import HuggingFacePipeline, HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings, OpenAI

# Configuration
from dotenv import load_dotenv
load_dotenv()

# Fallback to environment variables
HUGGING_FACE_HUB_API_TOKEN = os.getenv('HUGGING_FACE_HUB_API_TOKEN')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

# Constants
UPLOAD_FOLDER = 'uploaded_files'
VECTOR_STORE_PATH = "faiss_index"
LOGS_FOLDER = 'conversation_logs'
ALLOWED_EXTENSIONS = {'txt', 'pdf'}

# App Setup
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure required folders exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(LOGS_FOLDER, exist_ok=True)

# Clementina's Personality Template
CLEMENTINA_TEMPLATE = """You are Clementina, a compassionate and knowledgeable maternal health assistant. You have the warmth of a trusted midwife combined with evidence-based medical knowledge.

Your personality:
- Speak with gentle authority and professional warmth
- Use phrases like "I understand this can be concerning" and "Let me help you with that"
- Acknowledge the emotional aspects of pregnancy, childbirth, and parenting
- Be encouraging but realistic about health topics
- Show empathy for the physical and emotional challenges parents face

IMPORTANT MEDICAL DISCLAIMER: Always remind users that your guidance is educational and they should consult their healthcare provider for personalized medical advice, especially for urgent concerns.

Knowledge Base Context:
{context}

Question: {question}

Clementina's caring response:"""

def log_conversation(user_message, bot_response, sources, feedback=None):
    """Log conversations for quality improvement and analysis."""
    log_entry = {
        "timestamp": datetime.datetime.now().isoformat(),
        "user_message": user_message,
        "bot_response": bot_response,
        "sources": sources,
        "feedback": feedback,
        "session_id": request.remote_addr  # Simple session tracking
    }
    
    # Save to daily log file
    date_str = datetime.datetime.now().strftime("%Y-%m-%d")
    log_file = os.path.join(LOGS_FOLDER, f"conversations_{date_str}.jsonl")
    
    with open(log_file, 'a', encoding='utf-8') as f:
        f.write(json.dumps(log_entry) + '\n')

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_llm_and_embeddings(provider: str):
    """Returns the appropriate LLM and embeddings based on the provider."""
    if provider == 'openai':
        if not OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY not set. Please add it to .env or set as environment variable.")
        embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
        llm = OpenAI(temperature=0.3, openai_api_key=OPENAI_API_KEY)  # Slightly more creative for personality
        return llm, embeddings
    elif provider == 'local':
        if not HUGGING_FACE_HUB_API_TOKEN:
            print("⚠️  Warning: HUGGING_FACE_HUB_API_TOKEN not set. Using local model without API.")
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        llm = HuggingFacePipeline.from_model_id(
            model_id="google/flan-t5-large",
            task="text2text-generation",
            pipeline_kwargs={"max_new_tokens": 300, "temperature": 0.3},  # More tokens for personality
        )
        return llm, embeddings
    else:
        raise ValueError(f"Unsupported provider: {provider}. Choose 'openai' or 'local'.")

def create_index(file_path: str, provider: str = 'local'):
    """Loads a document, splits it, creates embeddings, and stores them in FAISS."""
    print(f"Processing medical content: {file_path} with provider: {provider}")
    
    # Choose loader based on file extension
    if file_path.lower().endswith('.pdf'):
        loader = PyPDFLoader(file_path)
    else:
        loader = TextLoader(file_path)
        
    documents = loader.load()
    if not documents:
        raise ValueError("Could not load any documents from the file.")

    # Smaller chunks for medical content to maintain precision
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800, 
        chunk_overlap=150,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    texts = text_splitter.split_documents(documents)
    if not texts:
        raise ValueError("Could not split the document into text chunks.")
        
    print(f"Split medical document into {len(texts)} chunks for precise retrieval.")

    _, embeddings = get_llm_and_embeddings(provider)

    # Load existing index if it exists, otherwise create new
    if os.path.exists(VECTOR_STORE_PATH):
        print("Merging with existing knowledge base...")
        db = FAISS.load_local(VECTOR_STORE_PATH, embeddings, allow_dangerous_deserialization=True)
        db.add_documents(texts)
    else:
        print("Creating new knowledge base...")
        db = FAISS.from_documents(texts, embeddings)
    
    db.save_local(VECTOR_STORE_PATH)
    print(f"Medical knowledge base updated and saved at '{VECTOR_STORE_PATH}'.")

def ask_question(query: str, provider: str = 'local'):
    """Answers a question using the RAG pipeline with Clementina's personality."""
    if not os.path.exists(VECTOR_STORE_PATH):
        return "I don't have access to my knowledge base yet. Please ask an administrator to upload some medical content first so I can help you better.", []

    llm, embeddings = get_llm_and_embeddings(provider)
    db = FAISS.load_local(VECTOR_STORE_PATH, embeddings, allow_dangerous_deserialization=True)
    
    # Use more relevant chunks for comprehensive medical responses
    retriever = db.as_retriever(search_kwargs={"k": 4})

    # Create custom prompt with Clementina's personality
    PROMPT = PromptTemplate(
        template=CLEMENTINA_TEMPLATE,
        input_variables=["context", "question"]
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT}
    )

    result = qa_chain.invoke(query)
    answer = result.get("result", "I'm sorry, I couldn't find information to help with that question.")
    sources = [doc.metadata.get('source', 'N/A') for doc in result.get("source_documents", [])]
    
    # Add medical disclaimer if health-related
    health_keywords = ['pregnancy', 'baby', 'birth', 'pain', 'bleeding', 'symptom', 'medical', 'health', 'doctor', 'medication']
    if any(keyword in query.lower() for keyword in health_keywords):
        disclaimer = "\n\n💗 Remember: This information is for educational purposes only. Please consult with your healthcare provider for personalized medical advice, especially if you have specific concerns or symptoms."
        answer += disclaimer
    
    return answer, sources

# Flask Routes
@app.route('/')
def serve_index():
    """Serves the main HTML page."""
    return render_template('index.html')

@app.route('/ask', methods=['POST'])
def handle_ask():
    """Handles the chat message from the user."""
    data = request.json
    query = data.get('message')
    provider = data.get('provider', 'local')

    if not query:
        return jsonify({"error": "No message provided."}), 400

    try:
        answer, sources = ask_question(query, provider)
        
        # Log the conversation
        log_conversation(query, answer, sources)
        
        return jsonify({
            "answer": answer, 
            "sources": list(set(sources)),
            "conversation_id": datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        })
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return jsonify({"error": "I'm experiencing technical difficulties. Please try again in a moment."}), 500

@app.route('/feedback', methods=['POST'])
def handle_feedback():
    """Handles user feedback on responses."""
    data = request.json
    conversation_id = data.get('conversation_id')
    rating = data.get('rating')  # 1-5 stars
    feedback_text = data.get('feedback', '')
    
    # Log feedback
    feedback_entry = {
        "timestamp": datetime.datetime.now().isoformat(),
        "conversation_id": conversation_id,
        "rating": rating,
        "feedback_text": feedback_text,
        "session_id": request.remote_addr
    }
    
    date_str = datetime.datetime.now().strftime("%Y-%m-%d")
    feedback_file = os.path.join(LOGS_FOLDER, f"feedback_{date_str}.jsonl")
    
    with open(feedback_file, 'a', encoding='utf-8') as f:
        f.write(json.dumps(feedback_entry) + '\n')
    
    return jsonify({"success": "Thank you for your feedback! It helps me improve."})

@app.route('/upload', methods=['POST'])
def handle_upload():
    """Handles file uploads to build the medical knowledge base."""
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

    return jsonify({"error": "File type not supported. Please upload .txt or .pdf files only."}), 400

@app.route('/analytics')
def get_analytics():
    """Provides basic analytics on conversations and feedback."""
    # This would be expanded for a full admin dashboard
    return jsonify({
        "total_conversations": "Available in conversation logs",
        "average_rating": "Available in feedback logs",
        "common_topics": "Available through log analysis"
    })

if __name__ == '__main__':
    print("🌸 Clementina Health Assistant starting up...")
    print("📁 Conversation logging enabled")
    print("⭐ Feedback system active")
    app.run(host='0.0.0.0', port=8080, debug=True)