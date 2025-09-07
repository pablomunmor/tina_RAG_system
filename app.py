import os
from flask import Flask, request, jsonify, render_template, send_from_directory
from werkzeug.utils import secure_filename

# Correct, non-deprecated imports
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA

# Provider-specific imports
from langchain_huggingface import HuggingFacePipeline, HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings, OpenAI

# --- Secrets ---
# For simplicity, we'll import the key from a local file.
# This makes it easier for the user than setting environment variables.
try:
    from secrets import HUGGING_FACE_HUB_API_TOKEN, OPENAI_API_KEY
except ImportError:
    HUGGING_FACE_HUB_API_TOKEN = None
    OPENAI_API_KEY = None

# --- Constants ---
UPLOAD_FOLDER = 'uploaded_files'
VECTOR_STORE_PATH = "faiss_index"
ALLOWED_EXTENSIONS = {'txt', 'pdf'}

# --- App Setup ---
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure the upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


# --- Helper Functions ---
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_llm_and_embeddings(provider: str):
    """Returns the appropriate LLM and embeddings based on the provider."""
    if provider == 'openai':
        if not OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY not set in secrets.py for 'openai' provider.")
        embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
        llm = OpenAI(temperature=0.2, openai_api_key=OPENAI_API_KEY)
        return llm, embeddings
    elif provider == 'local':
        if not HUGGING_FACE_HUB_API_TOKEN:
            raise ValueError("HUGGINGFACEHUB_API_TOKEN not set in secrets.py for 'local' provider.")
        # Using a popular open-source model for embeddings
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        # Using a popular open-source model for generation from the Hub
        # The HuggingFacePipeline is generally more robust for self-hosting
        llm = HuggingFacePipeline.from_model_id(
            model_id="google/flan-t5-large",
            task="text2text-generation",
            pipeline_kwargs={"max_new_tokens": 256},
        )
        return llm, embeddings
    else:
        raise ValueError(f"Unsupported provider: {provider}. Choose 'openai' or 'local'.")


# --- Core RAG Logic ---
def create_index(file_path: str, provider: str = 'local'):
    """
    Loads a document, splits it, creates embeddings, and stores them in FAISS.
    """
    print(f"Processing file: {file_path} with provider: {provider}")
    
    # Choose loader based on file extension
    if file_path.lower().endswith('.pdf'):
        loader = PyPDFLoader(file_path)
    else:
        loader = TextLoader(file_path)
        
    documents = loader.load()
    if not documents:
        raise ValueError("Could not load any documents from the file.")

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    texts = text_splitter.split_documents(documents)
    if not texts:
        raise ValueError("Could not split the document into text chunks.")
        
    print(f"Split document into {len(texts)} chunks.")

    _, embeddings = get_llm_and_embeddings(provider)

    # For simplicity, we create a new index for each upload.
    # A more advanced implementation would merge them.
    db = FAISS.from_documents(texts, embeddings)
    db.save_local(VECTOR_STORE_PATH)
    print(f"Vector store created/updated and saved at '{VECTOR_STORE_PATH}'.")


def ask_question(query: str, provider: str = 'local'):
    """
    Answers a question using the RAG pipeline.
    """
    if not os.path.exists(VECTOR_STORE_PATH):
        return "The knowledge base is empty. Please upload a document first through the admin panel.", []

    llm, embeddings = get_llm_and_embeddings(provider)

    db = FAISS.load_local(VECTOR_STORE_PATH, embeddings, allow_dangerous_deserialization=True)
    retriever = db.as_retriever()

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True
    )

    # The .invoke method for this chain expects the query string directly.
    result = qa_chain.invoke(query)
    answer = result.get("result", "Sorry, I could not find an answer.")
    sources = [doc.metadata.get('source', 'N/A') for doc in result.get("source_documents", [])]
    
    return answer, sources


# --- Flask Routes ---

@app.route('/')
def serve_index():
    """Serves the main HTML page."""
    return render_template('index.html')

@app.route('/ask', methods=['POST'])
def handle_ask():
    """Handles the chat message from the user."""
    data = request.json
    query = data.get('message')
    # Provider can be passed from the frontend in a real scenario
    # For now, we'll default to 'local'
    provider = data.get('provider', 'local') 

    if not query:
        return jsonify({"error": "No message provided."}), 400

    try:
        answer, sources = ask_question(query, provider)
        return jsonify({"answer": answer, "sources": list(set(sources))})
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return jsonify({"error": "An internal error occurred."}), 500


@app.route('/upload', methods=['POST'])
def handle_upload():
    """Handles file uploads to build the knowledge base."""
    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request."}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({"error": "No selected file."}), 400
        
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Provider can be passed from the frontend
        provider = request.form.get('provider', 'local')
        
        try:
            create_index(filepath, provider)
            return jsonify({"success": f"File '{filename}' uploaded and indexed successfully."})
        except Exception as e:
            return jsonify({"error": f"Failed to process file: {e}"}), 500

    return jsonify({"error": "File type not allowed."}), 400


if __name__ == '__main__':
    # Note: Using debug=True is not recommended for production
    app.run(host='0.0.0.0', port=8080, debug=True)
