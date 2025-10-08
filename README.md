# Clementina Health RAG Chatbot

A Retrieval-Augmented Generation (RAG) chatbot designed for maternal and family wellness support.

## Setup Instructions

### 1. Clone and Install Dependencies

```bash
git clone <your-repo-url>
cd <your-repo-name>
pip install -r requirements.txt
```

### 2. Set up API Keys

This application uses a `.env` file to manage API keys securely.

1.  Create your local environment file by copying the example:
    ```bash
    cp .env.example .env
    ```

2.  Open the `.env` file and add your API key(s):
    -   `HUGGING_FACE_HUB_API_TOKEN`: (Optional) Needed for the default local models. Get a token from [HuggingFace Settings](https://huggingface.co/settings/tokens).
    -   `OPENAI_API_KEY`: (Optional) Needed if you want to use OpenAI models. Get a key from [OpenAI Platform](https://platform.openai.com/api-keys).

### 3. Create Required Directories

```bash
mkdir templates
mkdir uploaded_files
mv index.html templates/  # Move HTML file to templates folder
```

### 4. Run the Application

```bash
python app.py
```

Visit `http://localhost:8080` in your browser.

## Usage

### Web Interface
- Open the web interface and click the chat button
- Upload documents through the admin panel to build the knowledge base
- Ask questions related to the uploaded content

### Command Line (Optional)
```bash
# Add a document to the knowledge base
python rag_chatbot.py --add sample_medical_content.txt

# Ask a question
python rag_chatbot.py --ask "What helps with morning sickness?"
```

## Features

- 🤖 **Dual Provider Support**: OpenAI and local HuggingFace models
- 📁 **Multi-format Support**: Text files (.txt), PDFs (.pdf), and Word Documents (.docx)
- 🎨 **Professional UI**: Responsive design with health-focused branding
- 🔒 **Secure**: API keys stored locally, not in repository
- 📊 **Admin Panel**: Upload files and manage knowledge base

## File Structure

```
├── app.py                 # Main Flask application
├── rag_chatbot.py        # Command-line interface
├── requirements.txt      # Python dependencies
├── .env.example         # Template for API keys
├── .env                 # Your actual API keys (gitignored)
├── templates/
│   └── index.html       # Web interface
├── uploaded_files/      # Uploaded documents (gitignored)
└── faiss_index/        # Vector store (gitignored)
```

## Security Notes

- Never commit `config.py` to version control
- API keys are loaded from local files or environment variables
- The `.gitignore` file prevents accidental key exposure