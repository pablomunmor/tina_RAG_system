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

## Deployment

This application is ready to be deployed to any hosting service that supports Python (e.g., Render, Heroku).

### Start Command
When deploying, you will need to provide a start command. Use the following command to run the application with the Gunicorn production server. The `--timeout 120` flag is recommended to prevent timeouts during large file uploads or initial model downloads.

```bash
gunicorn --timeout 120 app:app
```

### Persistent Storage (for Render, etc.)
To ensure that your uploaded files and the chatbot's knowledge base (the vector store) are not lost when the server restarts or redeploys, you must use persistent storage.

On a platform like **Render**, follow these steps:

1.  **Add a Disk**: In your service's dashboard, go to the "Disks" section and add a new disk.
    -   **Name**: `clementina-data` (or any name you prefer)
    -   **Mount Path**: `/var/data`
    -   **Size**: 1GB is a good starting point.

2.  **Set the Environment Variable**: Go to the "Environment" section and add the following environment variable:
    -   **Key**: `STORAGE_DIR`
    -   **Value**: `/var/data` (this must match the Mount Path from the previous step).

This configuration tells the application to save all persistent data to the attached disk, ensuring Tina's memory is preserved.

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

## Security

### Password Protection
To secure the web interface, you can enable password protection by setting the following environment variables in your `.env` file or in your deployment service's settings:

-   `APP_USER`: The username for login (e.g., `admin`).
-   `APP_PASSWORD`: The password for login.

If both variables are set, any visitor to the site will be prompted for a username and password before they can access any page.

### General Notes
- Never commit your `.env` file to version control.
- API keys are loaded from local files or environment variables.
- The `.gitignore` file prevents accidental key exposure.
