import os
import argparse

# Updated imports to match app.py
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_openai import OpenAIEmbeddings, OpenAI

# Import from secrets file
try:
    from secrets import OPENAI_API_KEY
except ImportError:
    OPENAI_API_KEY = None

# --- Constants ---
VECTOR_STORE_PATH = "faiss_index"

def check_openai_api_key():
    """Check if the OpenAI API key is set."""
    if not OPENAI_API_KEY:
        print("Error: The OPENAI_API_KEY is not set in config.py.")
        print("Please add your key to config.py and try again.")
        exit(1)

def create_index(file_path: str):
    """
    Loads a text file, splits it into chunks, creates embeddings,
    and stores them in a FAISS vector store.
    It merges with an existing index if one is found.
    """
    print(f"Processing file: {file_path}")
    
    # Load the document
    loader = TextLoader(file_path)
    documents = loader.load()
    if not documents:
        print("Could not load any documents from the file. Exiting.")
        return

    # Split the document into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    texts = text_splitter.split_documents(documents)
    if not texts:
        print("Could not split the document into text chunks. Exiting.")
        return
        
    print(f"Split document into {len(texts)} chunks.")

    # Create embeddings
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

    # Create or load the FAISS vector store
    if os.path.exists(VECTOR_STORE_PATH):
        print(f"Existing vector store found at '{VECTOR_STORE_PATH}'. Merging new documents.")
        try:
            db = FAISS.load_local(VECTOR_STORE_PATH, embeddings, allow_dangerous_deserialization=True)
            db.add_documents(texts)
        except Exception as e:
            print(f"Error loading or merging with the existing vector store: {e}")
            return
    else:
        print(f"No existing vector store found. Creating a new one at '{VECTOR_STORE_PATH}'.")
        db = FAISS.from_documents(texts, embeddings)

    # Save the vector store
    db.save_local(VECTOR_STORE_PATH)
    print(f"Vector store updated and saved successfully at '{VECTOR_STORE_PATH}'.")


def ask_question(query: str):
    """
    Loads the vector store and answers a question using the RetrievalQA chain.
    """
    if not os.path.exists(VECTOR_STORE_PATH):
        print(f"Error: Vector store not found at '{VECTOR_STORE_PATH}'.")
        print("Please add a document first using the --add option.")
        return

    # Load the vector store
    try:
        embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
        db = FAISS.load_local(VECTOR_STORE_PATH, embeddings, allow_dangerous_deserialization=True)
    except Exception as e:
        print(f"Error loading the vector store: {e}")
        return

    # Create the retriever
    retriever = db.as_retriever()

    # Create the RetrievalQA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=OpenAI(temperature=0.2, openai_api_key=OPENAI_API_KEY),
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True
    )

    # Ask the question - Updated to use .invoke() method
    result = qa_chain.invoke({"query": query})
    
    print("\n--- Answer ---")
    print(result["result"])
    print("\n--- Source Documents ---")
    for doc in result["source_documents"]:
        print(f"Source: {doc.metadata.get('source', 'N/A')}")
        # print(f"Content: {doc.page_content[:200]}...") # Optional: print snippet of source
    print("\n")


def main():
    """Main function to handle argument parsing and execution."""
    # First, ensure the API key is available
    check_openai_api_key()

    parser = argparse.ArgumentParser(description="A simple RAG chatbot.")
    
    # Use a mutually exclusive group to ensure only one action is performed at a time
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--add", metavar="FILE_PATH", type=str, help="Add a new text file to the knowledge base.")
    group.add_argument("--ask", metavar="QUESTION", type=str, help="Ask a question to the chatbot.")

    args = parser.parse_args()

    if args.add:
        if os.path.isfile(args.add):
            create_index(args.add)
        else:
            print(f"Error: The file '{args.add}' does not exist.")
    elif args.ask:
        ask_question(args.ask)


if __name__ == "__main__":
    main()