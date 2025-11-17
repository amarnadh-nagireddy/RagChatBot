 Document Intelligence Platform
A Retrieval-Augmented Generation (RAG) application built with Streamlit and LangChain. This tool allows you to upload PDF documents, index them into a persistent vectorstore (ChromaDB), and engage in a conversational Q&A session about their content using the Groq LPU.

(Suggestion: Replace the image above with a screenshot of your running application!)

🚀 Features
PDF Upload: Upload one or multiple PDF files at once.

Persistent Vectorstore: Embeddings are stored in a local ChromaDB instance in the chroma_persist directory.

Fast Indexing: Creates vector embeddings for uploaded documents.

Conversational Chat: A RAG chain answers questions based on the document context, complete with chat history.

Source Citing: The assistant's answers cite the page numbers and source documents.

Session Management: Clear chat history or delete embeddings associated with the current session.

Full Vectorstore Reset: A button to completely clear the vectorstore and metadata.

🛠️ Tech Stack
Frontend: Streamlit

LLM: Groq (via langchain-groq)

Embeddings: HuggingFace all-MiniLM-L6-v2 (via langchain-huggingface)

Vectorstore: ChromaDB

Orchestration: LangChain

PDF Processing: pypdf (via langchain-community)

⚙️ Setup & Installation
Follow these steps to get the application running on your local machine.

1. Clone the Repository
Clone this repository to your local machine:

git clone [https://github.com/your-username/your-repo-name.git](https://github.com/your-username/your-repo-name.git)
cd your-repo-name

2. Create a Virtual Environment
It's highly recommended to use a virtual environment to manage dependencies.

On macOS/Linux:

python3 -m venv venv
source venv/bin/activate

On Windows (cmd):

python -m venv venv
.\venv\Scripts\activate

3. Install Dependencies
Install all the required Python packages using the requirements.txt file:

pip install -r requirements.txt

4. Configure Environment Variables
This project requires a Groq API key to function. You may also need a HuggingFace token for certain embedding models, though all-MiniLM-L6-v2 is often accessible without one.

Create a file named .env in the root of the project directory.

Add your API keys to this file. Do not commit this file to GitHub!

Your .env file should look like this:

# .env
GROQ_API_KEY="your-groq-api-key-here"
HF_TOKEN="your-huggingface-token-here"

Get your Groq API key from GroqCloud.

Get your HuggingFace token from HuggingFace Settings.

▶️ How to Run
Once you have completed the setup, run the Streamlit application with the following command:

streamlit run app.py

The application will open in your default web browser.

📁 Project Structure
.
├── app.py              # The main Streamlit application
├── requirements.txt    # Python dependencies
├── .env                # Local environment variables (API keys)
├── chroma_persist/     # Directory where ChromaDB stores embeddings
├── tmp_uploads/        # Temporary storage for uploaded PDFs
└── README.md           # You are here!
