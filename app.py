import os
import uuid
import time
import json
import logging
from pathlib import Path
from typing import Optional, List, Dict, Tuple
from dotenv import load_dotenv
import streamlit as st
from datetime import datetime

from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_chroma import Chroma
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from chromadb import PersistentClient



logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY", "")
os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN", "")



@st.cache_resource
def get_embeddings():
    """Initialize embeddings with caching for performance."""
    return HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={"trust_remote_code": True}
    )

@st.cache_resource
def get_llm():
    """Initialize LLM with optimized settings."""
    return ChatGroq(
        groq_api_key=os.environ["GROQ_API_KEY"],
        model_name="openai/gpt-oss-20b",
        temperature=0.3,
        max_tokens=1024
    )

embeddings = get_embeddings()
llm = get_llm()

PERSIST_DIR = "chroma_persist"
METADATA_FILE = os.path.join(PERSIST_DIR, "document_metadata.json")
client = PersistentClient(path=PERSIST_DIR)

# Persist directory
os.makedirs(PERSIST_DIR, exist_ok=True)




def load_document_metadata() -> Dict:
    """Load document metadata from file."""
    if os.path.exists(METADATA_FILE):
        try:
            with open(METADATA_FILE, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Could not load metadata: {e}")
            return {}
    return {}

def save_document_metadata(metadata: Dict):
    """Save document metadata to file."""
    try:
        with open(METADATA_FILE, 'w') as f:
            json.dump(metadata, f, indent=2)
    except Exception as e:
        logger.error(f"Could not save metadata: {e}")

def get_collection_stats() -> Dict:
    """Get statistics about the current collection."""
    try:
        collection = client.get_collection("rag_collection")
        count = collection.count()
        return {"document_count": count}
    except Exception:
        return {"document_count": 0}

def load_or_create_vectorstore():
    """Load or create the Chroma vectorstore."""
    vectorstore = Chroma(
        client=client,
        collection_name="rag_collection",
        embedding_function=embeddings
    )
    return vectorstore




def create_vector_embedding(file_path: str, session_id: Optional[str] = None) -> Tuple[bool, str]:
    """Create vector embeddings for a PDF file.
    Returns (success: bool, filename: str)
    """
    try:
        vectorstore = load_or_create_vectorstore()
        filename = os.path.basename(file_path)
        
        logger.info(f"Processing {filename}")
        loader = PyPDFLoader(file_path)
        documents = loader.load()
        
        if not documents:
            return False, f"No text extracted from {filename}"
        
        for doc in documents:
            doc.metadata["source"] = filename
            doc.metadata["uploaded_at"] = datetime.now().isoformat()
            if session_id is not None:
                doc.metadata["session_id"] = session_id
        
        # Chunk settings
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", " ", ""]
        )
        final_docs = splitter.split_documents(documents)
        
        vectorstore.add_documents(final_docs, batch_size=100)
        
        # Update metadata
        metadata = load_document_metadata()
        metadata[filename] = {
            "chunks": len(final_docs),
            "pages": len(documents),
            "uploaded_at": datetime.now().isoformat(),
            "status": "indexed",
            "session_id": session_id
        }
        save_document_metadata(metadata)
        
        try:
            os.remove(file_path)
            logger.info(f"Deleted source file: {file_path}")
        except OSError as e:
            logger.warning(f"Could not delete {file_path}: {e}")
        
        return True, filename
    except Exception as e:
        logger.error(f"Error processing {file_path}: {e}")
        return False, str(e)




def get_rag_chain(vectorstore: Chroma):
    """Create an optimized RAG chain with conversational history."""
    retriever = vectorstore.as_retriever(
        search_kwargs={"k": 5}
    )
    
    system_prompt = (
        "You are an intelligent document assistant providing accurate, helpful answers. "
        "Use the provided context to answer questions thoroughly and clearly. "
        "If the information is not in the documents, say so explicitly. "
        "Cite specific sections when relevant.\n\n"
        "Context:\n{context}"
    )
    
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ])
    
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)
    
    store = {}

    def get_session_history(session_id: str) -> BaseChatMessageHistory:
        if session_id not in store:
            store[session_id] = ChatMessageHistory()
        return store[session_id]

    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
    )

    return conversational_rag_chain


def call_conversational_rag(conversational_rag_chain, question: str, session_id: str):
    """Invoke the conversational RAG chain and return the result dictionary."""
    try:
        input_payload = {"input": question}
        config_payload = {"configurable": {"session_id": session_id}}
        return conversational_rag_chain.invoke(input_payload, config=config_payload)
    except Exception as e:
        logger.error(f"RAG invocation failed: {e}")
        raise


def format_sources(context_documents, max_snippets: int = 5) -> str:
    """
    Format source documents with page references.
    Returns markdown-formatted string with sources.
    """
    # Build a mapping of filename -> set(page_numbers)
    sources_dict: Dict[str, set] = {}

    for doc in context_documents[:max_snippets]:
        md = getattr(doc, "metadata", {}) or {}

        # Determine the source: prefer explicit filename, otherwise try common keys
        source = md.get("source") or md.get("filename") or md.get("file") or md.get("source_name")

        # Normalize source to just a filename when possible
        if source:
            try:
                src_name = Path(str(source)).name
            except Exception:
                src_name = str(source)
        else:
            src_name = "Unknown"

        # Try multiple possible page metadata keys and normalize to 1-based page numbers
        page_num = None
        for key in ("page", "page_number", "page_no", "pageno"):
            if key in md:
                try:
                    page_num = int(md[key])
                    # In some loaders 'page' is zero-indexed
                    if key == "page":
                        page_num = page_num + 1
                except Exception:
                    page_num = None
                break

        # If no explicit page number is available, skip adding (avoid noisy references)
        if page_num is None:
            continue

        if src_name not in sources_dict:
            sources_dict[src_name] = set()
        sources_dict[src_name].add(page_num)

    if not sources_dict:
        return ""

    # Format as simple file -> pages list, no collection names
    sources_text = "\n\n**📄 Sources:**\n"
    for source, pages in sorted(sources_dict.items()):
        pages_str = ", ".join(f"Page {p}" for p in sorted(pages))
        sources_text += f"• **{source}** - {pages_str}\n"

    return sources_text


def delete_embeddings_for_session(session_id: str) -> Tuple[bool, str]:
    """
    Delete embeddings and metadata entries associated with a given session_id.
    Returns (success, message).
    """
    try:
        collection = client.get_collection("rag_collection")
        # Delete vectors whose metadata.session_id matches
        collection.delete(where={"session_id": session_id})

        # Remove metadata entries for files associated with this session
        metadata = load_document_metadata()
        to_delete = [name for name, info in metadata.items() if info.get("session_id") == session_id]
        for name in to_delete:
            metadata.pop(name, None)
        save_document_metadata(metadata)

        return True, f"Deleted embeddings for session {session_id} (files: {to_delete})"
    except Exception as e:
        logger.error(f"Failed to delete embeddings for session {session_id}: {e}")
        return False, str(e)




st.set_page_config(
    page_title="Document Assistant",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header { font-size: 2.5rem; color: #1f77e6; margin-bottom: 1rem; }
    .doc-badge { display: inline-block; background: #e0e7ff; color: #3730a3; 
                 padding: 0.5rem 1rem; border-radius: 0.5rem; margin: 0.25rem; font-size: 0.9rem; }
    .source-box { background: #f0f9ff; border-left: 4px solid #3b82f6; padding: 1rem; margin: 0.5rem 0; }
    </style>
""", unsafe_allow_html=True)

st.title("📄 Document Assistant — Ask your PDFs")

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "vectorstore" not in st.session_state:
    with st.spinner("Loading your saved documents..."):
        try:
            st.session_state.vectorstore = load_or_create_vectorstore()
        except Exception as e:
            st.error(f"Error loading saved documents: {e}")
            st.session_state.vectorstore = None

if "conv_chain" not in st.session_state:
    st.session_state.conv_chain = None
    if st.session_state.vectorstore is not None:
        try:
            st.session_state.conv_chain = get_rag_chain(st.session_state.vectorstore)
        except Exception as e:
            st.warning(f"Failed to prepare assistant: {e}")

if "messages" not in st.session_state:
    st.session_state.messages = []



with st.sidebar:
    st.header("📚 Your Documents")
    
    # Show document statistics
    stats = get_collection_stats()
    col1 = st.columns(1)[0]
    with col1:
        st.metric("📊 Sections Indexed", stats["document_count"])
    
    # Document metadata display
    metadata = load_document_metadata()
    if metadata:
        st.subheader("Uploaded Documents")
        for doc_name, info in metadata.items():
            with st.expander(f"📄 {doc_name}", expanded=False):
                st.write(f"**Chunks:** {info.get('chunks', 'N/A')}")
                st.write(f"**Pages:** {info.get('pages', 'N/A')}")
                st.write(f"**Status:** {info.get('status', 'unknown')}")
    
    st.divider()
    
    # File upload section
    st.subheader("📥 Add PDF files")
    uploaded_files = st.file_uploader(
        "Select PDF files",
        type=["pdf"],
        accept_multiple_files=True,
        help="Upload one or multiple PDF documents"
    )
    
    if st.button("📥 Upload", use_container_width=True):
        if not uploaded_files:
            st.warning("Please select files to index.")
        else:
            tmp_dir = "tmp_uploads"
            os.makedirs(tmp_dir, exist_ok=True)
            
            success_count = 0
            error_count = 0
            
            progress_container = st.container()
            with progress_container:
                st.info(f"Preparing {len(uploaded_files)} file(s) for search...")
                progress_bar = st.progress(0)
                
                for idx, uploaded_file in enumerate(uploaded_files):
                    tmp_path = os.path.join(tmp_dir, f"{uuid.uuid4()}_{uploaded_file.name}")
                    
                    with open(tmp_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    
                    success, filename = create_vector_embedding(tmp_path, session_id=st.session_state.session_id)
                    
                    if success:
                        success_count += 1
                        st.success(f"Indexed: {filename}")
                    else:
                        error_count += 1
                        st.error(f"Failed to index: {filename}")
                    
                    progress_bar.progress((idx + 1) / len(uploaded_files))
            
            # Update document index
            with st.spinner("Finalizing upload..."):
                try:
                    st.session_state.vectorstore = load_or_create_vectorstore()
                    st.session_state.conv_chain = None
                    st.success(f"Completed: {success_count} succeeded, {error_count} failed")
                except Exception as e:
                    st.error(f"Failed to update document index: {e}")
    
    st.divider()
    
    # Settings
    st.subheader("⚙️ Settings")

    if st.button("🔄 Reset Chat History", use_container_width=True):
        st.session_state.session_id = str(uuid.uuid4())
        st.session_state.messages = []
        st.session_state.conv_chain = None
        st.success("Chat history reset!")
        try:
            st.session_state.vectorstore = load_or_create_vectorstore()
        except Exception:
            st.session_state.vectorstore = None
        st.rerun()
    
    if st.button("🧹 Clear Vectorstore", use_container_width=True):
        try:
            client.delete_collection("rag_collection")
            st.session_state.vectorstore = load_or_create_vectorstore()
            st.session_state.conv_chain = None
            if os.path.exists(METADATA_FILE):
                os.remove(METADATA_FILE)
            st.success("Vectorstore cleared!")
            st.rerun()
        except Exception as e:
            st.error(f"Error clearing vectorstore: {e}")




# Only enable chat when at least one document is indexed
doc_count = get_collection_stats().get("document_count", 0)
if doc_count == 0:
    st.info("📤 Upload PDF files first to start asking questions.")
else:
    # Rebuild chain if needed
    if st.session_state.conv_chain is None:
        with st.spinner("Preparing assistant..."):
            try:
                st.session_state.conv_chain = get_rag_chain(st.session_state.vectorstore)
                st.info("Assistant ready!")
            except Exception as e:
                st.error(f"Failed to prepare assistant: {e}")

    # Display chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Chat input
    if user_input := st.chat_input("Ask a question about your files..."):
        if st.session_state.conv_chain is None:
            st.error("Assistant not ready. Please wait a moment and try again.")
        else:
            # Add user message
            st.session_state.messages.append({"role": "user", "content": user_input})
            with st.chat_message("user",avatar="🧑‍💻"):
                st.markdown(user_input)
            
            # Get response
            with st.chat_message("assistant",avatar="🤖"):
                with st.spinner("Searching your documents for an answer..."):
                    try:
                        start_time = time.time()
                        
                        result_dict = call_conversational_rag(
                            st.session_state.conv_chain,
                            user_input,
                            st.session_state.session_id
                        )
                        
                        answer = result_dict.get('answer', 'No answer generated')
                        context_docs = result_dict.get('context', [])
                        
                        elapsed = time.time() - start_time
                        
                        # Format final response
                        sources_text = format_sources(context_docs)
                        final_answer = f"{answer}{sources_text}\n\n⏱️ _Retrieved in {elapsed:.2f}s_"
                        
                        st.markdown(final_answer)
                        
                    except Exception as e:
                        logger.error(f"Error: {e}")
                        st.error(f"Error generating answer: {e}")
                        final_answer = f"❌ Error: {str(e)}"
                
                # Add assistant message to history
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": final_answer
                })
