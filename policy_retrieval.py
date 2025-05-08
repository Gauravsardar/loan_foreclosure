import streamlit as st
import os
import shutil
import logging
import tempfile
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, END
from typing import TypedDict, List, Optional
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
import chromadb
import uuid

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    st.error("GOOGLE_API_KEY not found in .env file. Please set it to continue.")
    st.stop()

# Paths and configurations
PDF_PATH = os.path.join('public', 'Employee Handbook.pdf')

# Create a unique session directory for Chroma database
if "persist_dir" not in st.session_state:
    # Generate a unique directory name to avoid conflicts between sessions
    unique_id = str(uuid.uuid4())
    st.session_state.persist_dir = os.path.join(tempfile.gettempdir(), f"chroma_db_{unique_id}")
    logger.info(f"Created unique Chroma DB path: {st.session_state.persist_dir}")

PERSIST_DIR = st.session_state.persist_dir

# Initialize Gemini model (use gemini-pro-1.5 or gemini-1.5-pro for best reasoning capabilities)
# The model parameter may need to be updated depending on the latest available Gemini models
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-preview-04-17",  # Change to the latest reasoning model
    google_api_key=GOOGLE_API_KEY,
    temperature=0.1,     # Lower temperature for more focused responses
)

# Function to reset Chroma database
def reset_chroma_db():
    try:
        if os.path.exists(PERSIST_DIR):
            shutil.rmtree(PERSIST_DIR)
            logger.info(f"Chroma database at {PERSIST_DIR} reset successfully.")
            return True
    except PermissionError as e:
        logger.warning(f"PermissionError during reset: {str(e)}")
        return False
    except Exception as e:
        logger.error(f"Error during reset: {str(e)}")
        return False
    return True

# Load and split PDF
@st.cache_data
def load_pdf_and_split():
    logger.info("Loading and splitting PDF...")
    if not os.path.exists(PDF_PATH):
        st.error(f"PDF file not found at {PDF_PATH}. Please ensure it exists.")
        logger.error(f"PDF not found at {PDF_PATH}")
        st.stop()
    try:
        loader = PyPDFLoader(PDF_PATH)
        docs = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(docs)
        logger.info(f"PDF loaded and split into {len(splits)} chunks.")
        return splits
    except Exception as e:
        st.error(f"Error loading PDF: {str(e)}")
        logger.error(f"PDF loading error: {str(e)}")
        st.stop()

# Load or create Chroma vector store
@st.cache_resource
def load_and_index_pdf(_splits=None, persist_dir=None):
    if persist_dir is None:
        persist_dir = PERSIST_DIR
        
    logger.info(f"Using Chroma database at: {persist_dir}")
    embedding_function = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # Configure Chroma client
    client_settings = chromadb.config.Settings(
        persist_directory=persist_dir, 
        is_persistent=True,
        anonymized_telemetry=False
    )
    
    try:
        logger.info("Creating new Chroma database...")
        if _splits is None:
            _splits = load_pdf_and_split()
            
        vectorstore = Chroma.from_documents(
            _splits,
            embedding_function,
            persist_directory=persist_dir,
            client_settings=client_settings
        )
        logger.info("Chroma database created successfully.")
        return vectorstore
    except Exception as e:
        logger.error(f"Error creating Chroma database: {str(e)}")
        raise e

# Define expanded LangGraph state with reasoning
class GraphState(TypedDict):
    query: str
    documents: List[Document]
    reasoning: Optional[str]  # Added reasoning field
    response: str

# LangGraph nodes
def retrieve_documents(state: GraphState) -> GraphState:
    logger.info(f"Retrieving documents for query: {state['query']}")
    vectorstore = st.session_state.vectorstore
    state["documents"] = vectorstore.similarity_search(state["query"], k=4)  # Increased to 4 documents
    logger.info(f"Retrieved {len(state['documents'])} documents.")
    return state

def analyze_and_reason(state: GraphState) -> GraphState:
    """New node that analyzes documents and performs reasoning."""
    logger.info("Analyzing documents and reasoning...")
    
    # Prepare context from retrieved documents
    context = "\n\n".join([f"Document {i+1}:\n{doc.page_content}" 
                           for i, doc in enumerate(state["documents"])])
    
    # Define reasoning prompt
    reasoning_prompt = ChatPromptTemplate.from_template(
        """You are a policy analysis system working with an Employee Handbook. 
        First, analyze these documents retrieved from the handbook. Then, reason step-by-step about how they relate to the user's query. 
        Focus on identifying the most relevant information, any policy constraints, and how to answer the query accurately.
        
        **Retrieved Documents**:
        {context}
        
        **User Query**:
        {query}
        
        **Step-by-Step Reasoning**:
        Let me think through this carefully:
        1. First, I need to understand what policy information the user is asking for.
        2. I should identify which of the retrieved documents contain relevant information.
        3. I need to analyze if there are any specific policy rules, exceptions, or processes mentioned.
        4. I should determine if there are any conditions or requirements related to the query.
        
        Begin your reasoning now:
        """
    )
    
    # Generate reasoning
    reasoning_chain = reasoning_prompt | llm
    reasoning_result = reasoning_chain.invoke({"context": context, "query": state["query"]})
    state["reasoning"] = reasoning_result.content
    logger.info("Reasoning completed.")
    return state

def generate_response(state: GraphState) -> GraphState:
    logger.info("Generating final response...")
    
    # Prepare context from retrieved documents
    context = "\n\n".join([f"Document {i+1}:\n{doc.page_content}" 
                           for i, doc in enumerate(state["documents"])])
    
    # Define response prompt that incorporates the reasoning
    response_prompt = ChatPromptTemplate.from_template(
        """You are a policy retrieval assistant. Based on the following excerpts from the Employee Handbook 
        and the reasoning analysis, answer the user's query. Provide a concise and accurate response, 
        referencing the handbook where relevant.

        **Handbook Excerpts**:
        {context}
        
        **User Query**:
        {query}
        
        **Your Reasoning Analysis**:
        {reasoning}
        
        **Final Answer**:
        Given the above analysis, here's my definitive answer to the user's query:
        """
    )
    
    # Generate response using reasoning
    response_chain = response_prompt | llm
    response_result = response_chain.invoke({
        "context": context, 
        "query": state["query"],
        "reasoning": state["reasoning"]
    })
    state["response"] = response_result.content
    logger.info("Final response generated.")
    return state

# Define LangGraph workflow with reasoning step
def create_graph():
    workflow = StateGraph(GraphState)
    
    # Add nodes including the new reasoning node
    workflow.add_node("retrieve", retrieve_documents)
    workflow.add_node("reason", analyze_and_reason)
    workflow.add_node("generate", generate_response)
    
    # Define edges
    workflow.set_entry_point("retrieve")
    workflow.add_edge("retrieve", "reason")
    workflow.add_edge("reason", "generate")
    workflow.add_edge("generate", END)
    
    # Compile graph
    return workflow.compile()

# Streamlit UI
st.title('Policy Retrieval Agent with Reasoning')
st.write('Ask a question about the Employee Handbook:')

# UI Controls in sidebar
with st.sidebar:
    st.header("Controls")
    
    # Database reset button
    if st.button("Reset Database", key="reset_db_button"):
        if reset_chroma_db():
            st.success("Database reset successfully.")
            st.cache_data.clear()
            st.cache_resource.clear()
            st.rerun()
        else:
            st.error("Failed to reset database. Try restarting the application.")
    
    # Show the current database path
    st.info(f"Using database at: {PERSIST_DIR}")
    
    # Show reasoning toggle
    show_reasoning = st.toggle("Show reasoning process", value=True, key="show_reasoning")
    
    st.markdown("---")
    st.caption("Powered by Streamlit, LangGraph, Chroma, and Google Gemini")

# Load vector store at startup
with st.spinner('Loading and indexing policy document...'):
    try:
        splits = load_pdf_and_split()
        st.session_state.vectorstore = load_and_index_pdf(_splits=splits, persist_dir=PERSIST_DIR)
    except Exception as e:
        st.error(f"Error loading index: {str(e)}")
        if st.button("Retry Loading", key="retry_loading"):
            st.rerun()
        st.stop()

# User query input
query = st.text_input('Enter your query:', key="query_input")

if query:
    with st.spinner('Retrieving, reasoning, and generating answer...'):
        try:
            # Initialize and run LangGraph
            graph = create_graph()
            initial_state = GraphState(query=query, documents=[], reasoning=None, response="")
            result = graph.invoke(initial_state)
            
            # Display results in expandable sections
            with st.expander("📚 Retrieved Policy Excerpts", expanded=False):
                for i, doc in enumerate(result["documents"]):
                    st.markdown(f"**Document {i+1}:**")
                    st.markdown(doc.page_content)
                    st.markdown("---")
            
            # Display reasoning process if enabled
            if show_reasoning and result["reasoning"]:
                with st.expander("🧠 Reasoning Process", expanded=True):
                    st.markdown(result["reasoning"])
            
            # Display final answer
            st.subheader('💡 Answer:')
            st.markdown(result["response"])
            
            # Add feedback buttons
            col1, col2 = st.columns(2)
            with col1:
                st.button("👍 Helpful", key="feedback_helpful")
            with col2:
                st.button("👎 Not Helpful", key="feedback_not_helpful")
            
        except Exception as e:
            st.error(f"Error processing query: {str(e)}")
            logger.error(f"Query processing error: {str(e)}")

# Add a download sample queries feature
with st.sidebar:
    st.markdown("---")
    st.subheader("Sample Queries")
    sample_queries = [
        "What are the documentations required for loan foreclosure?"
    ]
    
    for query in sample_queries:
        if st.button(f"Try: {query}", key=f"sample_{hash(query)}"):
            # This uses the JavaScript hack to set the input value
            st.session_state.query_input = query
            st.rerun()