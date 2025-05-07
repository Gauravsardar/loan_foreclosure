import streamlit as st
import os
from dotenv import load_dotenv
from phi.agent import Agent
from phi.model.google import Gemini
from phi.knowledge.pdf import PDFKnowledgeBase
from phi.embedder.openai import OpenAIEmbedder
from phi.vectordb.lancedb import LanceDb, SearchType

load_dotenv()  

PDF_PATH = os.path.join('public', 'Employee Handbook.pdf')

# Create a knowledge base from a PDF
knowledge_base = PDFKnowledgeBase(
    paths=[PDF_PATH],
    vector_db=LanceDb(
        table_name="employee_handbook",
        uri="tmp/lancedb",
        search_type=SearchType.vector,
        embedder=OpenAIEmbedder(model="text-embedding-3-small"),
    ),
)
# Load the knowledge base (do this once)
knowledge_base.load()

agent = Agent(
    model=Gemini(id = "gemini-2.0-flash"),
    knowledge=knowledge_base,
    show_tool_calls=True,
    markdown=True,
)

st.title('Policy Retrieval Agent (RAG Demo, phidata, Gemini)')
st.write('Ask a question about the Employee Handbook:')

with st.spinner('Loading and indexing policy document...'):
    pass

query = st.text_input('Enter your query:')

if query:
    with st.spinner('Retrieving answer...'):
        results = agent.retrieve(query, top_k=3)
        st.subheader('Top Relevant Policy Excerpts:')
        for doc in results:
            st.write(doc.page_content)

st.caption('Powered by Streamlit, phidata, LanceDB, and Gemini.') 