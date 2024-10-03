from langchain_groq import ChatGroq
from langchain_openai import OpenAIEmbeddings 
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings

import streamlit as st
import os
import time
from dotenv import load_dotenv
import tempfile

load_dotenv()

# os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')
groq_api_key = os.getenv('GROQ_API_KEY')
os.environ['GOOGLE_API_KEY'] = os.getenv('GOOGLE_API_KEY')

# Set the page layout to wide
st.set_page_config(page_title='PDF Q&A', layout='wide')

# Display title in the sidebar
st.sidebar.title('Gen AI RAG System')
st.sidebar.write("A PDF Q&A RAG system designed to comprehend your PDF's context and provide accurate answers to your questions quickly—give it a try!")

st.title('Ask my PDF')

llm = ChatGroq(groq_api_key = groq_api_key,
               model_name = 'llama3-8b-8192')

prompt = ChatPromptTemplate.from_template(
"""
Answer the questions based on the provided context only.
Please provide the most accurate response based on the question
<context>
{context}
<context>
Questions: {input}
"""    
)

def vector_embedding(uploaded_pdf_file):
    
    if uploaded_pdf_file is not None:

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file.write(uploaded_pdf_file.read())  # Save uploaded PDF to temp file
            temp_file_path = temp_file.name  # Get the file path
        
        with st.spinner("Loading PDF..."):

            st.session_state.loader = PyPDFLoader(temp_file_path) #data ingestion
            st.session_state.embeddings = GoogleGenerativeAIEmbeddings(model = 'models/embedding-001')
            st.session_state.docs = st.session_state.loader.load() #document loading
            st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap = 200) #chunk creation
            st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs[:20]) #splitting
            st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings) #vector embeddings


# Function to render chat bubble
def chat_bubble(message, is_user=False):
    alignment = "right" if is_user else "left"
    color = "#e0f7fa" if not is_user else "#c8e6c9"  # Different colors for user and assistant
    return f"""
    <div style="display: flex; justify-content: {alignment};">
        <div style="max-width: 60%; margin: 5px; padding: 10px; border-radius: 15px; background-color: {color};">
            <p style="margin: 0;">{message}</p>
        </div>
    </div>
    """


uploaded_pdf_file = st.file_uploader('Upload PDF', type=['pdf'])

# Add custom CSS to change button color
st.markdown(
    """
    <style>
        .stButton > button {
            background-color: #007BFF; /* Change to primary blue */
            color: white; /* Text color */
            border: none; /* Remove border */
        }
        .stButton > button:hover {
            background-color: #0056b3; /* Darker shade for hover effect */
        }
    </style>
    """,
    unsafe_allow_html=True
)

if st.button('Learn PDF (Creates a Knowledge Base)') and uploaded_pdf_file:
    vector_embedding(uploaded_pdf_file) 
    st.write('Knowledge Based Created!') #vector db is available in st.session_state

prompt1 = st.text_input('Ask a question')


if prompt1 and 'vectors' in st.session_state:
    document_chain = create_stuff_documents_chain(llm, prompt)
    retriever = st.session_state.vectors.as_retriever()
    retriever_chain = create_retrieval_chain(retriever, document_chain)
    start = time.process_time()
    response = retriever_chain.invoke({'input': prompt1})
    # print('Response time : ', time.process_time()-start)
    # st.write(response['answer'])

    response_time = time.process_time() - start

    # Display the user question in a chat bubble
    st.markdown(chat_bubble(prompt1, is_user=True), unsafe_allow_html=True)

    # Display the answer in a chat bubble
    st.markdown(chat_bubble(response['answer'], is_user=False), unsafe_allow_html=True)

    # Display the response time
    st.write(f"Response time: {response_time:.2f} seconds")

    with st.expander('Document Similarity Search'):
        for i, doc in enumerate(response['context']):
            st.write(doc.page_content)
            st.write("-------------------------------")


st.markdown(
    """
    <style>
        footer {
            position: fixed;
            left: 0;
            bottom: 0;
            width: 100%;
            text-align: center;
            padding: 10px;
            background-color: #f1f1f1; /* Change this color as needed */
        }
    </style>
    <footer>
        <p>Powered by LangChain (Llama3 model & Google GenAI) & Groq API</p>
    </footer>
    """,
    unsafe_allow_html=True
)

