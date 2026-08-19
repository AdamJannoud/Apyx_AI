import streamlit as st
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os

os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]

st.set_page_config(page_title="Apyx AI Expert", page_icon="🤖", layout="centered")
st.title("🔴 Apyx Digital Credit AI Assistant")
st.markdown("Ask me anything about Digital Credit, $STRC, $SATA, or our dual-token model on Solana!")

@st.cache_resource 
def process_pdf_and_create_vectorstore():
    pdf_path = "Digital Credit Engineering and the Apyx Financial Protocol.pdf"
    text = ""
    
    pdf_reader = PdfReader(pdf_path)
    for page in pdf_reader.pages:
        text += page.extract_text()
        
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_text(text)
    
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    vector_store = FAISS.from_texts(chunks, embedding=embeddings)
    return vector_store

vector_store = process_pdf_and_create_vectorstore()

prompt_template = """
You are the official Apyx Protocol AI Assistant. 
CRITICAL RULE: You MUST answer the question based ONLY on the provided context below. Do not invent information.
If the answer is not in the context, say: "I can only answer questions related to the Apyx Protocol and Digital Credit based on official documents."

Context:
{context}

Question: {question}

Answer (Keep it professional, concise, and Web3-native):
"""
prompt = PromptTemplate.from_template(prompt_template)
model = ChatGoogleGenerativeAI(model="gemini-3.7-flash", temperature=0.3)
output_parser = StrOutputParser()

chain = prompt | model | output_parser

user_question = st.text_input("Enter your question here:")

if user_question:
    docs = vector_store.similarity_search(user_question, k=4)
    context_text = "\n\n".join([doc.page_content for doc in docs])
    
    with st.spinner("Analyzing Apyx Documents..."):
        response = chain.invoke({"context": context_text, "question": user_question})
        
    st.success("Response:")
    st.write(response)