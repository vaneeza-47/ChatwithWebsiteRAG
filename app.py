import os
import requests
from dotenv import load_dotenv
from bs4 import BeautifulSoup

import streamlit as st
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai

# --------------- SETUP ------------------

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    raise ValueError('Please add "GOOGLE_API_KEY" to .env')

genai.configure(api_key=GOOGLE_API_KEY)
DB_DIR = "chroma_db"


# ----------------- WEBSITE LOADER -----------------

def load_website_content(url):
    headers = {
        "User-Agent": 
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/123.0.0.0 Safari/537.36"
    }

    response = requests.get(url, headers=headers, timeout=10)
    response.raise_for_status() # this will raise 404/403 page errors

    soup = BeautifulSoup(response.text, "html.parser") # convert raw html into navigatable structure

    # Remove the junk
    for tag in soup(["script", "style", "nav", "footer", "header"]):
        tag.extract()

    # Clean text
    text = soup.get_text(separator="\n")
    cleaned_text = "\n".join(
        line.strip() for line in text.splitlines() if line.strip()
    )

    return [Document(page_content=cleaned_text, metadata={"source": url})]


# ---------------- CHUNKING ----------------------
# split the cleaned text into chunks

def split_documents(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
        length_function=len
    )
    return splitter.split_documents(docs)


# ----------------- EMBEDDINGS  -----------------
 
def get_gemini_embeddings():
    return GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=GOOGLE_API_KEY
    )

# ---------------- VECTOR STORE ----------------------

def get_vectorstore(docs, url):
    for d in docs:
        d.metadata["source"] = url

    vectordb = Chroma.from_documents(
        documents=docs,
        embedding=get_gemini_embeddings(),
        persist_directory=DB_DIR
    )
    vectordb.persist() # saves db to disk
    return vectordb


# load existing vector stores
def load_existing_vectorstores():
    # if no db exists then do not crash
    if not os.path.exists(DB_DIR):
        return None
    
    return Chroma(
        embedding_function=get_gemini_embeddings(),
        persist_directory=DB_DIR
    )

# ------------------ ANSWER GENERATION ------------------

def chat_with_context(question, context):
    model = genai.GenerativeModel("gemini-2.5-flash")

    prompt = f"""
        You are a helpful assistant for answering questions using the provided website content.
        Use only the retrieved context to answer.
        If unsure, say "I don't know" — do not make up an answer.
        \n\n

        CONTEXT:
        {context}

        QUESTION:
        {question}    
    """
    response = model.generate_content(prompt)
    return response.text


# -----------------STREAMLIT UI ---------------------

st.set_page_config(page_title="Chat with Websites", layout="wide")
st.markdown(
    """
        <div 
            style="
                background:#27F5D3;
                padding:20px;
                border-radius:12px;
                margin-bottom:20px;
                border:3px solid #000000;
            ">
            <h2 style="color:black; margin:0;">
                Chat With Website Using Gemini
            </h2>
        </div>
    """, unsafe_allow_html=True
)

left, right = st.columns([1, 2])


# ----------------------- LEFT COLUMN (LOAD WEBSITE) -----------------------------

with left:
    st.subheader("🌐 1. Load Website URL")

    url = st.text_input("Enter a website URL")
    if "vectordb" not in st.session_state:
        st.session_state.vectordb = load_existing_vectorstores()

    if st.button("Fetch and Index URL"):
        if not url:
            st.warning("Please enter a URL first.")
        else:
            try:
                with st.spinner("Fetching and Indexing Website..."):
                    docs = load_website_content(url)
                    chunks = split_documents(docs)
                    vectordb = get_vectorstore(chunks, url)
                    st.session_state.vectordb = vectordb

                st.success("Website indexed succesfully!")

            except Exception as e:
                st.error(f"Error while loading: {e}")

            
# ------------------- RIGHT COLUMN (CHAT) ------------------
with right:
    st.subheader("🤖 2. Chat with Website Content")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # show chat history
    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).markdown(msg["content"])

    question = st.chat_input("Ask Something")

    if question:
        if not st.session_state.vectordb:
            st.warning("Please enter a URL first.")
        else:
            # user message
            st.session_state.messages.append(
                {"role": "user", 
                 "content": question}
                 )
            
            st.chat_message("user").markdown(question)

            with st.chat_message("🤖"):
                with st.spinner("Thinking..."):
                    vectordb = st.session_state.vectordb

                    # rerive top 4 chunks
                    # convert question to embeddings 
                    retriever = vectordb.as_retriever(search_kwargs={"k":4})
                    docs = retriever.invoke(question)

                    # extract text from documents
                    context = "\n\n".join([d.page_content for d in docs])
                    answer = chat_with_context(question, context)

                    st.markdown(answer)

                    # chat answer
                    st.session_state.messages.append(
                        {"role": "🤖", 
                         "content": answer}
                        )




