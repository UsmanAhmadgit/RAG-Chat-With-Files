import streamlit as st
import os
from PyPDF2 import PdfReader
import docx
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from streamlit_chat import message
import google.generativeai as genai
from langchain.llms.base import LLM
from typing import Optional
from pydantic import Field

# Configure Google Gemini API
key = st.secrets["gemini_key"]
genai.configure(api_key = key)
model = genai.GenerativeModel("gemini-1.5-flash")

# Custom LangChain Wrapper for Google Gemini
class GoogleGeminiLLM(LLM):
    model_name: str = Field(default="gemini-1.5-flash")  # Declare fields
    temperature: float = Field(default=0.7)

    @property
    def _llm_type(self) -> str:
        return "google_gemini"

    def _call(self, prompt: str, stop: Optional[list] = None) -> str: 
        response = model.generate_content(prompt)
        return response.text


# Main Streamlit App
def main():
    st.set_page_config(page_title="Chat with your file")
    st.header("RAG - Chat With Your Files 🧠")

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None
    if "processComplete" not in st.session_state:
        st.session_state.processComplete = None

    with st.sidebar:
        uploaded_files = st.file_uploader("Upload your file", type=['pdf', 'docx'], accept_multiple_files=True)
        process = st.button("Process")

    if process:
        if not uploaded_files:
            st.info("Please upload a file to continue.")
            st.stop()

        files_text = get_files_text(uploaded_files)
        st.write("File loaded...")
        # Get text chunks
        text_chunks = get_text_chunks(files_text)
        st.write("File chunks created...")
        # Create vector store
        vectorstore = get_vectorstore(text_chunks)
        st.write("Vector Store Created...")
        # Create conversation chain
        st.session_state.conversation = get_conversation_chain(vectorstore)
        st.session_state.processComplete = True

    if st.session_state.processComplete:
        user_question = st.chat_input("Ask a question about your files.")
        if user_question:
            handle_user_input(user_question)


# Function to get the input file and read the text from it.
def get_files_text(uploaded_files):
    text = ""
    for uploaded_file in uploaded_files:
        file_extension = os.path.splitext(uploaded_file.name)[1]
        if file_extension == ".pdf":
            text += get_pdf_text(uploaded_file)
        elif file_extension == ".docx":
            text += get_docx_text(uploaded_file)
    return text


# Function to read PDF Files
def get_pdf_text(pdf):
    pdf_reader = PdfReader(pdf)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text


# Function to read DOCX files
def get_docx_text(file):
    doc = docx.Document(file)
    all_text = [para.text for para in doc.paragraphs]
    return " ".join(all_text)


# Function to split text into chunks
def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=900,
        chunk_overlap=100,
        length_function=len
    )
    return text_splitter.split_text(text)


# Function to create a vector store
def get_vectorstore(text_chunks):
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return FAISS.from_texts(text_chunks, embeddings)


# Function to create a conversation chain
def get_conversation_chain(vectorstore):
    llm = GoogleGeminiLLM(model_name="gemini-1.5-flash", temperature=0.5)
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain
 

# Function to handle user input
def handle_user_input(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']

    # Layout of input/response containers
    response_container = st.container()

    with response_container:
        for i, messages in enumerate(st.session_state.chat_history):
            if i % 2 == 0:
                message(messages.content, is_user=True, key=str(i))
            else:
                message(messages.content, key=str(i))


if __name__ == '__main__':
    main()
