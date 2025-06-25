import os
import chardet
import streamlit as st
import pdfplumber
from docx import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set Streamlit page config
st.set_page_config(page_title="Custom ChatBot", page_icon="🤖", layout="wide")
st.title("🤖 Your ChatBot, Your Rules — Create It Your Way")

# Upload section
st.sidebar.header("📄 Upload Files (.txt, .pdf, .docx)")
upload_folder = "uploaded_files"
os.makedirs(upload_folder, exist_ok=True)

uploaded_files = st.sidebar.file_uploader(
    "Upload .txt, .pdf, .docx files", type=["txt", "pdf", "docx"], accept_multiple_files=True
)

# Initialize session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "docs" not in st.session_state:
    st.session_state.docs = []
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "processed_files" not in st.session_state:
    st.session_state.processed_files = set()
if "embedding_ready" not in st.session_state:
    st.session_state.embedding_ready = False

# Gemini API client
client = OpenAI(
    api_key=os.getenv("API_KEY"),
    base_url=os.getenv("BASE_URL")
)

# 📅 Extract text from various formats
def extract_text(file):
    ext = file.name.split('.')[-1].lower()
    try:
        if ext == "txt":
            raw = file.getvalue()
            detected = chardet.detect(raw)
            encoding = detected["encoding"] or "utf-8"
            return raw.decode(encoding)
        elif ext == "pdf":
            with pdfplumber.open(file) as pdf:
                return "\n".join([page.extract_text() for page in pdf.pages if page.extract_text()])
        elif ext == "docx":
            doc = Document(file)
            return "\n".join([para.text for para in doc.paragraphs])
        else:
            return ""
    except Exception as e:
        st.sidebar.error(f"❌ Error reading {file.name}: {e}")
        return ""

# 🧠 Process all uploaded files if new
if uploaded_files:
    file_contents = []
    new_files_uploaded = False

    for uploaded_file in uploaded_files:
        if uploaded_file.name in st.session_state.processed_files:
            st.sidebar.info(f"🔁 Skipped (already processed): {uploaded_file.name}")
            continue

        file_path = os.path.join(upload_folder, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        content = extract_text(uploaded_file)
        if content:
            file_contents.append(content)
            st.session_state.processed_files.add(uploaded_file.name)
            st.sidebar.success(f"✅ {uploaded_file.name} processed.")
            new_files_uploaded = True
        else:
            st.sidebar.warning(f"⚠️ Skipped: {uploaded_file.name}")

    if new_files_uploaded and file_contents:
        full_text = "\n".join(file_contents)
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
        new_docs = splitter.create_documents([full_text])
        st.session_state.docs.extend(new_docs)

        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        st.session_state.vectorstore = FAISS.from_documents(st.session_state.docs, embeddings)

        st.session_state.embedding_ready = True
        st.sidebar.info(f"📚 Total Chunks: {len(st.session_state.docs)}")
        st.sidebar.success("🧠 Datastore updated with new content.")
    elif not new_files_uploaded and st.session_state.vectorstore:
        st.session_state.embedding_ready = True

# 💬 Chat interface
if st.session_state.embedding_ready and st.session_state.vectorstore:
    st.markdown("### 💬 Start chatting below")
    user_input = st.chat_input("Type your question here...")

    if user_input:
        relevant_docs = st.session_state.vectorstore.similarity_search(user_input, k=3)
        context = "\n\n".join([doc.page_content for doc in relevant_docs])

        messages = [{"role": "system", "content": "You are a helpful assistant. Only answer using the provided context. If the answer is not in the context, say 'I don't know.'. Also your name is Itee's Chatbot incase someone asks. If somebody asks questions out of context simply write either ' I don't know!' or ask them to ask questions as per the provided information by them/"}]
        for user, bot in st.session_state.chat_history:
            messages.append({"role": "user", "content": user})
            messages.append({"role": "assistant", "content": bot})
        messages.append({"role": "user", "content": f"Context:\n{context}\n\nQuestion: {user_input}"})

        try:
            response = client.chat.completions.create(model="gemini-2.0-flash", messages=messages)
            answer = response.choices[0].message.content
        except Exception as e:
            answer = f"⚠️ Error from Gemini: {e}"

        st.session_state.chat_history.append((user_input, answer))

    for user_msg, bot_msg in st.session_state.chat_history:
        with st.chat_message("user"):
            st.markdown(user_msg)
        with st.chat_message("assistant"):
            st.markdown(bot_msg)

elif uploaded_files and not st.session_state.embedding_ready:
    st.info("⏳ Processing uploaded files... Please wait.")
else:
    st.warning("📌 Upload at least one supported file (.txt, .pdf, .docx) to start chatting.")
