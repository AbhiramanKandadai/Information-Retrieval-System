import os
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
#from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings


from langchain_community.vectorstores import FAISS
#from langchain.embeddings import GooglePalmEmbeddings
#from langchain_community.llms import GooglePalm
#from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
#from langchain.chains import ConversationalRetrievalChain
#from langchain_classic.chains import RetrievalQA, LLMChain

from langchain_classic.chains import ConversationalRetrievalChain
from langchain_classic.memory import ConversationBufferMemory
from dotenv import load_dotenv

load_dotenv()
GOOGLE_API_KEY = os.getenv("GROQ_API_KEY")
os.environ['GOOGLE_API_KEY'] = GOOGLE_API_KEY
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")



def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap = 20)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    embeddings = HuggingFaceEmbeddings()
    vector_store = FAISS.from_texts(text_chunks, embedding = embeddings)
    return vector_store

def get_conversationa_chain(vector_store):
    llm = ChatGroq(
    model="llama-3.1-8b-instant",
)
    memory = ConversationBufferMemory(memory_key = "chat_history", return_messages = True)
    conversation_chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever = vector_store.as_retriever(), memory = memory)
    return conversation_chain

