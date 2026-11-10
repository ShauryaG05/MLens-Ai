
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
import os
from dotenv import load_dotenv

load_dotenv()


embeddings = HuggingFaceEmbeddings(
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
)

print("Document Loaded")
loader = PyPDFLoader("./Docs.pdf")

doc = loader.load()

# google/embeddinggemma-2


splitter = RecursiveCharacterTextSplitter(chunk_size = 500, chunk_overlap=0)

texts = splitter.split_documents(doc)

print("Splited the resources successfully")
print(len(texts))

vector_store = Chroma.from_documents(
    documents = texts,
    collection_name="pdf_collection",
    embedding=embeddings,
    persist_directory="/Users/shaurya/Desktop/Directory/Root/ChatBot/"

)

print("Stored Document In Vector DataBase :)")