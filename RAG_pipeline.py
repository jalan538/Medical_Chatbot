from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS


# step 1: load the raw pdf file

DATA_DIR = "data/"
def laod_pdf_files(data):
    loader = DirectoryLoader(data,
                            glob="*.pdf", 
                            loader_cls=PyPDFLoader)
    documents = loader.load()
    return documents

documents = laod_pdf_files(data = DATA_DIR)

print(f"length of PDF page: {len(documents)} ")

#print(f"Loaded {len(documents)} documents from {DATA_DIR}")

#step 2: create the documents into chunks

def create_chunks(extracred_documents):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_documents(extracred_documents)
    return chunks

text_chunks = create_chunks(extracred_documents=documents)


print(f"length of text chunks: {len(text_chunks)} ")



# step 3: create embeddings model

def get_embeddings_model():
    embeddings_model = HuggingFaceEmbeddings(model_name ="sentence-transformers/all-MiniLM-L6-v2")
    return embeddings_model

embeddings_model = get_embeddings_model()

print(f"Embeddings model: {embeddings_model} ")



# step 4 : Store the embeddings into vector database using FAISS

DB_FAISS_PATH = "vectorestore/db_faiss"
db = FAISS.from_documents(text_chunks, embeddings_model)
db.save_local(DB_FAISS_PATH)

