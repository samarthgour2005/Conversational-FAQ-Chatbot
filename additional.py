# create_db.py
import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient, models
from langchain_qdrant import Qdrant
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

load_dotenv()

client=QdrantClient(
    url=os.getenv("QDRANT_HOST"),
    api_key=os.getenv("QDRANT_API_KEY")
)

vector_config=models.VectorParams(size= 768, distance=models.Distance.COSINE)

client.recreate_collection(
    collection_name=os.getenv("QDRANT_COLECTION_NAME"),
    vectors_config=vector_config
)

embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

vector_store=Qdrant(
    client=client,
    collection_name=os.getenv("QDRANT_COLECTION_NAME"),
    embeddings=embeddings
)


loader1 = PyPDFLoader("file1")
docs1 = loader1.load()

# Load second PDF
loader2 = PyPDFLoader("file2")
docs2 = loader2.load()

# Combine all documents
docs = docs1 + docs2

text_split=RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=30    
)
text=text_split.split_documents(docs)

print(f"Total documents chunks: {len(text)}")

metadata=[
# personal meta data according to the requirement and the number of the chunks created
]
for i,doc in enumerate(text):
    doc.metadata['topic']=metadata[i]
    # print(f"metadata {i+1} :",doc.metadata['topic'] ,end="  ")

vector_store.add_documents(text)
print("Data added to Qdrant DB successfully.")