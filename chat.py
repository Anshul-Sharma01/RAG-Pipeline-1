from pyexpat import model

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_huggingface import HuggingFaceEmbeddings 
from langchain_qdrant import QdrantVectorStore
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os

load_dotenv()


embedding_model = HuggingFaceEmbeddings(
    model_name ="all-MiniLM-L6-v2"
)

vector_db = QdrantVectorStore.from_existing_collection(
    embedding = embedding_model,
    url = "http://localhost:6333",
    collection_name = "First-RAG-Pipeline"
)

# User Input

user_query = input("What do you want to do ??")

query_result = vector_db.similarity_search(query = user_query)

context = "\n\n\n".join([f"Page Content : {result.page_content}\nPage Number : {result.metadata['page_label']}\nFile Location : {result.metadata['source']}"
for result in query_result])


SYSTEM_PROMPT = f"""
You are a helpful AI Assistant that answers user queries based solely on the provided context retrieved from a PDF file, including the page contents and their page numbers.

You must answer only using the given context and, whenever applicable, guide the user to the specific page number(s) where they can find more detailed information.

Context:
{context}
"""


llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
)

messages = [
    SystemMessage(content=SYSTEM_PROMPT),
    HumanMessage(content=user_query)
]

response = llm.invoke(messages)
print("Response : ", response.content)