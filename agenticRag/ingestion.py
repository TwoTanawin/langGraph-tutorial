from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.document_loaders import WebBaseLoader
# from langchain_openai import OpenAIEmbeddings
from langchain_aws import BedrockEmbeddings
import boto3

from dotenv import load_dotenv
import os

load_dotenv()

# Set USER_AGENT from .env
os.environ["USER_AGENT"] = os.getenv("USER_AGENT", "AgenticRAGBot/1.0") 

session = boto3.Session(profile_name="hydroneo", region_name="ap-southeast-1")
bedrock_client = session.client("bedrock-runtime")

urls = [
    "https://lilianweng.github.io/posts/2023-06-23-agent/",
    "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
    "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
]

all_docs = []
for url in urls:
    docs = WebBaseLoader(url).load()
    all_docs.extend(docs)

docs_list = all_docs
# for sublist in docs:
#     for item in sublist:
#         docs_list.append(item)
        
text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=250, chunk_overlap=0
)
doc_splits = text_splitter.split_documents(docs_list)

embeddings = BedrockEmbeddings(
    model_id="cohere.embed-english-v3",
    client=bedrock_client
)

# vectorstore = Chroma.from_documents(
#     documents=doc_splits,
#     collection_name="rag-chroma",
#     embedding=embeddings,
#     persist_directory="./.chroma",
# )

retriever = Chroma(
    collection_name="rag-chroma",
    persist_directory="./.chroma",
    embedding_function=embeddings,
).as_retriever()