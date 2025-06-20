import os
from pathlib import Path
from decouple import config
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

OPENAI_API_KEY = config('OPENAI_API_KEY')
os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY

pdf_path = Path(__file__).parent.parent / 'data' / 'laptop_manual.pdf'

pdf_loader = PyPDFLoader(pdf_path)

chunks = pdf_loader.load_and_split(text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200))

embendding_model = OpenAIEmbeddings(model='text-embedding-3-small')

vector_store_path = Path(__file__).parent / 'vector-store'

vector_store = Chroma.from_documents(
    documents=chunks,
    embedding=embendding_model,
    persist_directory=str(vector_store_path),
    collection_name='laptop-manual'
)