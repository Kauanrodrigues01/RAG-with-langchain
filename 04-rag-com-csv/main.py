import os
from pathlib import Path
from decouple import config
from langchain import hub
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

OPENAI_API_KEY = config('OPENAI_API_KEY')
os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY

llm = ChatOpenAI(
    model='gpt-3.5-turbo',
    temperature=0.0,
    max_tokens=1000,
)

csv_path = Path(__file__).parent.parent / 'data' / 'carros.csv'

csv_loader = CSVLoader(csv_path)

chunks = csv_loader.load_and_split(text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200))

embendding_model = OpenAIEmbeddings(model='text-embedding-3-small')

vector_store_path = Path(__file__).parent / 'vector-store'

vector_store = Chroma.from_documents(
    documents=chunks,
    embedding=embendding_model,
    persist_directory=str(vector_store_path),
    collection_name='carros'
)

retrieve = vector_store.as_retriever()

rag_prompt = hub.pull('rlm/rag-prompt')

rag_chain = (
    {
        'context': retrieve,
        'question': RunnablePassthrough()
    }
    | rag_prompt
    | llm
    | StrOutputParser()
)

response = rag_chain.invoke('Quais os modelos de carros tem?')

print(response)