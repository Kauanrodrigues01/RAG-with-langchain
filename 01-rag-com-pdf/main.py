import os
from pathlib import Path
from decouple import config
from langchain import hub
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import PyPDFLoader
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

pdf_path = Path(__file__).parent.parent / 'data' / 'laptop_manual.pdf'

pdf_loader = PyPDFLoader(file_path=pdf_path)

docs = pdf_loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
)
chunks = text_splitter.split_documents(documents=docs)

embedding = OpenAIEmbeddings(model='text-embedding-3-small')

vector_store = Chroma.from_documents(
    documents=chunks,
    embedding=embedding,
    collection_name='laptop_manual',
)

retriever = vector_store.as_retriever()

prompt = hub.pull('rlm/rag-prompt')

rag_chain = (
    {
        'context': retriever,
        'question': RunnablePassthrough(),  # Pede para o usuário inserir a pergunta
    }
    | prompt
    | llm
    | StrOutputParser()
)

question = 'Como funciona o cancelamento de ruido inteligente?'

response = rag_chain.invoke(question)

print(response)