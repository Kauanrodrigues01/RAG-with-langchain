import os
from pathlib import Path
from decouple import config
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

OPENAI_API_KEY = config('OPENAI_API_KEY')
os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY

llm = ChatOpenAI(
    model='gpt-3.5-turbo',
    temperature=0.0,
    max_tokens=1000,
)

persist_directory = Path(__file__).parent / 'vector-store'
embedding = OpenAIEmbeddings(model='text-embedding-3-small')

vector_store = Chroma(
    persist_directory=str(persist_directory),
    embedding_function=embedding,
    collection_name='laptop-manual',
)
retriever = vector_store.as_retriever()

system_prompt = '''
Use o contexto para responder as perguntas.
Contexto: {context}
'''
prompt = ChatPromptTemplate.from_messages(
    [
        ('system', system_prompt),
        ('human', '{input}'),
    ]
)
question_answer_chain = create_stuff_documents_chain(
    llm=llm,
    prompt=prompt,
)
chain = create_retrieval_chain(
    retriever=retriever,
    combine_docs_chain=question_answer_chain,
)

query = 'Qual a marcado e modelo do notebook?'

response = chain.invoke(
    {'input': query},
)
print(response)
