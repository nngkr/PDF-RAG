import os
from dotenv import load_dotenv
from langchain_classic.chains import retrieval_qa
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import (create_stuff_documents_chain)
from langchain_core.prompts import ChatPromptTemplate

from langchain_community.document_loaders import UnstructuredPDFLoader


#Load the environment

load_dotenv()

#working directory

working_dir=os.path.dirname(os.path.abspath((__file__)))

#Load embedding model
embedding = HuggingFaceEmbeddings()

#load the llm model(openai/gpt-oss-20b) from Groq
groq_api_key=(os.getenv("GROQ_API_KEY"))
if not groq_api_key:
    raise ValueError(
        "GROQ_API_KEY is missing .And it to Streamlit Secretes"
    )


llm= ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0
)


# Document - Engiction

def process_documents_to_chroma_db(file_name):
    Loader=UnstructuredPDFLoader(f"{working_dir}/{file_name}")
    documents=Loader.load()
    #splite the text into chunks for embeddingd
    text_splitters=RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    texts=text_splitters.split_documents(documents)
    # store the document chunks in Chroma vector db
    vector_db = Chroma.from_documents(
        documents=texts,
        embedding=embedding,
        persist_directory=f"{working_dir}/doc_vectorstore"
    )
    return 0

# question to answer

def answer_question(user_question):
    #Load the persist Chroma vector db
    vector_db=Chroma(
        persist_directory=(f"{working_dir}/doc_vectorstore"),
        embedding_function=embedding
    )
    
    # Creat a retriver for document search
    
    retriever =vector_db.as_retriever()
    
    
    # Create a RetrievalQA chain to answer to question using llm model
    
    prompt = ChatPromptTemplate.from_messages([
    ("system",     "Answer the question using the following context:\n\n{context}"),
    ("human", "{input}")
    ])

    question_answer_chain = create_stuff_documents_chain(
        llm, prompt
    )

    rag_chain = create_retrieval_chain(
        retriever, question_answer_chain
    )

    response = rag_chain.invoke({
        "input": user_question
    })

    return response["answer"]
