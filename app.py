import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader

from dotenv import load_dotenv
load_dotenv()

## load GROQ API KEY
#os.environ['OPENAI_API_KEY']=os.getenv("OPENAI_API_KEY")
os.environ['GROQ_API_KEY']=os.getenv("GROQ_API_KEY")

groq_api_key=os.getenv("GROQ_API_KEY")

llm=ChatGroq(groq_api_key=groq_api_key, model_name="Llama3-8b-8292")

prompt=ChatPromptTemplate.from_template(
    """
    Answer the questions based on the provided context only.
    Please provide the most accurate response based on the question
    <context>
    {context}
    <context>
    Question:{input}
    """
)

def create_vector_embedding():
    if "vectors" not in st.session_state:
        st.session_state.embeddings=OllamaEmbeddings()
        ## The below one is Data Ingestion step
        st.session_state.loader=PyPDFDirectoryLoader("research_papers")
        ##after ingestion will load the document
        st.session_state.docs=st.session_state.loader.load()
        ## Chunk size means that in each chunk it will hold 1000 chars
        st.session_state.text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
        st.session_state.final_documents=st.session_state.text_splitter.split_documents(st.session_state.docs[:50])
        ## Storing the vector in vector databases
        st.session_state.vectors=FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)
        
        
user_prompt=st.text_input("Enter your Query from the research papers")

if st.button("Document Embedding"):
    create_vector_embedding()
    st.write("Vector Database is ready")
    
import time

if user_prompt:
    ## stuffing all docs to LLM
    document_chain=create_stuff_documents_chain(llm,prompt)
    ## will return the most relevant document chunks based on similarity
    retriever=st.session_state.vectors.as_retriever()
    ## Now will use both and create a retiever chain 
    retrieval_chain=create_retrieval_chain(retriever, document_chain)
    
    ## will understand the power of grok API
    start=time.process_time()
    response=retrieval_chain.invoke({'input':user_prompt})
    print(f"Response Time :{time.process_time()-start}")
    
    st.write(response['answer'])
    
    ## with Streamlit expander
    with st.expander("Document similairity Search"):
        for i, doc in enumerate(response['contect']):
            st.write(doc.page_content)
            st.write('--------------------------')
        