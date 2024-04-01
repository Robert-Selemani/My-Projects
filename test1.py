import streamlit as st
import os 

#from langchain_community.document_loaders import WebBaseLoader

from langchain_community.document_loaders import PyPDFLoader

from langchain_community.vectorstores import Chroma

#from langchain_community import embeddings

from langchain_community.embeddings import OllamaEmbeddings

from langchain_community.llms import Ollama

from langchain_core.runnables import RunnablePassthrough

from langchain_core.output_parsers import StrOutputParser

from langchain_core.prompts import ChatPromptTemplate

from langchain.text_splitter import CharacterTextSplitter 

#Process PDFs


 # URL processing
def process_input(docs_list, question):
    model_local = Ollama(model="mistral")
    
    # Convert string of URLs to list
    #urls_list = urls.split("\n")
    #docs = [PyPDFLoader(pdf) for pdf in pdf_docs]
    #docs_list = [item for sublist in docs for item in sublist]
    
    #split the text into chunks
    
    text_splitter = CharacterTextSplitter.from_tiktoken_encoder(chunk_size=7500, chunk_overlap=100)
    doc_splits = text_splitter.split_documents(docs_list)
    
    #convert text chunks into embeddings and store in vector database

    vectorstore = Chroma.from_documents(
        documents=doc_splits,
        collection_name="rag-chroma",
        embedding=OllamaEmbeddings(model='nomic-embed-text'),
    )
    retriever = vectorstore.as_retriever()
    
    #perform the RAG 
    
    after_rag_template = """Answer the question based only on the following context:{context} Question:{question}"""
    #print(after_rag_template)

    after_rag_prompt = ChatPromptTemplate.from_template(after_rag_template)
    #print(after_rag_prompt)
    after_rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | after_rag_prompt
        | model_local
        | StrOutputParser()
    )

    return after_rag_chain.invoke(question)


    # streamlit UI

st.title(" Chat with your SOPs")
st.write("Enter SOPs and a question to query the documents")


#UI for input fields 

#urls = st.text_area("Enter URLs separated by a New Line", height =150)
UploadedFiles = st.file_uploader("Upload your SOPs here and click on 'Upload'", accept_multiple_files=True)

if st.button("Upload"):
    with st.spinner("Uploading SOPs"):
    #get the pdf text
        DocumentList = []
        for UploadedFile in UploadedFiles:
            with open(os.path.join("UploadedSOPs",UploadedFile.name), "wb") as f:
                f.write(UploadedFile.getbuffer())
            DocumentList.append(os.path.join("UploadedSOPs",UploadedFile.name))

        
        

        
        

question = st.text_input("Question")

# button = st.text_input("Question")

if st.button('Query Documents'):
    with st.spinner('Processing ....'):
       #print(DocumentList)
        DocumentList = os.listdir(r'UploadedSOPs')
        docs = [PyPDFLoader(os.path.join("UploadedSOPs",pdf)).load() for pdf in DocumentList]
        docs_list = [item for sublist in docs for item in sublist]


        answer =process_input(docs_list, question)
        #print(answer)
        st.text_area("Answer", value=answer, height=300)


#These comments should b removed before production
#streamlit run app.py --server.enableXsrfProtection false