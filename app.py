# import subprocess
import nest_asyncio
import streamlit as st
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_community.document_loaders import TextLoader
import pdfminer.high_level
import tempfile
import os

nest_asyncio.apply()


# -------------------------------------------------------

def main():
    st.title('I am your AI SOP Chatbot')

    menu = ['Ask the bot', 'About the bot']
    choice = st.sidebar.selectbox('Menu', menu)

    if choice == 'Ask the bot':
        st.subheader('I am happy to be helping you today!')

        uploaded_file = st.file_uploader('Upload PDFs:')
        if uploaded_file is not None:
            # Read the file:
            uploaded_pdfs = uploaded_file.getvalue()
            st.write('File successfully uploaded!')

            uploaded_pdfs_bytes = uploaded_pdfs

            # Create a temporary file with a name
            def extract_text_from_pdf(uploaded_pdfs_bytes):
                with tempfile.NamedTemporaryFile(delete=False, dir=tempfile.gettempdir()) as temp_pdf:
                    temp_pdf.write(uploaded_pdfs_bytes)
                    temp_pdf.seek(0)
                    text = pdfminer.high_level.extract_text(temp_pdf.name)  # Use the temporary file's name
                return text

            query = st.text_input("Ask me")

            # Button to process input
            if st.button('Send'):
                with st.spinner('Processing...'):
                    text = extract_text_from_pdf(uploaded_pdfs)  # Extract text
                    answer = process_input(text, query)  # Pass extracted text
                    st.success("Answer generated!")  # Example success message
                    st.text_area("Answer", value=answer, height=300, disabled=True)

    else:
        st.subheader('About the bot')
        st.caption('I am an SOP Chatbot. Ask me anything from your uploaded SOP documents '
                   'and I will give you a summarised answer with references to relevant '
                   'document(s)')


def extract_text_from_pdf(uploaded_pdfs_bytes):
    txt = pdfminer.high_level.extract_text(uploaded_pdfs_bytes)  # Extract txt using pdfminer.six
    return txt  # Return extracted txt


# PDF Processing
def process_input(txt, query):
    with tempfile.NamedTemporaryFile(delete=False, dir=tempfile.gettempdir()) as temp_file:
        # temp_file.write(txt.encode('utf-8'))
        file_path = os.path.join(tempfile.gettempdir(), temp_file.name)

    # Pass the file path to TextLoader
    loader = TextLoader(file_path, encoding="utf-8")
    docs_list = loader.load()
    # text_splitter = CharacterTextSplitter
    text_splitter = CharacterTextSplitter.from_tiktok_encoder(chunk_size=7500, chunk_overlap=100)
    doc_splits = text_splitter.split_documents(docs_list)  # Should work now
    model_local = Ollama(model="mistral")

    # convert text chunks into embeddings and store in vector database

    vectorstore = Chroma.from_documents(
        documents=doc_splits,
        collection_name="rag-chroma",
        embedding=OllamaEmbeddings(model='nomic-embed-text'),
    )
    retriever = vectorstore.as_retriever()

    # perform the RAG

    after_rag_template = """Answer the question based only on the following context:
    {context}
    Question: {query}
    """
    after_rag_prompt = ChatPromptTemplate.from_template(after_rag_template)
    after_rag_chain = (
            {"context": retriever, "query": RunnablePassthrough()}
            | after_rag_prompt
            | model_local
            | StrOutputParser()
    )
    return after_rag_chain.invoke(query)


if __name__ == '__main__':
    main()

# -------------------------------------------------------
