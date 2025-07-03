import os
import streamlit as st
import pickle
import time
from langchain_google_genai import ChatGoogleGenerativeAI
from secret_key import GOOGLE_API_KEY
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.chains.qa_with_sources.loading import load_qa_with_sources_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS

from dotenv import load_dotenv
load_dotenv()

st.title("Equity News Research Tool ⚖️")
st.sidebar.title("News Article URLs")

urls = []
for i in range(3):
    url = st.sidebar.text_input(f"URL {i+1}")
    urls.append(url)

process_url_clicked = st.sidebar.button("Process URLs")

main_placeholder = st.empty()
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.9, max_output_tokens=500)

embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=GOOGLE_API_KEY
    )


if process_url_clicked:
    # load Data
    loader = UnstructuredURLLoader(urls=urls)
    main_placeholder.text("Data Loading....Started...✅✅✅")
    data = loader.load()
    # split data
    text_splitter = RecursiveCharacterTextSplitter(
        separators=['\n\n', '\n', '.', ','],
        chunk_size = 1000
    )
    main_placeholder.text("Text Splitter...Started...✅✅✅")
    docs = text_splitter.split_documents(data)
    # create embeddings and save it to FAISS Index
    
    vectorstore_google = FAISS.from_documents(docs, embeddings)
    main_placeholder.text("Embedding Vector Started Building...✅✅✅")
    time.sleep(2)

    vectorstore_google.save_local("faiss_index_store")

index_file_path = os.path.join("faiss_index_store", "index.faiss")

if os.path.exists(index_file_path):
    vectorindex_google = FAISS.load_local(
        folder_path="faiss_index_store",
        embeddings=embeddings,
        allow_dangerous_deserialization=True
    )

    chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=vectorindex_google.as_retriever())
    
    query = main_placeholder.text_input("Question: ")
    if query:
        result = chain({"question": query}, return_only_outputs=True)

        st.header("Answer")
        st.write(result["answer"])

        sources = result.get("sources", "")
        if sources:
            st.subheader("Sources:")
            for source in sources.split("\n"):
                st.write(source)
else:
    st.warning("Please click 'Process URLs' first to generate the FAISS index.")
