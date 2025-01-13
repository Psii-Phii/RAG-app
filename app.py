from dotenv import load_dotenv
import os
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
def main():
    load_dotenv()
    st.set_page_config(page_title="Chat PDF")
    st.header("Chat PDF")

    # text processing 
    pdf = st.file_uploader("upload PDF here")

    if pdf is not None:
        pdf_reader = PdfReader(pdf)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        

        text_split  = CharacterTextSplitter(separator = "\n",
        chunk_size = 1000,
        chunk_overlap = 500, length_function = len)

        #split the orignal text 
        chunk = text_split.split_text(text)
        
        #embeddings 
        embeding = OpenAIEmbeddings()
        info = FAISS.from_texts(chunk, embeding)  #semantic index

        query  = st.text_input("Please input your question:")
        if query:
            rel_info   = info.similarity_search(query) #find relevant info
            chain  =  load_qa_chain(llm = OpenAI(), chain_type="stuff")
            response = chain.run(input_documents= rel_info,  question=query) #handing it over to GPT
            st.write(response)
if __name__ == '__main__':
    main()