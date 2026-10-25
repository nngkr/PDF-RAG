import os
import streamlit as st
from rag_utility import process_documents_to_chroma_db,answer_question

#set working_directory
working_dir=os.path.dirname(os.path.abspath((__file__)))

#streamlit page
st.title("🤖 - NNGKR RAG-Document")

#upload a file
uploader_file=st.file_uploader("Upload a File",type=["pdf"])

if uploader_file is not None:
    #define save path
    save_path=os.path.join(working_dir,uploader_file.name)
    #save the path
    with open(save_path,"wb") as f:
        f.write(uploader_file.getbuffer())
        
    process_document=process_documents_to_chroma_db(uploader_file.name)
    st.info("Document passs sucessfully")
    
#text widget to get user input
user_question=st.text_area("Ask your question about the document")

if st.button("Answer"):
    answer=answer_question(user_question)
    
    st.markdown("Krishna's response")
    st.markdown(answer)
    
