import os
import streamlit as st
st.set_page_config(layout="wide", initial_sidebar_state="collapsed")

from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import WebBaseLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts.chat import *
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
import json
from st_aggrid import AgGrid, GridOptionsBuilder
from demo import WebsiteChatApp



load_dotenv()

api_key = st.secrets.api_credential.openai_key

def __init__(self):
        """Initialize the WebsiteChatApp instance."""
        self.OPENAI_API_KEY = api_key
        self.system_template = """
            Use the following pieces of context to answer the user's question.
            If you don't know the answer, just say that you don't know, 
            don't try to make up an answer by your own.
        """
        self.messages = [
            SystemMessagePromptTemplate.from_template(self.system_template),
            HumanMessagePromptTemplate.from_template("{question}"),
        ]
        self.prompt = ChatPromptTemplate.from_messages(self.messages)
        self.chain_type_kwargs = {"prompt": self.prompt}
        self.session_state = st.session_state
        self.load_predefined_websites()

col1,col2 = st.columns(2)

with col1:
    st.subheader("Please Enter Website URL")
    col4,col5 = st.columns((3,1))
    with col4:
        website_url = st.text_input("Enter Website URL", key="user_input", label_visibility="collapsed")
        
    with col5:
        if st.button("Add Website", type="primary"):
            if website_url:
                self.session_state.predefined_websites.append(website_url)
                self.save_websites_to_session()
                st.success(f"Website {website_url} added successfully!")

    website_chat_app = WebsiteChatApp()

    st.subheader("Predefined Websites:")
    for website_url in website_chat_app.session_state.predefined_websites:
        st.write(f"- {website_url}")

with col2:
    st.subheader("Ask a question")
    col6,col7 = st.columns((4,1))
    with col6:
        prompt = st.text_input("Ask a question", key="query", label_visibility="collapsed")
    col8, col9 = st.columns((1,3.75))
    with col8:
        if st.button("Generate Individual Answers", type="primary"):
            website_chat_app.generate_individual_answers(prompt)
    with col9:
        if st.button("Generate Summarized Answer", type="primary"):
            website_chat_app.generate_summarized_answer(prompt)