import os
import streamlit as st
from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import WebBaseLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts.chat import *
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
import json


load_dotenv()

api_key = st.secrets.api_credential.openai_key
os.environ['OPENAI_API_KEY'] = api_key

class WebsiteChatApp:
    col1,col2 = st.columns(2)
    """Class representing a Streamlit web app for chatting with predefined websites."""

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



    def load_predefined_websites(self):
        """Load predefined websites from the session file."""
        if "predefined_websites" not in self.session_state:
            self.session_state.predefined_websites = self.load_websites_from_session()

    def load_websites_from_session(self):
        """Load websites from the session file."""
        session_file_path = "websites_session.json"
        if os.path.exists(session_file_path):
            with open(session_file_path, "r") as f:
                return json.load(f)
        return [
            "https://hbsp.harvard.edu/home/",
            "https://hbr.org/",
            "https://www.accenture.com/us-en/insights/voices",
            "https://www.zs.com/insights",
            "https://www.mckinsey.com/featured-insights",
            "https://www.bcg.com/publications",
            "https://publishing.insead.edu/",
        ]

    def save_websites_to_session(self):
        """Save websites to the session file."""
        session_file_path = "websites_session.json"
        with open(session_file_path, "w") as f:
            json.dump(self.session_state.predefined_websites, f)

    def load_web_data(self, url):
        """Load web data from the given URL using a WebBaseLoader."""
        try:
            loader = WebBaseLoader(url)
            return loader.load()
        except Exception as e:
            return None

    def split_documents(self, data):
        """Split the web data into documents using CharacterTextSplitter."""
        try:
            text_splitter = CharacterTextSplitter(separator='\n', chunk_size=500, chunk_overlap=40)
            return text_splitter.split_documents(data)
        except Exception as e:
            return None

    def generate_individual_answers(self, prompt):
        """Generate answers for each individual site."""
        abs_path: str = os.path.dirname(os.path.abspath(__file__))
        db_dir: str = os.path.join(abs_path, "db")

        table_data = []

        for website_url in self.session_state.predefined_websites:
            data = self.load_web_data(website_url)
            if data:
                docs = self.split_documents(data)
                openai_embeddings = OpenAIEmbeddings()
                vectordb = Chroma.from_documents(documents=docs, embedding=openai_embeddings, persist_directory=db_dir)
                vectordb.persist()
                retriever = vectordb.as_retriever(search_kwargs={"k": 3})

                llm = ChatOpenAI(model_name='gpt-3.5-turbo')
                qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)

                response = qa(prompt)

                table_data.append({"Source": website_url, "Response": response})

        st.table(table_data)

    def generate_summarized_answer(self, prompt):
        """Generate a summarized answer taking reference from each site."""
        abs_path: str = os.path.dirname(os.path.abspath(__file__))
        db_dir: str = os.path.join(abs_path, "db")

        all_responses = []

        for website_url in self.session_state.predefined_websites:
            data = self.load_web_data(website_url)
            if data:
                docs = self.split_documents(data)
                openai_embeddings = OpenAIEmbeddings()
                vectordb = Chroma.from_documents(documents=docs, embedding=openai_embeddings, persist_directory=db_dir)
                vectordb.persist()
                retriever = vectordb.as_retriever(search_kwargs={"k": 3})

                llm = ChatOpenAI(model_name='gpt-3.5-turbo')
                qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)

                response = qa(prompt)

                if isinstance(response, dict) and "text" in response:
                    all_responses.append(response["text"])

        merged_text = "\n\n".join(all_responses)

        gpt_prompt = f"Act as a content Summarizer and give the best summarized answer for {prompt}" \
                     f"by taking reference from {merged_text}. Make sure output would be properly" \
                     f"formatted and meaningfull. If answer is not found, dont add anything by " \
                     f"your own, just return as not found"
        gpt_response = qa(gpt_prompt)

        if isinstance(gpt_response, dict):
            summarized_response = gpt_response["result"]

            st.markdown("<h2 style='text-align: center; color: blue;'>Summarized Answer:</h2>", unsafe_allow_html=True)
            st.write(summarized_response)

    with col1:
        def add_website(self):
            """Add more websites dynamically."""
            website_url = st.text_input("Enter Website URL")

            if st.button("Add Website", type="primary"):
                if website_url:
                    self.session_state.predefined_websites.append(website_url)
                    self.save_websites_to_session()
                    st.success(f"Website {website_url} added successfully!")


def main():
    """Main function to run the Streamlit web app."""
    st.markdown("<h1 style='text-align: center; color: yellow;'>WebAnswerHub</a></h1>",
                unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: center; color:red;'>Enter your question (query/prompt) below 👇</h3>",
                unsafe_allow_html=True)

    website_chat_app = WebsiteChatApp()

    st.subheader("Predefined Websites:")
    for website_url in website_chat_app.session_state.predefined_websites:
        st.write(f"- {website_url}")

    st.subheader("Add More Websites:")
    website_chat_app.add_website()

    prompt = st.text_input("Ask a question (query/prompt)")

    if st.button("Generate Individual Answers", type="primary"):
        website_chat_app.generate_individual_answers(prompt)

    if st.button("Generate Summarized Answer", type="primary"):
        website_chat_app.generate_summarized_answer(prompt)


if __name__ == '__main__':
    main()
