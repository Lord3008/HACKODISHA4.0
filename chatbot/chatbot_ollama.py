# from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import Ollama
import streamlit as st
import os
from dotenv import load_dotenv

load_dotenv()

os.environ["LANGCHAIN_TRACKING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")

## PROMPT TEMPLATE

prompt = ChatPromptTemplate.from_messages(
    [
        ("system","You are a helpful answer. Please response to the user questions."),
        ("user","Question:{question}")

    ]
)

# streamlit framework
st.title('Cryptex Users Support ')
st.title('AI CHATBOT')
input_text = st.text_input("Search the topic u want related Cryptex")

# ollama LLM
llm = Ollama(model = "llama2")
output_parser = StrOutputParser()
chain = prompt|llm|output_parser

if input_text:
    st.write(chain.invoke({'question':input_text}))