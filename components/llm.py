import os
from dotenv import load_dotenv
import streamlit as st

# Try to load from .env file for local development
load_dotenv()

# Get API key from Streamlit secrets if available, otherwise from environment variables
def get_api_key():
    # First try to get from Streamlit secrets
    try:
        return st.secrets["OPENAI_API_KEY"]
    except (KeyError, FileNotFoundError):
        # Fall back to environment variable
        return os.environ.get('OPENAI_API_KEY')

OPENAI_API_KEY = get_api_key()

from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings

# ChatOpenAI
llm = ChatOpenAI(
    model="gpt-4o-mini", 
    temperature=0,
    api_key=OPENAI_API_KEY
)

# Embedding モデル
oai_embeddings = OpenAIEmbeddings(
    #好きなモデルをコメントアウトして利用
    #model="text-embedding-ada-002"
    model="text-embedding-3-small",
    #model="text-embedding-3-large"
    api_key=OPENAI_API_KEY
)

# 動作確認
# python llm.pyを実行して返事が来ればOK
if __name__ == "__main__":
    res = llm.invoke("こんにちは")
    print(res)
