import os
from dotenv import load_dotenv
load_dotenv() # .envファイルは親ディレクトリ方向に探索されるので同じフォルダでなくてもOK
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
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
    model="text-embedding-3-small"
    #model="text-embedding-3-large"
    )

# 動作確認
# python llm.pyを実行して返事が来ればOK
if __name__ == "__main__":
    res = llm.invoke("こんにちは")
    print(res)
