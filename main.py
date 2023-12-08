from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import LlamaCppEmbeddings
from langchain.vectorstores import chroma

from flask import Flask

loader = TextLoader("./docs/raw.txt")
docs = loader.load()

text_splitter = CharacterTextSplitter(chunk_overlap=0)
texts = text_splitter.split_documents(docs)

_texts = []
for i in range(len(texts)):
    _texts.append(texts[i].page_content)
    
embeddings = LlamaCppEmbeddings(model_path="./models/llama-7b.ggmlv3.q4_0.bin")

db = chroma.Chroma.from_documents(texts, embeddings)

app = Flask(__name__)

@app.route("/")
def hello_world():
    return "<p>Hello, World!</p>"

@app.route("/query")
def query():
    query = "Peux-tu générer un question à partir du texte, et me donner 4 réponses dont seulement 1 est vrai ?"
    query_vector = embeddings.embed_query(query)
    docs = db.similarity_search_by_vector(query_vector, k=1)
    return docs[0].page_content