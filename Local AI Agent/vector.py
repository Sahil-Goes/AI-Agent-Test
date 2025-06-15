from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
import os
import pandas as pd

#reading csv file
df = pd.read_csv("realistic_restaurant_reviews.csv")
#defining the ollama model we will be using
embeddings = OllamaEmbeddings(model="mxbai-embed-large")

db_location = "./chroma_langchain_db"
add_documents = not os.path.exists(db_location)

#checking if location already exists
#if it doesn't then we create it by converting our data --> documents
if add_documents:
    documents = []
    ids = []

    for i, row in df.iterrows():
        document = Document(
            page_content= row["Title"] + " " + row["Review"],
            metadata = {"rating": row["Rating"], "date": row["Date"]},
            id=str(i)
        )

        ids.append(str(i))
        documents.append(document)

#initializing the vector store
vector_store = Chroma(
    collection_name="restaurant_reviews",
    persist_directory=db_location,
    embedding_function=embeddings
)

#If it doesnt exist then add documents to vector store
if add_documents:
    vector_store.add_documents(documents=documents, ids=ids)

#to grab relevant files
retriever = vector_store.as_retriever(
    search_kwargs = {"k":5}
)