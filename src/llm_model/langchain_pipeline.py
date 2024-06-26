# Databricks notebook source
# Adding Langchain to ChromaDB

# COMMAND ----------

import os
os.chdir("../..")

# COMMAND ----------

os.getcwd()

# COMMAND ----------

import pandas as pd
DATA_DIR = 'src/data'
RESEARCH_PAPER_DB = f'{DATA_DIR}/papers_titles_&_abstracts_1470.csv'
papers_df = pd.read_csv(RESEARCH_PAPER_DB)
papers_df.head()

# COMMAND ----------

from langchain.chains import RetrievalQA
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
import chromadb

chroma_client = chromadb.Client()
chroma_client = chromadb.PersistentClient(
        path="src/data/chroma_vector_database",
         )


# COMMAND ----------

collection = chroma_client.get_collection("medical_research_papers")
# collection = chroma_client.create_collection("medical_research_papers")

# COMMAND ----------

collection.add(
    documents=[row["PAPER_TITLE"] + " " + row["PAPER_ABSTRACT"] for idx, row in papers_df.iterrows()], 
    ids=[row["PAPER_LINK"] for idx, row in papers_df.iterrows()], 
    metadatas=[
        {
            "title": row["PAPER_TITLE"],
            "abstract": row["PAPER_ABSTRACT"],
            "url": row["PAPER_LINK"], 
            } 
        for _, row in papers_df.iterrows()
        ]
)

# COMMAND ----------

from langchain.vectorstores import Chroma
# from langchain.embeddings.openai import OpenAIEmbeddings
from langchain_community.embeddings.sentence_transformer import (
    SentenceTransformerEmbeddings,
)
API = os.getenv("OPENAI_API_KEY")
# create the open-source embedding function, lemonfox does not support openai embeddings 
# embedding_function = OpenAIEmbeddings(
#     api_key=API,
#     base_url="https://api.lemonfox.ai/v1"
# )

# create the open-source embedding function
embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

chroma_db = Chroma(persist_directory="src/data/chroma_vector_database", embedding_function=embedding_function, collection_name="medical_research_papers")

# COMMAND ----------

query = "Results about methamphetamine"
print(chroma_db.similarity_search(query))

# COMMAND ----------

from langchain.chat_models import ChatOpenAI

API = os.getenv("OPENAI_API_KEY")
model = ChatOpenAI(
    api_key=API,
    base_url="https://api.lemonfox.ai/v1"
)

chain = RetrievalQA.from_chain_type(llm=model,
                                    chain_type="stuff",
                                    retriever=chroma_db.as_retriever())

# COMMAND ----------

response = chain("Can you tell me where I should focus my research into heart conditons given I am looking for a PhD topic?")

# COMMAND ----------

response

# COMMAND ----------

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

torch.random.manual_seed(0)

model = AutoModelForCausalLM.from_pretrained(
    "microsoft/Phi-3-mini-128k-instruct", 
    device_map="cuda", 
    torch_dtype="auto", 
    trust_remote_code=True, 
)
tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-128k-instruct")

messages = [
    {"role": "user", "content": "Can you provide ways to eat combinations of bananas and dragonfruits?"},
    {"role": "assistant", "content": "Sure! Here are some ways to eat bananas and dragonfruits together: 1. Banana and dragonfruit smoothie: Blend bananas and dragonfruits together with some milk and honey. 2. Banana and dragonfruit salad: Mix sliced bananas and dragonfruits together with some lemon juice and honey."},
    {"role": "user", "content": "What about solving an 2x + 3 = 7 equation?"},
]

pipe = pipeline("text-generation", model=model,tokenizer=tokenizer,)

generation_args = {
    "max_new_tokens": 500,
    "return_full_text": False,
    "temperature": 0.0,
    "do_sample": False,
}

output = pipe(messages, **generation_args)
print(output[0]['generated_text'])

