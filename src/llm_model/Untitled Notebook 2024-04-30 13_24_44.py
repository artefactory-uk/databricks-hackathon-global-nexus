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
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain_community.embeddings.sentence_transformer import (
    SentenceTransformerEmbeddings,
)
import chromadb

chroma_client = chromadb.Client()
chroma_client = chromadb.PersistentClient(
        path="src/vector_database/chroma_data_langchain_test",
         )


# COMMAND ----------

collection = chroma_client.get_collection("langchain_database")
# collection = chroma_client.create_collection("langchain_database")

# COMMAND ----------

collection.add(
    documents=[row["PAPER_TITLE"] + row["PAPER_ABSTRACT"] for idx, row in papers_df.iterrows()], 
    ids=[row["PAPER_LINK"] for idx, row in papers_df.iterrows()], 
)

# COMMAND ----------

from transformers import BertConfig, BertModel, BertTokenizer

# Initializing a BERT google-bert/bert-base-uncased style configuration
configuration = BertConfig()

# Initializing a model (with random weights) from the google-bert/bert-base-uncased style configuration
model = BertModel(configuration)

# Accessing the model configuration
configuration = model.config

# COMMAND ----------

from langchain.vectorstores import Chroma
#from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_community.embeddings.sentence_transformer import (
    SentenceTransformerEmbeddings,
)

# create the open-source embedding function
embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

chroma_db = Chroma(persist_directory="src/vector_database/chroma_data_langchain_test", embedding_function=embedding_function, collection_name="langchain_database")

# COMMAND ----------

query = "Results about methamphetamine"
print(chroma_db.similarity_search(query))

# COMMAND ----------

!pip install -U sentence-transformers

# COMMAND ----------

model_names = ['dbrx_instruct']
dbutils.widgets.dropdown("model_name", model_names[0], model_names)


# COMMAND ----------

# Default catalog name when installing the model from Databricks Marketplace.
# Replace with the name of the catalog containing this model
catalog_name = "databricks_dbrx_models"

# You should specify the newest model version to load for inference
version = "1"
model_name = dbutils.widgets.get("model_name")
model_uc_path = f"{catalog_name}.models.{model_name}"
endpoint_name = f'{model_name}_marketplace'

# Minimum desired provisioned throughput
min_provisioned_throughput = 500

# Maximum desired provisioned throughput
max_provisioned_throughput = 1000

# COMMAND ----------

import requests
import json

# Get the API endpoint and token for the current notebook context
API_ROOT = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiUrl().get()
API_TOKEN = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()

# COMMAND ----------

API_TOKEN

# COMMAND ----------

from huggingface_hub import login
login()

# COMMAND ----------

from sentence_transformers import SentenceTransformer
from langchain_core.runnables.base import Runnable
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel
import torch

access_token = "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"
# tokenizer = AutoTokenizer.from_pretrained("databricks/dbrx-instruct", token=access_token)
# model = AutoModelForCausalLM.from_pretrained("databricks/dbrx-instruct", device_map="auto", torch_dtype=torch.bfloat16, token=access_token)

# model = AutoModel.from_pretrained("databricks/dbrx-instruct", token=access_token)

from langchain.llms import Databricks
from langchain_core.messages import HumanMessage, SystemMessage



llm_model = Databricks(endpoint_name="databricks-dbrx-instruct")

chain = RetrievalQA.from_chain_type(llm=llm_model,
                                    chain_type="stuff",
                                    retriever=chroma_db.as_retriever())

# COMMAND ----------

response = chain("Can you tell me any research about methamphetamine?")

# COMMAND ----------

db_rec = langchainChroma.get()

# COMMAND ----------

!pip install --upgrade transformers

# COMMAND ----------

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

tokenizer = AutoTokenizer.from_pretrained("databricks/dbrx-instruct", token=access_token)
model = AutoModelForCausalLM.from_pretrained("databricks/dbrx-instruct", device_map="auto", torch_dtype=torch.bfloat16, token=access_token)

input_text = "What does it take to build a great LLM?"
messages = [{"role": "user", "content": input_text}]
input_ids = tokenizer.apply_chat_template(messages, return_dict=True, tokenize=True, add_generation_prompt=True, return_tensors="pt").to("cuda")

outputs = model.generate(**input_ids, max_new_tokens=200)
print(tokenizer.decode(outputs[0]))


# COMMAND ----------


