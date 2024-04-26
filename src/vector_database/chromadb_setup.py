# Databricks notebook source
!pip freeze

# COMMAND ----------

import pandas as pd

# COMMAND ----------

df = pd.read_csv('papers_titles_&_abstracts_1470.csv')

# COMMAND ----------

!chroma run

# COMMAND ----------

import chromadb
chroma_client = chromadb.Client()
collection = chroma_client.create_collection("vector_database")


# COMMAND ----------

# Add docs to the collection. Can also update and delete. Row-based API coming soon!
collection.add(
    documents=[df.iloc[0]["PAPER_TITLE"], df.iloc[1]["PAPER_TITLE"]], # we embed for you, or bring your own
    metadatas=[{"source": "notion"}, {"source": "google-docs"}], # filter on arbitrary metadata!
    ids=["doc1", "doc2"], # must be unique for each doc 
)

# COMMAND ----------

document_list = [row["PAPER_TITLE"] + " " + row["PAPER_ABSTRACT"] for idx, row in df.iterrows()]

# COMMAND ----------

document_ids = [row["PAPER_LINK"] for idx, row in df.iterrows()]

# COMMAND ----------

collection.add(
    documents=document_list,
    ids=document_ids,
)

# COMMAND ----------

df.head()


# COMMAND ----------

results = collection.query(
    query_texts=["Give me any results about methamphetaminea"],
    n_results=5
    # where={"metadata_field": "is_equal_to_this"}, # optional filter
    # where_document={"$contains":"search_string"}  # optional filter
) 

# COMMAND ----------

results

# COMMAND ----------

results["documents"][0][1]

# COMMAND ----------


