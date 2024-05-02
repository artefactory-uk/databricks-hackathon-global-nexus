import gradio as gr
import pandas as pd
import chromadb
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings.sentence_transformer import (
    SentenceTransformerEmbeddings,
)
from langchain_community.chat_models import ChatOpenAI
from transformers import pipeline
from dotenv import load_dotenv
import os
from src.constants.paths import VECTOR_DATABASE_PATH
from src.constants.parameters import VECTOR_DATABASE_NAME

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
BASE_URL = os.getenv("BASE_URL")


def _load_vector_database():
    chroma_client = chromadb.PersistentClient(
        path=VECTOR_DATABASE_PATH,
    )
    # collection = chroma_client.get_collection(VECTOR_DATABASE_NAME)
    # Remove below, used for testing
    collection = chroma_client.get_collection("langchain_database")
    return collection


def _load_langchain_chroma_vector_database():
    embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    chroma_db = Chroma(
        persist_directory=VECTOR_DATABASE_PATH,
        embedding_function=embedding_function,
        collection_name=VECTOR_DATABASE_NAME,
    )
    return chroma_db


def _abstract_summarisation(relevant_docs_df: pd.DataFrame) -> pd.DataFrame:
    summariser = pipeline("summarization", model="Falconsai/text_summarization")
    relevant_docs_df["Summary"] = relevant_docs_df["Relevant Documents"].apply(
        summariser
    )
    return relevant_docs_df


def get_langchain_chat_model_response_from_query(query_text: str):
    chroma_db = _load_langchain_chroma_vector_database()
    llm_model = ChatOpenAI(
        api_key=OPENAI_API_KEY,
        base_url=BASE_URL,
    )
    chain = RetrievalQA.from_chain_type(
        llm=llm_model, chain_type="stuff", retriever=chroma_db.as_retriever()
    )
    response = chain(query_text)["result"]
    return response


def retrieve_similar_docs_and_summarisation_from_query(query_text: str):
    collection = _load_vector_database()
    results = collection.query(
        query_texts=[query_text],
        n_results=5,
    )
    similar_docs = results["documents"][0]
    doc_urls = results["ids"][0]
    similar_docs_df = pd.DataFrame(
        {"Relevant Documents": similar_docs, "Link": doc_urls}
    )
    llm_chat_response = get_langchain_chat_model_response_from_query(query_text)
    relevant_docs_summary = summarise_relevant_documents(similar_docs_df)
    return similar_docs_df, llm_chat_response, relevant_docs_summary


def summarise_relevant_documents(similar_docs_df: pd.DataFrame):
    summarisation_results = _abstract_summarisation(similar_docs_df)
    documents_summary = ""
    for _, row in summarisation_results.iterrows():
        documents_summary += row["Relevant Documents"] + "\n"
        documents_summary += row["Link"] + "\n"
        documents_summary += row["Summary"][0]["summary_text"] + "\n" + "\n"

    return documents_summary


def launch_similarity_and_summarisation_service():
    with gr.Blocks(theme=gr.themes.base.Base(), title="Ask us anything") as front_end:
        gr.Markdown(
            """
            # Retreive relevant research papers to your query
            """
        )
        textbox = gr.Textbox(label="Please enter your question:")
        with gr.Row():
            button = gr.Button("Submit", variant="primary")
        with gr.Column():
            output_chat = gr.Textbox(label="Chat response:")
            output_summary = gr.Textbox(label="Summarisation:")
            # output_summary = gr.List(type="pandas", label="Summarisation:")
            output_relevant_docs = gr.List(type="pandas", label="Relevant documents:")

        button.click(
            retrieve_similar_docs_and_summarisation_from_query,
            textbox,
            outputs=[output_relevant_docs, output_chat, output_summary],
        )
    front_end.launch()


if __name__ == "__main__":
    launch_similarity_and_summarisation_service()
