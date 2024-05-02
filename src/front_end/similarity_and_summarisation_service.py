import gradio as gr
import pandas as pd
from typing import Any
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
from src.constants.parameters import (
    VECTOR_DATABASE_NAME,
    EMBEDDING_MODEL,
    SUMMARISATION_MODEL,
    NUMBER_OF_SIMILAR_DOCUMENTS_TO_RETRIEVE,
)


def _load_langchain_chroma_vector_database() -> Chroma:
    embedding_function = SentenceTransformerEmbeddings(model_name=EMBEDDING_MODEL)
    chroma_db = Chroma(
        persist_directory=VECTOR_DATABASE_PATH,
        embedding_function=embedding_function,
        collection_name=VECTOR_DATABASE_NAME,
    )
    return chroma_db


def _abstract_summarisation(relevant_docs_df: pd.DataFrame) -> pd.DataFrame:
    summariser = pipeline("summarization", model=SUMMARISATION_MODEL)
    relevant_docs_df["Summary"] = relevant_docs_df["Abstract"].apply(summariser)
    return relevant_docs_df


def _extract_metadata(results: list[Any], metadata_type: str) -> list[Any]:
    length_results_list = len(results)
    doc_property = [
        results[index].metadata[metadata_type] for index in range(length_results_list)
    ]
    return doc_property


def _create_similar_results_dataframe(results: list[Any]) -> pd.DataFrame:
    doc_titles = _extract_metadata(results, "title")
    doc_urls = _extract_metadata(results, "url")
    doc_abstracts = _extract_metadata(results, "abstract")
    similar_docs_df = pd.DataFrame(
        {
            "Document Title": doc_titles,
            "Link": doc_urls,
            "Abstract": doc_abstracts,
        }
    )
    return similar_docs_df


def _summarise_relevant_documents(similar_docs_df: pd.DataFrame) -> str:
    summarisation_results = _abstract_summarisation(similar_docs_df)
    documents_summary = "\n".join(
        f'{row["Document Title"]}\n{row["Link"]}\n{row["Summary"][0]["summary_text"]}\n'
        for _, row in summarisation_results.iterrows()
    )
    return documents_summary


def get_langchain_chat_model_response_from_query(
    query_text: str, chroma_vector_database: Chroma
) -> str:
    llm_model = ChatOpenAI(
        api_key=OPENAI_API_KEY,
        base_url=BASE_URL,
    )
    chain = RetrievalQA.from_chain_type(
        llm=llm_model,
        chain_type="stuff",
        retriever=chroma_vector_database.as_retriever(),
    )
    response = chain(query_text)["result"]
    return response


def retrieve_similar_docs_and_summarisation_from_query(
    query_text: str,
) -> tuple[pd.DataFrame, str, str]:
    chroma_db = _load_langchain_chroma_vector_database()
    results = chroma_db.similarity_search(
        query_text, k=NUMBER_OF_SIMILAR_DOCUMENTS_TO_RETRIEVE
    )
    llm_chat_response = get_langchain_chat_model_response_from_query(
        query_text, chroma_db
    )
    similar_docs_df = _create_similar_results_dataframe(results)
    relevant_docs_summary = _summarise_relevant_documents(similar_docs_df)
    similar_docs_snippet = similar_docs_df[["Document Title", "Link"]]
    return similar_docs_snippet, llm_chat_response, relevant_docs_summary


def launch_similarity_and_summarisation_service():
    with gr.Blocks(
        theme=gr.themes.base.Base(), title="Ask us about Medical Research Papers!"
    ) as front_end:
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
            output_relevant_docs = gr.List(type="pandas", label="Relevant documents:")

        button.click(
            retrieve_similar_docs_and_summarisation_from_query,
            textbox,
            outputs=[output_relevant_docs, output_chat, output_summary],
        )
    front_end.launch()


if __name__ == "__main__":
    load_dotenv()
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    BASE_URL = os.getenv("BASE_URL")
    launch_similarity_and_summarisation_service()
