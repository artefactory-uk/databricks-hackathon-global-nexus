import gradio as gr
import pandas as pd
import chromadb
from src.constants.paths import VECTOR_DATABASE_PATH
from src.constants.parameters import VECTOR_DATABASE_NAME


def _load_vector_database():
    chroma_client = chromadb.PersistentClient(
        path=VECTOR_DATABASE_PATH,
    )
    # collection = chroma_client.get_collection(VECTOR_DATABASE_NAME)
    # Remove below, used for testing
    collection = chroma_client.get_collection("example_database")
    return collection


def retrieve_similar_docs_and_summarisation_from_query(query_text: str):
    collection = _load_vector_database()
    results = collection.query(
        query_texts=[query_text],
        n_results=5,
    )
    similar_docs = results["documents"][0]
    similar_docs_df = pd.DataFrame({"Relevant Documents": similar_docs})
    summary = summarise_relevant_documents(similar_docs_df)
    return similar_docs_df, summary


def summarise_relevant_documents(similar_docs: pd.DataFrame):
    # LLM summarisation code here
    summarised_text = "Summarised text"

    return summarised_text


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
            output_summary = gr.Textbox(label="Summary:")
            # output_summary = gr.List(type="pandas", label="Summarisation:")
            output = gr.List(type="pandas", label="Relevant documents:")

        button.click(
            retrieve_similar_docs_and_summarisation_from_query,
            textbox,
            outputs=[output, output_summary],
        )
    front_end.launch()


if __name__ == "__main__":
    launch_similarity_and_summarisation_service()
