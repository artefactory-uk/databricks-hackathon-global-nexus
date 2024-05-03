from dotenv import load_dotenv
import os
from src.front_end.layout.header import header
from src.front_end.callbacks.callbacks import (
    retrieve_similar_docs_and_summarisation_from_query,
)


def callback_function(query_text: str) -> tuple:
    return retrieve_similar_docs_and_summarisation_from_query(
        query_text, OPENAI_API_KEY, BASE_URL
    )


def launch_similarity_and_summarisation_service():
    front_end = header(callback_function=callback_function)
    front_end.launch(allowed_paths=["."])


if __name__ == "__main__":
    load_dotenv()
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    BASE_URL = os.getenv("BASE_URL")
    launch_similarity_and_summarisation_service()
