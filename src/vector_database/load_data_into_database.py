import chromadb
import pandas as pd

from src.constants.paths import RESEARCH_PAPER_DATA
from src.constants.parameters import VECTOR_DATABASE_NAME


def _get_document_list(df: pd.DataFrame) -> list:
    return [
        row["PAPER_TITLE"] + " " + row["PAPER_ABSTRACT"] for _, row in df.iterrows()
    ]


def _get_document_ids(df: pd.DataFrame) -> list:
    return [row["PAPER_LINK"] for _, row in df.iterrows()]


def _add_document_metadata(df: pd.DataFrame) -> list:
    doc_metadata = [
        {
            "title": row["PAPER_TITLE"],
            "abstract": row["PAPER_ABSTRACT"],
            "url": row["PAPER_LINK"],
        }
        for _, row in df.iterrows()
    ]
    return doc_metadata


def load_data_into_database(df: pd.DataFrame) -> None:
    chroma_client = chromadb.Client()
    collection = chroma_client.create_collection(VECTOR_DATABASE_NAME)
    collection.add(
        documents=_get_document_list(df),
        ids=_get_document_ids(df),
        metadatas=_add_document_metadata(df),
    )


if __name__ == "__main__":
    research_paper_df = pd.read_csv(RESEARCH_PAPER_DATA)
    print("Loading data into database")
    load_data_into_database(research_paper_df)
    print("Loading complete!")
