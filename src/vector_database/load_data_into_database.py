import chromadb
import pandas as pd
from src.constants.paths import RESEARCH_PAPER_DATA
from src.constants.parameters import VECTOR_DATABASE_NAME


def _get_document_list(df):
    return [
        row["PAPER_TITLE"] + " " + row["PAPER_ABSTRACT"] for idx, row in df.iterrows()
    ]


def _get_document_ids(df):
    return [row["PAPER_LINK"] for idx, row in df.iterrows()]


def load_data_into_database(df):
    chroma_client = chromadb.Client()
    collection = chroma_client.create_collection(VECTOR_DATABASE_NAME)
    collection.add(
        documents=_get_document_list(df),
        ids=_get_document_ids(df),
    )


if __name__ == "__main__":
    research_paper_df = pd.read_csv(RESEARCH_PAPER_DATA)
    load_data_into_database(research_paper_df)
