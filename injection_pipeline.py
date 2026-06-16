import os
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv
from functools import partial

load_dotenv()


##file loader
def load_file(docs_path="./docs"):
    print(f"loading files from {docs_path}")

    if not os.path.exists(docs_path):
        raise FileNotFoundError(f"Directory {docs_path} does not exist")

    loader = DirectoryLoader(
        path=docs_path, glob="*.txt", loader_cls=partial(TextLoader, encoding="utf-8")
    )
    documents = loader.load()

    if len(documents) == 0:
        raise FileNotFoundError(f"No files found in {docs_path}")

    for i, doc in enumerate(documents[:2]):  # Show first 2 documents
        print(f"\nDocument {i + 1}:")
        print(f"  Source: {doc.metadata['source']}")
        print(f"  Content length: {len(doc.page_content)} characters")
        print(f"  Content preview: {doc.page_content[:100]}...")
        print(f"  metadata: {doc.metadata}")

    return documents


def main():
    documents = load_file()
    print(f"\nLoaded {len(documents)} documents")


if __name__ == "__main__":
    main()
