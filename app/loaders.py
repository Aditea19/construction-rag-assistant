from langchain_community.document_loaders import TextLoader

def load_documents():
    docs = []
    for name in ["doc1.md", "doc2.md", "doc3.md"]:
        loader = TextLoader(f"data/{name}", encoding="utf-8")
        loaded = loader.load()
        for d in loaded:
            d.metadata["source_file"] = name
        docs.extend(loaded)
    return docs
