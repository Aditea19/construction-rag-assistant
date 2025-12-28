def format_sources(docs):
    return list(set(d.metadata.get("source_file","Unknown") for d in docs))
