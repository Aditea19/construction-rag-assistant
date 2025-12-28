from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_models import ChatOllama

def make_qa_chain(retriever):
    llm = ChatOllama(model="phi3")

    prompt = ChatPromptTemplate.from_template("""
You are a construction assistant.

RULES:
- Answer ONLY from the context below.
- If missing, say: Not available in the documents.

Context:
{context}

Question: {question}
""")

    def chain(q):
        docs = retriever.get_relevant_documents(q)
        context = "\n\n".join([d.page_content for d in docs])
        answer = (prompt | llm | StrOutputParser()).invoke({
            "context": context,
            "question": q
        })
        return {"answer": answer, "context": docs}

    return chain
