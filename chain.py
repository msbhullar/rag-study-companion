# chain.py
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from retriever import get_retriever

load_dotenv()

def build_chain():
    llm = ChatOpenAI(
        model_name="gpt-3.5-turbo",
        temperature=0,
        openai_api_key=os.getenv("OPENAI_API_KEY")
    )

    prompt = PromptTemplate.from_template("""You are a helpful study assistant.
Use the following context from the student's study materials to answer the question.
If the answer is not in the context, say "I don't have enough information in your documents to answer that."

Context:
{context}

Question: {question}

Answer:""")

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    chain = (
        {"context": get_retriever() | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain


def ask(question: str):
    chain = build_chain()
    answer = chain.invoke(question)
    print(f"\nQuestion: {question}")
    print(f"\nAnswer: {answer}")


if __name__ == "__main__":
    ask("What tasks are due in module 6?")