# memory_chain.py
# Manual conversation memory implementation with RAG chain.
# Stores last 3 exchanges and passes them as context to the LLM.

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from advanced_retriever import advanced_retrieve

load_dotenv()

# Manual memory — stores last k exchanges as a list of dicts
class ConversationBufferWindowMemory:
    def __init__(self, k=3):
        self.k = k
        self.history = []

    def save(self, question: str, answer: str):
        self.history.append({"human": question, "ai": answer})
        # Keep only last k exchanges
        if len(self.history) > self.k:
            self.history = self.history[-self.k:]

    def load(self) -> str:
        if not self.history:
            return "No previous conversation."
        lines = []
        for exchange in self.history:
            lines.append(f"Human: {exchange['human']}")
            lines.append(f"AI: {exchange['ai']}")
        return "\n".join(lines)


prompt = PromptTemplate.from_template("""You are a helpful study assistant with memory of the conversation.
Use the context from the student's documents AND the chat history to answer.
If the answer is not in the context, say "I don't have enough information in your documents to answer that."

Chat History:
{chat_history}

Context from documents:
{context}

Question: {question}

Answer:""")


def build_memory_chain():
    llm = ChatOpenAI(
        model_name="gpt-3.5-turbo",
        temperature=0,
        openai_api_key=os.getenv("OPENAI_API_KEY")
    )
    memory = ConversationBufferWindowMemory(k=3)
    chain = prompt | llm | StrOutputParser()

    def run(question: str) -> str:
        docs = advanced_retrieve(question)
        context = "\n\n".join(doc.page_content for doc in docs)
        chat_history = memory.load()

        answer = chain.invoke({
            "question": question,
            "context": context,
            "chat_history": chat_history
        })

        memory.save(question, answer)
        return answer

    return run


def main():
    print("RAG Study Companion with Memory. Type 'quit' to exit.\n")
    chain = build_memory_chain()

    while True:
        question = input("Ask a question: ").strip()
        if question.lower() in ["quit", "exit", "q"]:
            break
        if not question:
            continue
        answer = chain(question)
        print(f"\nAnswer: {answer}\n")


if __name__ == "__main__":
    main()