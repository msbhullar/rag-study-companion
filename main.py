# main.py
from chain import build_chain

def main():
    print("RAG Study Companion ready. Type 'quit' to exit.\n")
    chain = build_chain()

    while True:
        question = input("Ask a question: ").strip()
        if question.lower() in ["quit", "exit", "q"]:
            break
        if not question:
            continue
        answer = chain.invoke(question)
        print(f"\nAnswer: {answer}\n")

if __name__ == "__main__":
    main()