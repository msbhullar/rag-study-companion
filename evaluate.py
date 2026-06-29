# evaluate.py
# Custom RAG evaluation pipeline measuring:
# - Faithfulness: is the answer grounded in the retrieved context?
# - Answer Relevancy: does the answer address the question?
# - Context Precision: are retrieved chunks relevant?

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from advanced_retriever import advanced_retrieve
from chain import build_chain

load_dotenv()

EVAL_QUESTIONS = [
    "What tasks are due in module 6?",
    "What is an activation function?",
    "What is the deadline for module 6 submissions?",
    "What is ReLU?",
    "What topics are covered in module 6?",
]

evaluator_llm = ChatOpenAI(
    model_name="gpt-3.5-turbo",
    temperature=0,
    openai_api_key=os.getenv("OPENAI_API_KEY")
)

def score_faithfulness(question, answer, context):
    prompt = PromptTemplate.from_template("""
You are an evaluator. Given a question, an answer, and context from a document,
score how faithful the answer is to the context on a scale of 0.0 to 1.0.
1.0 = answer is fully grounded in the context.
0.0 = answer contains information not in the context.
Return ONLY a number between 0.0 and 1.0, nothing else.

Question: {question}
Context: {context}
Answer: {answer}
Score:""")
    chain = prompt | evaluator_llm | StrOutputParser()
    result = chain.invoke({"question": question, "answer": answer, "context": context[:1000]})
    try:
        return float(result.strip())
    except:
        return 0.0

def score_relevancy(question, answer):
    prompt = PromptTemplate.from_template("""
You are an evaluator. Score how relevant this answer is to the question on a scale of 0.0 to 1.0.
1.0 = answer directly addresses the question.
0.0 = answer is completely off-topic.
Return ONLY a number between 0.0 and 1.0, nothing else.

Question: {question}
Answer: {answer}
Score:""")
    chain = prompt | evaluator_llm | StrOutputParser()
    result = chain.invoke({"question": question, "answer": answer})
    try:
        return float(result.strip())
    except:
        return 0.0

def score_context_precision(question, context):
    prompt = PromptTemplate.from_template("""
You are an evaluator. Score how relevant the retrieved context is to the question on a scale of 0.0 to 1.0.
1.0 = context is highly relevant to the question.
0.0 = context is completely unrelated to the question.
Return ONLY a number between 0.0 and 1.0, nothing else.

Question: {question}
Context: {context}
Score:""")
    chain = prompt | evaluator_llm | StrOutputParser()
    result = chain.invoke({"question": question, "context": context[:1000]})
    try:
        return float(result.strip())
    except:
        return 0.0

def run_evaluation():
    print("Building RAG chain...")
    chain = build_chain()

    faithfulness_scores = []
    relevancy_scores = []
    precision_scores = []

    print(f"\nEvaluating {len(EVAL_QUESTIONS)} questions...\n")
    print("-" * 60)

    for question in EVAL_QUESTIONS:
        print(f"Q: {question}")

        answer = chain.invoke(question)
        docs = advanced_retrieve(question)
        context = "\n".join(doc.page_content for doc in docs)

        f_score = score_faithfulness(question, answer, context)
        r_score = score_relevancy(question, answer)
        p_score = score_context_precision(question, context)

        faithfulness_scores.append(f_score)
        relevancy_scores.append(r_score)
        precision_scores.append(p_score)

        print(f"A: {answer[:120]}...")
        print(f"   Faithfulness: {f_score:.2f} | Relevancy: {r_score:.2f} | Precision: {p_score:.2f}")
        print()

    avg_f = sum(faithfulness_scores) / len(faithfulness_scores)
    avg_r = sum(relevancy_scores) / len(relevancy_scores)
    avg_p = sum(precision_scores) / len(precision_scores)

    print("=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"Faithfulness:      {avg_f:.3f}")
    print(f"Answer Relevancy:  {avg_r:.3f}")
    print(f"Context Precision: {avg_p:.3f}")
    print(f"Overall Score:     {(avg_f + avg_r + avg_p) / 3:.3f}")
    print("=" * 60)
    print("Score guide: 0 = worst, 1 = best")

if __name__ == "__main__":
    run_evaluation()