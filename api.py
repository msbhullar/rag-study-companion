# api.py
# FastAPI backend that exposes the RAG chain as an API
# with streaming support for the frontend.

import os
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from advanced_retriever import advanced_retrieve
from ingest import ingest_pdf
import shutil

load_dotenv()

app = FastAPI()

# Allow Next.js frontend to talk to this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QuestionRequest(BaseModel):
    question: str

prompt = PromptTemplate.from_template("""You are a helpful study assistant.
Use the following context from the student's study materials to answer the question.
If the answer is not in the context, say "I don't have enough information in your documents to answer that."

Context:
{context}

Question: {question}

Answer:""")

def get_llm():
    return ChatOpenAI(
        model_name="gpt-3.5-turbo",
        temperature=0,
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        streaming=True
    )

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/ask")
async def ask(request: QuestionRequest):
    docs = advanced_retrieve(request.question)
    context = "\n\n".join(doc.page_content for doc in docs)

    llm = ChatOpenAI(
        model_name="gpt-3.5-turbo",
        temperature=0,
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        streaming=False
    )

    chain = prompt | llm | StrOutputParser()
    answer = chain.invoke({
        "question": request.question,
        "context": context
    })

    return {"answer": answer}

@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    temp_path = f"/tmp/{file.filename}"
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    ingest_pdf(temp_path)
    return {"message": f"{file.filename} ingested successfully"}