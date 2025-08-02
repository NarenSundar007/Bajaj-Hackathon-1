import os
import aiohttp
import json
import hashlib
import google.generativeai as genai
import numpy as np
from typing import List, Dict, Any
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pydantic_settings import BaseSettings
import uvicorn
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import TokenTextSplitter
from langchain_community.vectorstores.faiss import FAISS
from langchain.schema import Document
import tempfile
import asyncio
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor
import faiss

# Load env vars
load_dotenv()

# Settings
class Settings(BaseSettings):
    gemini_api_key: str = os.getenv("GEMINI_API_KEY", "")
    bearer_token: str = os.getenv("BEARER_TOKEN", "")
    llm_model: str = "gemini-1.5-pro"
    embedding_model: str = "models/embedding-001"
    chunk_size: int = 512
    chunk_overlap: int = 64
    api_host: str = "0.0.0.0"
    api_port: int = int(os.getenv("PORT", 8000))
    debug: bool = os.getenv("ENVIRONMENT", "development") != "production"
    data_dir: str = "./data"

    class Config:
        env_file = ".env"
        extra = "ignore"

settings = Settings()
os.makedirs(settings.data_dir, exist_ok=True)

# Gemini setup
genai.configure(api_key=settings.gemini_api_key)

# Models
class BatchQueryRequest(BaseModel):
    documents: str
    questions: List[str]

class BatchQueryResponse(BaseModel):
    answers: List[str]

_executor = ThreadPoolExecutor(max_workers=10)

class QueryRetrievalSystem:
    def __init__(self):
        self.llm = ChatGoogleGenerativeAI(model=settings.llm_model, google_api_key=settings.gemini_api_key)
        self.prompt_template = PromptTemplate(
            input_variables=["query", "clauses"],
            template="""Based on the provided clauses, answer the query in detail, citing specific parts of the clauses where relevant. Return a JSON object with the following structure:
{{
  "answer": "string",
  "meets_criteria": boolean,
  "applicable_conditions": ["string"],
  "rationale": "string",
  "confidence_score": number,
  "supporting_evidence": ["string"]
}}
Query: {query}

Clauses: {clauses}"""
        )
        self.llm_chain = self.prompt_template | self.llm
        self.vector_store_cache = {}

    async def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        loop = asyncio.get_event_loop()

        def _embed(text: str):
            try:
                result = genai.embed_content(
                    model=settings.embedding_model,
                    content=text,
                    task_type="RETRIEVAL_DOCUMENT"
                )
                return result["embedding"]
            except Exception as e:
                return None

        results = await asyncio.gather(*[loop.run_in_executor(_executor, _embed, text) for text in texts])
        valid = [r for r in results if r is not None]
        return valid

    def cosine_similarity(self, a: List[float], b: List[float]) -> float:
        a, b = np.array(a), np.array(b)
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    async def process_document(self, blob_url: str) -> FAISS:
        if blob_url in self.vector_store_cache:
            return self.vector_store_cache[blob_url]

        cache_id = hashlib.sha256(blob_url.encode()).hexdigest()
        cache_path = os.path.join(settings.data_dir, cache_id)
        if os.path.exists(cache_path):
            vs = FAISS.load_local(cache_path)
            self.vector_store_cache[blob_url] = vs
            return vs

        async with aiohttp.ClientSession() as session:
            async with session.get(blob_url, timeout=30) as response:
                response.raise_for_status()
                pdf_bytes = await response.read()

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf:
            temp_pdf.write(pdf_bytes)
            temp_pdf_path = temp_pdf.name

        loader = PyMuPDFLoader(temp_pdf_path)
        documents = loader.load()
        text_splitter = TokenTextSplitter(chunk_size=settings.chunk_size, chunk_overlap=settings.chunk_overlap)
        chunks = text_splitter.split_documents(documents)
        chunk_texts = [doc.page_content for doc in chunks]

        embeddings = await self.generate_embeddings(chunk_texts)
        if not embeddings or len(embeddings) != len(chunk_texts):
            os.remove(temp_pdf_path)
            raise ValueError("Failed to embed all chunks.")

        # Store embeddings in document metadata
        chunks_with_embeddings = [Document(page_content=text, metadata={"embedding": emb}) for text, emb in zip(chunk_texts, embeddings)]

        # Create FAISS index with pre-computed embeddings
        embeddings_array = np.array(embeddings).astype('float32')
        dimension = embeddings_array.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings_array)

        # Create LangChain FAISS object
        vector_store = FAISS(
            embedding_function=None,  # Not needed since we handle query embedding separately
            index=index,
            docstore={i: chunks_with_embeddings[i] for i in range(len(chunks_with_embeddings))},
            index_to_docstore_id={i: i for i in range(len(chunks_with_embeddings))}
        )

        vector_store.save_local(cache_path)
        self.vector_store_cache[blob_url] = vector_store

        os.remove(temp_pdf_path)
        return vector_store

    def rerank_chunks(self, query: str, docs: List[Document], query_emb: np.ndarray) -> List[Document]:
        chunk_embs = [doc.metadata["embedding"] for doc in docs]
        scores = [self.cosine_similarity(query_emb, np.array(emb)) for emb in chunk_embs]
        ranked = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)
        return [doc for _, doc in ranked[:5]]

    async def process_batch_queries(self, document_url: str, questions: List[str]) -> List[str]:
        vector_store = await self.process_document(document_url)
        index = vector_store.index
        docstore = vector_store.docstore

        async def handle(q: str) -> str:
            # Embed the query asynchronously
            query_emb = await self.generate_embeddings([q])
            if not query_emb:
                return "Failed to embed query"
            query_emb = np.array(query_emb[0]).astype('float32').reshape(1, -1)

            # Search the FAISS index
            D, I = index.search(query_emb, k=10)
            docs = [docstore[i] for i in I[0] if i in docstore]

            # Rerank using pre-computed embeddings
            reranked = self.rerank_chunks(q, docs, query_emb[0])
            text = "\n".join([doc.page_content for doc in reranked])

            try:
                response = await self.llm_chain.ainvoke({"query": q, "clauses": text})
                return self.parse_llm_response(response).get("answer", "Unable to determine")
            except Exception as e:
                return "LLM error"

        return await asyncio.gather(*(handle(q) for q in questions))

    def parse_llm_response(self, response: Any) -> Dict[str, Any]:
        if hasattr(response, "content"):
            response = response.content
        response = str(response).strip()
        if response.startswith("```json"):
            response = response[7:-3].strip()
        elif response.startswith("```"):
            response = response[3:-3].strip()
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            print("[JSON ERROR] Failed to parse JSON from LLM.")
            return {"answer": "Invalid JSON"}

# FastAPI
app = FastAPI(
    title="LLM-Powered Query-Retrieval System",
    description="Query legal/financial docs with LLM",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# Middleware for Bearer token authentication
@app.middleware("http")
async def authenticate_bearer_token(request: Request, call_next):
    if request.method == "POST" and request.url.path == "/hackrx/run":
        auth_header = request.headers.get("Authorization")
        if not auth_header:
            raise HTTPException(status_code=401, detail="Authorization header missing")
        if not auth_header.startswith("Bearer "):
            raise HTTPException(status_code=401, detail="Invalid Authorization header format")
        token = auth_header[len("Bearer "):].strip()
        if token != settings.bearer_token:
            raise HTTPException(status_code=401, detail="Invalid Bearer token")
    response = await call_next(request)
    return response

query_system = QueryRetrievalSystem()

query_system = QueryRetrievalSystem()

@app.post("/hackrx/run", response_model=BatchQueryResponse)
async def process_batch_queries(request: BatchQueryRequest):
    answers = await query_system.process_batch_queries(request.documents, request.questions)
    return BatchQueryResponse(answers=answers)

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    print(f"[ERROR] Unhandled Exception: {exc}")
    return JSONResponse(status_code=500, content={"error": "Internal server error", "detail": str(exc)})

if __name__ == "__main__":
    uvicorn.run("app:app", host=settings.api_host, port=settings.api_port, reload=settings.debug)