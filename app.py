import os
import aiohttp
import json
import hashlib
from collections import OrderedDict
import google.generativeai as genai
import numpy as np
from typing import List, Dict, Any
from fastapi import FastAPI, HTTPException
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
import logging
import shutil
import docx2txt
from bs4 import BeautifulSoup
from urllib.parse import urlparse

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.WARNING if os.getenv("ENVIRONMENT", "development") == "production" else logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Settings
class Settings(BaseSettings):
    gemini_api_key: str = os.getenv("GEMINI_API_KEY", "")
    llm_model: str = "gemini-1.5-pro"
    embedding_model: str = "models/embedding-001"
    chunk_size: int = 512
    chunk_overlap: int = 64
    api_host: str = "0.0.0.0"
    api_port: int = int(os.getenv("PORT", 8000))
    debug: bool = os.getenv("ENVIRONMENT", "development") != "production"
    data_dir: str = "./data"
    max_cache_size: int = 10  # Max number of in-memory vector stores
    max_disk_size: int = 1024 * 1024 * 1024  # 1 GB disk cache limit
    max_memory_size: int = 100 * 1024 * 1024  # 100 MB memory cache limit

    class Config:
        env_file = ".env"
        extra = "ignore"

settings = Settings()
os.makedirs(settings.data_dir, exist_ok=True)

# Gemini setup
genai.configure(api_key=settings.gemini_api_key)

# Thread pool executor
_executor = ThreadPoolExecutor(max_workers=max(os.cpu_count() or 4, 4))

# Models
class BatchQueryRequest(BaseModel):
    documents: str  # URL to document or email content
    questions: List[str]

class BatchQueryResponse(BaseModel):
    answers: List[str]

# Helper functions
def get_document_type(input_str: str) -> str:
    if input_str.startswith("http"):
        path = urlparse(input_str).path
        ext = os.path.splitext(path)[1].lower()
        if ext == ".pdf":
            return "pdf"
        elif ext == ".docx":
            return "docx"
        else:
            return "unknown"
    return "email"

def clean_text(text: str) -> str:
    return " ".join(text.split()).strip()

# Document processing functions
async def process_pdf(pdf_path: str) -> List[Dict[str, Any]]:
    chunks = []
    try:
        loader = PyMuPDFLoader(pdf_path)
        documents = loader.load()
        text_splitter = TokenTextSplitter(chunk_size=settings.chunk_size, chunk_overlap=settings.chunk_overlap)
        split_docs = text_splitter.split_documents(documents)
        chunks.extend({"content": doc.page_content, "metadata": {"type": "text", "page": doc.metadata.get("page", 1)}} for doc in split_docs)
    except Exception as e:
        logging.error(f"Error processing PDF {pdf_path}: {e}")
    return chunks

async def process_docx(docx_path: str) -> List[Dict[str, Any]]:
    chunks = []
    try:
        text = docx2txt.process(docx_path)
        if text:
            text_splitter = TokenTextSplitter(chunk_size=settings.chunk_size, chunk_overlap=settings.chunk_overlap)
            split_texts = text_splitter.split_text(clean_text(text))
            chunks.extend({"content": text, "metadata": {"type": "text"}} for text in split_texts)
    except Exception as e:
        logging.error(f"Error processing DOCX {docx_path}: {e}")
    return chunks

async def process_email(email_content: str) -> List[Dict[str, Any]]:
    try:
        if "<html" in email_content.lower():
            text = BeautifulSoup(email_content, "html.parser").get_text()
        else:
            text = email_content
        text_splitter = TokenTextSplitter(chunk_size=settings.chunk_size, chunk_overlap=settings.chunk_overlap)
        split_texts = text_splitter.split_text(clean_text(text))
        return [{"content": chunk, "metadata": {"type": "text", "source": "email"}} for chunk in split_texts]
    except Exception as e:
        logging.error(f"Error processing email content: {e}")
        return []

class QueryRetrievalSystem:
    def __init__(self):
        logging.info("[INIT] Initializing QueryRetrievalSystem...")
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
        self.vector_store_cache = OrderedDict()  # LRU cache for vector stores
        self.cache_access_times = {}  # Track last access time for LRU
        self.document_hashes = {}  # Track document content hashes
        self.memory_usage = 0  # Track in-memory vector store size

    async def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        logging.info(f"[EMBED] Generating embeddings for {len(texts)} texts...")
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
                logging.error(f"[EMBED ERROR] Failed to embed: {text[:60]}... → {e}")
                return None

        results = await asyncio.gather(*[loop.run_in_executor(_executor, _embed, text) for text in texts])
        valid = [r for r in results if r is not None]
        logging.info(f"[EMBED] Completed. Success: {len(valid)} / {len(texts)}")
        return valid

    def cosine_similarity(self, a: List[float], b: List[float]) -> float:
        a, b = np.array(a), np.array(b)
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    def get_directory_size(self, directory: str) -> int:
        total_size = 0
        for dirpath, _, filenames in os.walk(directory):
            for f in filenames:
                fp = os.path.join(dirpath, f)
                total_size += os.path.getsize(fp)
        return total_size

    def estimate_vector_store_size(self, vector_store: FAISS) -> int:
        # Rough estimate: sum of index size (in memory) and docstore content
        index_size = vector_store.index.ntotal * vector_store.index.d * 4  # 4 bytes per float
        docstore_size = sum(len(doc.page_content.encode()) for doc in vector_store.docstore.values())
        return index_size + docstore_size

    def manage_cache(self, url: str, vector_store: FAISS, cache_path: str):
        # Estimate size of new vector store
        estimated_size = self.estimate_vector_store_size(vector_store)
        
        # Check memory cache
        while self.memory_usage + estimated_size > settings.max_memory_size and self.vector_store_cache:
            old_url, _ = self.vector_store_cache.popitem(last=False)
            old_size = self.estimate_vector_store_size(self.vector_store_cache.get(old_url, vector_store))
            self.memory_usage -= old_size
            logging.info(f"[CACHE] Evicted in-memory vector store for {old_url}")
            if old_url in self.cache_access_times:
                del self.cache_access_times[old_url]
            if old_url in self.document_hashes:
                del self.document_hashes[old_url]

        # Check disk cache
        disk_size = self.get_directory_size(settings.data_dir)
        while disk_size + estimated_size > settings.max_disk_size and self.vector_store_cache:
            old_url = next(iter(self.vector_store_cache))
            cache_id = hashlib.sha256(old_url.encode()).hexdigest()
            old_cache_path = os.path.join(settings.data_dir, cache_id)
            if os.path.exists(old_cache_path):
                try:
                    shutil.rmtree(old_cache_path)
                    logging.info(f"[CACHE] Deleted disk cache for {old_url} at {old_cache_path}")
                except Exception as e:
                    logging.error(f"[CACHE] Failed to delete disk cache {old_cache_path}: {e}")
            del self.vector_store_cache[old_url]
            self.memory_usage -= self.estimate_vector_store_size(self.vector_store_cache.get(old_url, vector_store))
            if old_url in self.cache_access_times:
                del self.cache_access_times[old_url]
            if old_url in self.document_hashes:
                del self.document_hashes[old_url]

        # Add new vector store to cache
        self.vector_store_cache[url] = vector_store
        self.memory_usage += estimated_size
        self.cache_access_times[url] = asyncio.get_event_loop().time()
        vector_store.save_local(cache_path)
        logging.info(f"[CACHE] Cached vector store for {url} at {cache_path}")

    async def process_document(self, input_str: str) -> FAISS:
        logging.info(f"[DOC] Processing document: {input_str}")
        cache_id = hashlib.sha256(input_str.encode()).hexdigest()
        cache_path = os.path.join(settings.data_dir, cache_id)

        # Check if document is cached and content hasn't changed
        content_hash = None
        if input_str.startswith("http"):
            async with aiohttp.ClientSession() as session:
                async with session.head(input_str, timeout=30) as response:
                    content = await response.read()
                    content_hash = hashlib.sha256(content).hexdigest()
            if input_str in self.vector_store_cache and self.document_hashes.get(input_str) == content_hash:
                logging.info("[DOC] Cache hit in memory.")
                self.cache_access_times[input_str] = asyncio.get_event_loop().time()
                return self.vector_store_cache[input_str]

        # Check disk cache
        if os.path.exists(cache_path):
            try:
                vs = FAISS.load_local(cache_path, embeddings=None, allow_dangerous_deserialization=True)
                self.manage_cache(input_str, vs, cache_path)
                if content_hash:
                    self.document_hashes[input_str] = content_hash
                logging.info(f"[DOC] Loaded FAISS index from disk cache: {cache_path}")
                return vs
            except Exception as e:
                logging.error(f"[DOC] Failed to load FAISS index from {cache_path}: {e}")

        # Process document based on type
        doc_type = get_document_type(input_str)
        if doc_type in ["pdf", "docx"]:
            async with aiohttp.ClientSession() as session:
                async with session.get(input_str, timeout=30) as response:
                    logging.info(f"[DOC] Downloading {doc_type.upper()}...")
                    response.raise_for_status()
                    content = await response.read()
                    content_hash = hashlib.sha256(content).hexdigest()
            suffix = ".pdf" if doc_type == "pdf" else ".docx"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
                temp_file.write(content)
                temp_file_path = temp_file.name
            try:
                chunks = await process_pdf(temp_file_path) if doc_type == "pdf" else await process_docx(temp_file_path)
            finally:
                os.remove(temp_file_path)
        elif doc_type == "email":
            content_hash = hashlib.sha256(input_str.encode()).hexdigest()
            chunks = await process_email(input_str)
        else:
            raise ValueError(f"Unsupported document type: {doc_type}")

        if not chunks:
            raise ValueError(f"No content extracted from: {input_str}")

        chunk_texts = [chunk["content"] for chunk in chunks]
        embeddings = await self.generate_embeddings(chunk_texts)
        if len(embeddings) != len(chunk_texts):
            raise ValueError("Failed to embed all chunks")

        # Store embeddings in document metadata
        chunks_with_embeddings = [Document(page_content=c["content"], metadata={**c["metadata"], "embedding": e})
                                 for c, e in zip(chunks, embeddings)]

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

        # Manage cache
        self.document_hashes[input_str] = content_hash
        self.manage_cache(input_str, vector_store, cache_path)

        return vector_store

    def rerank_chunks(self, query: str, docs: List[Document], query_emb: np.ndarray) -> List[Document]:
        logging.info(f"[RERANK] Reranking chunks for query: {query}")
        chunk_embs = [doc.metadata["embedding"] for doc in docs]
        scores = [self.cosine_similarity(query_emb, np.array(emb)) for emb in chunk_embs]
        ranked = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)
        logging.info(f"[RERANK] Top 5 scores: {[round(s, 3) for s, _ in ranked[:5]]}")
        return [doc for _, doc in ranked[:5]]

    async def process_batch_queries(self, document_url: str, questions: List[str]) -> List[str]:
        logging.info(f"[PROCESS] Running batch for {len(questions)} questions...")
        vector_store = await self.process_document(document_url)
        index = vector_store.index
        docstore = vector_store.docstore

        async def handle(q: str) -> str:
            logging.info(f"[QUERY] Processing: {q}")
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
                logging.info(f"[LLM] Response: {response}")
                return self.parse_llm_response(response).get("answer", "Unable to determine")
            except Exception as e:
                logging.error(f"[LLM ERROR] Failed LLM call: {e}")
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
            logging.error("[JSON ERROR] Failed to parse JSON from LLM.")
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

query_system = QueryRetrievalSystem()

@app.post("/hackrx/run", response_model=BatchQueryResponse)
async def process_batch_queries(request: BatchQueryRequest):
    logging.info(f"[API] Received document: {request.documents}")
    logging.info(f"[API] Questions: {request.questions}")
    answers = await query_system.process_batch_queries(request.documents, request.questions)
    logging.info(f"[API] Final answers: {answers}")
    return BatchQueryResponse(answers=answers)

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logging.error(f"[ERROR] Unhandled Exception: {exc}")
    return JSONResponse(status_code=500, content={"error": "Internal server error", "detail": str(exc)})

if __name__ == "__main__":
    uvicorn.run("app:app", host=settings.api_host, port=settings.api_port, reload=settings.debug)