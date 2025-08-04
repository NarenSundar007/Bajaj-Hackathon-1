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
from langchain_text_splitters import TokenTextSplitter, RecursiveCharacterTextSplitter
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
    chunk_size: int = 600  # Larger chunks = fewer embeddings
    chunk_overlap: int = 60  # Reduced overlap
    max_retrieval_chunks: int = 12  # More chunks for better context
    rerank_top_k: int = 6  # More final chunks for comprehensive answers
    max_concurrent_embeddings: int = 12  # Increased concurrency
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
async def process_pdf(pdf_path: str) -> List[Document]:
    try:
        logging.info(f"[PDF] Loading PDF from: {pdf_path}")
        loader = PyMuPDFLoader(pdf_path)
        documents = loader.load()
        logging.info(f"[PDF] Loaded {len(documents)} pages")
        
        # Clause-aware text splitting for better accuracy
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size,  # Larger chunks (600)
            chunk_overlap=settings.chunk_overlap,  # Reduced overlap (60)
            separators=[
                "\n\n",  # Paragraph breaks (priority)
                "\n",    # Line breaks
                ". ",    # Sentence boundaries
                " "      # Word boundaries only
            ]
        )
        
        # Process each page and extract clause information
        all_chunks = []
        for page_num, doc in enumerate(documents):
            # Skip if page is too short or likely header/footer page
            if len(doc.page_content.strip()) < 100:
                continue
                
            # Split page content into chunks
            page_chunks = text_splitter.split_text(doc.page_content)
            
            for chunk_idx, chunk_text in enumerate(page_chunks):
                # More aggressive filtering for meaningful content
                if len(chunk_text.strip()) < 80:  # Increased minimum length
                    continue
                    
                if is_header_footer(chunk_text):
                    continue
                
                # Skip chunks that are mostly numbers/formatting
                words = chunk_text.split()
                if len([w for w in words if w.isalpha() and len(w) > 2]) < 10:
                    continue
                
                # Extract clause numbers using regex patterns
                clause_patterns = [
                    r'\b\d+\.\d+(?:\.\w+)?\b',  # 4.2.c, 1.1, 5.15.v
                    r'\bClause\s+\d+(?:\.\d+)?(?:\.\w+)?\b',  # Clause 4.2.c
                    r'\bSection\s+\d+(?:\.\d+)?(?:\.\w+)?\b',  # Section 5.15.v
                ]
                
                extracted_clauses = []
                for pattern in clause_patterns:
                    import re
                    matches = re.findall(pattern, chunk_text, re.IGNORECASE)
                    extracted_clauses.extend(matches)
                
                # Create enhanced metadata
                metadata = {
                    **doc.metadata,
                    "page": page_num + 1,
                    "chunk_index": chunk_idx,
                    "clauses": list(set(extracted_clauses)),  # Remove duplicates
                    "chunk_type": classify_chunk_type(chunk_text),
                    "word_count": len(chunk_text.split())
                }
                
                all_chunks.append(Document(
                    page_content=chunk_text.strip(),
                    metadata=metadata
                ))
        
        logging.info(f"[PDF] Created {len(all_chunks)} clause-aware chunks")
        return all_chunks
        
    except Exception as e:
        logging.error(f"Error processing PDF {pdf_path}: {e}")
        return []

def classify_chunk_type(text: str) -> str:
    """Classify chunk content type for better retrieval"""
    text_lower = text.lower()
    
    if any(word in text_lower for word in ['definition', 'means', 'shall mean', 'refers to']):
        return "definition"
    elif any(word in text_lower for word in ['exclusion', 'excluded', 'not covered', 'does not cover']):
        return "exclusion"
    elif any(word in text_lower for word in ['coverage', 'covered', 'benefit', 'shall pay']):
        return "coverage"
    elif any(word in text_lower for word in ['premium', 'payment', 'due', 'payable']):
        return "payment"
    elif any(word in text_lower for word in ['waiting period', 'shall commence', 'effective date']):
        return "waiting_period"
    elif any(word in text_lower for word in ['claim', 'notification', 'settlement']):
        return "claims"
    else:
        return "general"

def is_header_footer(text: str) -> bool:
    """Enhanced filtering of headers, footers, and non-content text"""
    text_clean = text.strip().lower()
    
    # Very short text
    if len(text_clean) < 30:
        return True
    
    # Common header/footer patterns
    header_footer_patterns = [
        'page ', 'confidential', 'proprietary', 'copyright',
        'policy number', 'effective date:', 'renewal date:',
        'table of contents', 'index', 'page no', 'page:',
        'national insurance company', 'company limited',
        'continued...', '...continued', 'end of page',
        'www.', 'http', 'email:', 'tel:', 'fax:',
        'all rights reserved', 'version', 'print date'
    ]
    
    if any(pattern in text_clean for pattern in header_footer_patterns):
        return True
    
    # Just page numbers or short references
    if text_clean.isdigit() or len(text_clean.split()) < 5:
        return True
    
    # Mostly punctuation or special characters
    alpha_chars = sum(1 for c in text_clean if c.isalpha())
    if alpha_chars < len(text_clean) * 0.6:  # Less than 60% alphabetic
        return True
        
    return False

async def process_docx(docx_path: str) -> List[Document]:
    try:
        text = docx2txt.process(docx_path)
        if text:
            cleaned_text = clean_text(text)
            text_splitter = TokenTextSplitter(chunk_size=settings.chunk_size, chunk_overlap=settings.chunk_overlap)
            split_texts = text_splitter.split_text(cleaned_text)
            return [Document(page_content=chunk, metadata={"type": "text"}) for chunk in split_texts]
        return []
    except Exception as e:
        logging.error(f"Error processing DOCX {docx_path}: {e}")
        return []

async def process_email(email_content: str) -> List[Document]:
    try:
        if "<html" in email_content.lower():
            text = BeautifulSoup(email_content, "html.parser").get_text()
        else:
            text = email_content
        cleaned_text = clean_text(text)
        text_splitter = TokenTextSplitter(chunk_size=settings.chunk_size, chunk_overlap=settings.chunk_overlap)
        split_texts = text_splitter.split_text(cleaned_text)
        return [Document(page_content=chunk, metadata={"type": "text", "source": "email"}) for chunk in split_texts]
    except Exception as e:
        logging.error(f"Error processing email content: {e}")
        return []

class QueryRetrievalSystem:
    def __init__(self):
        logging.info("[INIT] Initializing QueryRetrievalSystem...")
        self.llm = ChatGoogleGenerativeAI(model=settings.llm_model, google_api_key=settings.gemini_api_key)
        self.prompt_template = PromptTemplate(
            input_variables=["query", "clauses"],
            template="""You are a precision insurance policy analyst. Your task is to provide factual, clause-grounded answers based ONLY on the provided policy excerpts.

## STRICT GUIDELINES:
1. **Answer ONLY using the provided clauses** - Never add external knowledge
2. **Include clause references** when explicitly mentioned in the text (e.g., "Clause 4.2.c", "Section 5.15")
3. **Use exact wording** from the policy for critical details (amounts, time periods, conditions)
4. **If information is missing**, clearly state: "Not mentioned in the provided clauses"
5. **For partial information**, explain what IS covered and note limitations
6. **Maintain factual precision** - prefer exact quotes over paraphrasing

## RESPONSE FORMAT:
- Start with direct answer (Yes/No/Partially if applicable)
- Provide specific details with exact values (amounts, periods, percentages)
- Include conditions, limitations, or exclusions if mentioned
- End with clause reference if available in the text
- Use confident language ONLY when explicitly supported by the clauses

## EXAMPLES:
✅ GOOD: "Yes, maternity expenses are covered including normal delivery and caesarean section. Coverage requires 24 months of continuous enrollment. Benefit limited to ₹50,000 per pregnancy. (As per Clause 3.1.14)"

❌ BAD: "Yes, maternity is typically covered under most health insurance policies with waiting periods."

## IMPORTANT:
- Quote exact amounts, percentages, and time periods
- Distinguish between "covered" vs "may be covered" vs "excluded"
- If multiple clauses apply, explain how they interact
- Never assume coverage details not explicitly stated

Return your response as a JSON object:
{{
  "answer": "Precise, clause-grounded answer with exact details and references",
  "confidence_score": 0.95,
  "clause_references": ["4.2.c", "5.15.v"],
  "coverage_status": "covered|excluded|conditional|not_mentioned"
}}

Query: {query}

Retrieved Policy Clauses:
{clauses}

Analysis and Answer:"""
        )
        self.llm_chain = self.prompt_template | self.llm
        self.vector_store_cache = OrderedDict()  # LRU cache for vector stores
        self.cache_access_times = {}  # Track last access time for LRU
        self.document_hashes = {}  # Track document content hashes
        self.memory_usage = 0  # Track in-memory vector store size

    def flush_vector_cache(self):
        """Flush all cached vector stores to ensure clean state for new documents"""
        logging.info("[CACHE] Flushing all vector stores from memory")
        self.vector_store_cache.clear()
        self.cache_access_times.clear()
        self.document_hashes.clear()
        self.memory_usage = 0
        
    def clear_disk_cache(self):
        """Clear all disk cache to prevent data contamination"""
        try:
            if os.path.exists(settings.data_dir):
                shutil.rmtree(settings.data_dir)
                os.makedirs(settings.data_dir, exist_ok=True)
                logging.info("[CACHE] Cleared all disk cache")
        except Exception as e:
            logging.error(f"[CACHE] Failed to clear disk cache: {e}")

    async def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Ultra-fast embedding generation with optimized batching and concurrency"""
        if not texts:
            return []
            
        import time
        start_time = time.time()
        
        # Optimized concurrency settings
        max_concurrent = settings.max_concurrent_embeddings  # 12 concurrent requests
        batch_size = 1  # Keep single requests for Gemini API
        
        async def _embed_single(text: str):
            try:
                # Optimize text processing - keep substantial content
                cleaned_text = text.strip()[:8000]  # Increased limit for larger chunks
                if not cleaned_text or len(cleaned_text) < 10:
                    return None
                    
                result = genai.embed_content(
                    model=settings.embedding_model,
                    content=cleaned_text,
                    task_type="RETRIEVAL_DOCUMENT"
                )
                return result["embedding"]
            except Exception as e:
                logging.warning(f"[EMBED] Failed: {str(e)[:30]}...")
                return None

        # High-speed concurrent processing
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def _controlled_embed(text: str):
            async with semaphore:
                return await _embed_single(text)
        
        # Process all texts concurrently - no timeouts for accuracy
        embedding_tasks = [_controlled_embed(text) for text in texts]
        all_embeddings = await asyncio.gather(*embedding_tasks, return_exceptions=True)
        
        # Filter valid embeddings efficiently
        valid_embeddings = [
            emb for emb in all_embeddings 
            if emb is not None and not isinstance(emb, Exception)
        ]
        
        end_time = time.time()
        total_time = end_time - start_time
        rate = len(valid_embeddings) / total_time if total_time > 0 else 0
        
        logging.info(f"[EMBED] Generated {len(valid_embeddings)}/{len(texts)} embeddings in {total_time:.2f}s ({rate:.1f} emb/s)")
        
        return valid_embeddings

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
            try:
                timeout = aiohttp.ClientTimeout(total=10, connect=5)  # Quick check for caching
                async with aiohttp.ClientSession(timeout=timeout) as session:
                    async with session.head(input_str) as response:  # Use HEAD request for quick check
                        logging.info(f"[DOC] Checking document headers: {response.status}")
                        if response.status == 200:
                            # Use ETag or Last-Modified if available, otherwise download for hash
                            etag = response.headers.get('ETag')
                            last_modified = response.headers.get('Last-Modified')
                            if etag:
                                content_hash = etag.strip('"')
                                logging.info(f"[DOC] Using ETag for cache check: {content_hash[:20]}...")
                            elif last_modified:
                                content_hash = hashlib.sha256(last_modified.encode()).hexdigest()
                                logging.info(f"[DOC] Using Last-Modified for cache check")
            except Exception as e:
                logging.warning(f"[DOC] Could not check headers, will download: {e}")
                
            if input_str in self.vector_store_cache and content_hash and self.document_hashes.get(input_str) == content_hash:
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
            try:
                timeout = aiohttp.ClientTimeout(total=60, connect=10)  # Increased timeout
                async with aiohttp.ClientSession(timeout=timeout) as session:
                    logging.info(f"[DOC] Downloading {doc_type.upper()} from: {input_str[:100]}...")
                    async with session.get(input_str) as response:
                        logging.info(f"[DOC] Response status: {response.status}")
                        response.raise_for_status()
                        
                        # Stream download for large files
                        content = bytearray()
                        downloaded = 0
                        async for chunk in response.content.iter_chunked(8192):
                            content.extend(chunk)
                            downloaded += len(chunk)
                            if downloaded % (1024 * 1024) == 0:  # Log every MB
                                logging.info(f"[DOC] Downloaded {downloaded // (1024*1024)}MB...")
                        
                        content = bytes(content)
                        content_hash = hashlib.sha256(content).hexdigest()
                        logging.info(f"[DOC] Download complete. Size: {len(content)} bytes")
                        
            except asyncio.TimeoutError:
                logging.error(f"[DOC] Timeout downloading {input_str}")
                raise ValueError(f"Timeout downloading document: {input_str}")
            except Exception as e:
                logging.error(f"[DOC] Error downloading {input_str}: {e}")
                raise ValueError(f"Failed to download document: {e}")
                
            suffix = ".pdf" if doc_type == "pdf" else ".docx"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
                temp_file.write(content)
                temp_file_path = temp_file.name
            try:
                logging.info(f"[DOC] Processing {doc_type.upper()} file...")
                chunks = await process_pdf(temp_file_path) if doc_type == "pdf" else await process_docx(temp_file_path)
                logging.info(f"[DOC] Extracted {len(chunks)} chunks from {doc_type.upper()}")
            finally:
                os.remove(temp_file_path)
        elif doc_type == "email":
            content_hash = hashlib.sha256(input_str.encode()).hexdigest()
            chunks = await process_email(input_str)
        else:
            raise ValueError(f"Unsupported document type: {doc_type}")

        if not chunks:
            raise ValueError(f"No content extracted from: {input_str}")

        # Extract text from Document objects
        chunk_texts = [doc.page_content for doc in chunks]
        embeddings = await self.generate_embeddings(chunk_texts)
        if len(embeddings) != len(chunk_texts):
            raise ValueError("Failed to embed all chunks")

        # Store embeddings in document metadata
        chunks_with_embeddings = [Document(page_content=doc.page_content, metadata={**doc.metadata, "embedding": e})
                                 for doc, e in zip(chunks, embeddings)]

        # Create FAISS index with pre-computed embeddings - optimized for speed
        embeddings_array = np.array(embeddings).astype('float32')
        dimension = embeddings_array.shape[1]
        
        # Always use flat index for better speed on smaller datasets
        # IVF clustering was causing the warning and slower performance
        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings_array)
        logging.info(f"[FAISS] Created flat index with {len(embeddings)} vectors")

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
        """Advanced hybrid reranking with clause-aware scoring for maximum accuracy"""
        if not docs:
            return []
            
        query_lower = query.lower()
        query_keywords = set(query_lower.split())
        
        def calculate_hybrid_score(doc: Document, semantic_score: float) -> float:
            content_lower = doc.page_content.lower()
            metadata = doc.metadata
            
            # Base semantic similarity (50% weight for precision)
            final_score = semantic_score * 0.5
            
            # Clause-specific bonuses (30% weight)
            clause_bonus = 0.0
            if metadata.get("clauses"):
                # Bonus for having clause references
                clause_bonus += 0.15
                
                # Extra bonus if query mentions specific clause numbers
                query_clause_refs = []
                import re
                for pattern in [r'\b\d+\.\d+(?:\.\w+)?\b', r'\bclause\s+\d+', r'\bsection\s+\d+']:
                    query_clause_refs.extend(re.findall(pattern, query_lower))
                
                if query_clause_refs:
                    doc_clauses = [str(c).lower() for c in metadata["clauses"]]
                    if any(qref in " ".join(doc_clauses) for qref in query_clause_refs):
                        clause_bonus += 0.15  # Perfect clause match
            
            final_score += clause_bonus * 0.3
            
            # Content type relevance (10% weight)
            chunk_type = metadata.get("chunk_type", "general")
            type_bonus = 0.0
            
            # Match query intent to chunk type
            if any(word in query_lower for word in ['definition', 'means', 'what is']):
                type_bonus = 0.2 if chunk_type == "definition" else 0.0
            elif any(word in query_lower for word in ['exclusion', 'excluded', 'not covered']):
                type_bonus = 0.2 if chunk_type == "exclusion" else 0.0
            elif any(word in query_lower for word in ['coverage', 'covered', 'benefit']):
                type_bonus = 0.2 if chunk_type == "coverage" else 0.0
            elif any(word in query_lower for word in ['waiting period', 'when', 'effective']):
                type_bonus = 0.2 if chunk_type == "waiting_period" else 0.0
            
            final_score += type_bonus * 0.1
            
            # Keyword overlap with phrase matching (10% weight)
            doc_words = set(content_lower.split())
            keyword_overlap = len(query_keywords.intersection(doc_words)) / max(len(query_keywords), 1)
            
            # Exact phrase bonus
            phrase_bonus = 0.0
            for keyword in query_keywords:
                if len(keyword) > 3 and keyword in content_lower:
                    phrase_bonus += 0.1
            
            final_score += (keyword_overlap + min(phrase_bonus, 0.2)) * 0.1
            
            # Quality indicators (10% weight)
            quality_score = 0.0
            
            # Prefer chunks with reasonable length (not too short/long)
            word_count = metadata.get("word_count", len(doc.page_content.split()))
            if 50 <= word_count <= 400:  # Optimal range
                quality_score += 0.1
            
            # Penalty for very short chunks
            if word_count < 25:
                quality_score -= 0.2
                
            final_score += quality_score * 0.1
            
            return max(0.0, min(1.0, final_score))  # Clamp between 0 and 1
        
        # Calculate semantic scores
        chunk_embs = [doc.metadata["embedding"] for doc in docs]
        semantic_scores = [self.cosine_similarity(query_emb, np.array(emb)) for emb in chunk_embs]
        
        # Calculate hybrid scores with detailed logging
        scored_docs = []
        for doc, sem_score in zip(docs, semantic_scores):
            hybrid_score = calculate_hybrid_score(doc, sem_score)
            scored_docs.append((hybrid_score, doc, sem_score))
        
        # Sort by hybrid score (highest first)
        scored_docs.sort(key=lambda x: x[0], reverse=True)
        
        # Log top results for debugging (reduced logging)
        if logging.getLogger().isEnabledFor(logging.INFO):
            logging.info(f"[RERANK] Top 3 results for: '{query[:30]}...'")
            for i, (score, doc, sem_score) in enumerate(scored_docs[:3]):
                clauses = doc.metadata.get("clauses", [])
                chunk_type = doc.metadata.get("chunk_type", "general")
                preview = doc.page_content[:60].replace('\n', ' ')
                logging.info(f"[RERANK] #{i+1}: Score={score:.3f} Type={chunk_type} Text='{preview}...'")
        
        # Return top K results
        return [doc for _, doc, _ in scored_docs[:settings.rerank_top_k]]

    def preprocess_query(self, query: str) -> str:
        """Enhanced query preprocessing for insurance policy documents"""
        # Normalize query
        query = query.strip()
        
        # Insurance domain-specific query expansion
        insurance_synonyms = {
            # Coverage terms
            'coverage': ['benefit', 'protection', 'insurance', 'covered'],
            'benefit': ['coverage', 'protection', 'allowance', 'compensation'],
            'claim': ['settlement', 'reimbursement', 'payment', 'compensation'],
            
            # Medical terms
            'maternity': ['pregnancy', 'childbirth', 'delivery', 'obstetric'],
            'dental': ['oral', 'tooth', 'teeth', 'orthodontic'],
            'vision': ['eye', 'optical', 'glasses', 'contact lens'],
            'mental health': ['psychiatric', 'psychology', 'behavioral health'],
            
            # Time-related
            'waiting period': ['waiting time', 'qualification period', 'elimination period'],
            'effective date': ['start date', 'commencement date', 'inception date'],
            
            # Exclusions
            'excluded': ['not covered', 'exception', 'limitation', 'restriction'],
            'pre-existing': ['PED', 'prior condition', 'existing condition'],
            
            # Amounts and limits
            'deductible': ['excess', 'co-payment', 'copay'],
            'maximum': ['limit', 'cap', 'ceiling', 'upper limit'],
            'premium': ['cost', 'price', 'fee', 'payment']
        }
        
        # Expand query with relevant synonyms
        expanded_terms = []
        query_words = query.lower().split()
        
        for word in query_words:
            if word in insurance_synonyms:
                expanded_terms.extend(insurance_synonyms[word][:2])  # Add top 2 synonyms
        
        # Check for multi-word phrases
        query_lower = query.lower()
        for phrase, synonyms in insurance_synonyms.items():
            if len(phrase.split()) > 1 and phrase in query_lower:
                expanded_terms.extend(synonyms[:2])
        
        # Combine original query with expansions
        if expanded_terms:
            enhanced_query = f"{query} {' '.join(set(expanded_terms))}"
        else:
            enhanced_query = query
        
        # Add clause-specific search terms if numbers detected
        import re
        if re.search(r'\b\d+\.\d+', query):
            enhanced_query += " clause section provision"
        
        logging.info(f"[QUERY] Enhanced {len(expanded_terms)} terms for: '{query[:40]}...'")
        return enhanced_query.strip()

    async def process_batch_queries(self, document_url: str, questions: List[str]) -> List[str]:
        batch_start_time = asyncio.get_event_loop().time()
        
        # Only clear cache if it's a different document
        current_doc_hash = hashlib.sha256(document_url.encode()).hexdigest()
        if not hasattr(self, '_last_doc_hash') or self._last_doc_hash != current_doc_hash:
            logging.info("[CACHE] Processing new document - clearing cache")
            self.flush_vector_cache()
            self.clear_disk_cache()
            self._last_doc_hash = current_doc_hash
        else:
            logging.info("[CACHE] Same document detected - reusing cache")
        
        # Process document - no timeouts, let it finish properly
        vector_store = await self.process_document(document_url)
        index = vector_store.index
        docstore = vector_store.docstore

        async def handle(q: str) -> str:
            start_time = asyncio.get_event_loop().time()
            
            try:
                # Process query properly without rushing
                enhanced_query = self.preprocess_query(q)
                
                # Generate embedding - no timeout
                query_emb = await self.generate_embeddings([enhanced_query])
                
                if not query_emb:
                    return "Failed to embed query"
                    
                query_emb = np.array(query_emb[0]).astype('float32').reshape(1, -1)

                # Vector search
                D, I = index.search(query_emb, k=settings.max_retrieval_chunks)
                docs = [docstore[i] for i in I[0] if i in docstore]

                # Reranking
                reranked = self.rerank_chunks(q, docs, query_emb[0])
                
                # Create enhanced context with clause metadata
                context_parts = []
                for i, doc in enumerate(reranked):
                    clauses = doc.metadata.get("clauses", [])
                    chunk_type = doc.metadata.get("chunk_type", "general")
                    page = doc.metadata.get("page", "unknown")
                    
                    clause_info = f" [Clauses: {', '.join(clauses)}]" if clauses else ""
                    type_info = f" [Type: {chunk_type}]" if chunk_type != "general" else ""
                    page_info = f" [Page: {page}]"
                    
                    context_parts.append(f"Context {i+1}{clause_info}{type_info}{page_info}:\n{doc.page_content.strip()}")
                
                context_text = "\n\n" + "\n\n".join(context_parts)

                # LLM call - no timeout, let it think properly
                response = await self.llm_chain.ainvoke({"query": q, "clauses": context_text})
                
                parsed_response = self.parse_llm_response(response)
                answer = parsed_response.get("answer", "Unable to determine")
                confidence = parsed_response.get("confidence_score", 0.5)
                clause_refs = parsed_response.get("clause_references", [])
                coverage_status = parsed_response.get("coverage_status", "not_mentioned")
                
                # Enhanced logging for quality monitoring
                end_time = asyncio.get_event_loop().time()
                response_time = end_time - start_time
                
                query_preview = q[:60] + ('...' if len(q) > 60 else '')
                answer_preview = answer[:200] + ('...' if len(answer) > 200 else '')
                
                logging.info(f"[ANSWER] Query: '{query_preview}'")
                logging.info(f"[ANSWER] Response Time: {response_time:.2f}s")
                logging.info(f"[ANSWER] Coverage Status: {coverage_status}")
                logging.info(f"[ANSWER] Confidence: {confidence:.2f}")
                logging.info(f"[ANSWER] Clause References: {clause_refs}")
                logging.info(f"[ANSWER] Answer: {answer_preview}")
                
                # Quality validation warnings
                if "validation_warnings" in parsed_response:
                    logging.warning(f"[VALIDATION] Quality concerns: {parsed_response['validation_warnings']}")
                
                # Return structured answer or just the text based on need
                return answer
                
            except Exception as e:
                logging.error(f"[ERROR] Query failed: {e}")
                return f"Query processing failed: {str(e)}"

        # Process all questions concurrently - no timeout limits
        results = await asyncio.gather(*(handle(q) for q in questions), return_exceptions=True)
        
        # Handle any exceptions in results
        final_results = []
        for result in results:
            if isinstance(result, Exception):
                final_results.append(f"Query processing error: {str(result)}")
            else:
                final_results.append(result)
        
        # Calculate total batch processing time
        batch_end_time = asyncio.get_event_loop().time()
        total_time = batch_end_time - batch_start_time
        avg_time_per_question = total_time / len(questions) if questions else 0
        
        logging.info(f"[BATCH] Completed {len(questions)} questions in {total_time:.2f}s")
        logging.info(f"[BATCH] Average time per question: {avg_time_per_question:.2f}s")
        
        # Optional: Flush cache after processing all questions to ensure clean state
        # Uncomment the lines below if you want to clear cache after each batch
        # logging.info("[BATCH] Flushing cache after completing all questions")
        # self.flush_vector_cache()
        # self.clear_disk_cache()
        
        return final_results

    def parse_llm_response(self, response: Any) -> Dict[str, Any]:
        """Enhanced response parsing with validation and fallback"""
        if hasattr(response, "content"):
            response = response.content
        response = str(response).strip()
        
        # Clean up markdown formatting
        if response.startswith("```json"):
            response = response[7:-3].strip()
        elif response.startswith("```"):
            response = response[3:-3].strip()
        
        try:
            parsed = json.loads(response)
            
            # Validate required fields and add defaults
            validated_response = {
                "answer": parsed.get("answer", "Unable to determine from provided clauses"),
                "confidence_score": min(max(parsed.get("confidence_score", 0.5), 0.0), 1.0),
                "clause_references": parsed.get("clause_references", []),
                "coverage_status": parsed.get("coverage_status", "not_mentioned")
            }
            
            # Validate coverage_status field
            valid_statuses = ["covered", "excluded", "conditional", "not_mentioned"]
            if validated_response["coverage_status"] not in valid_statuses:
                validated_response["coverage_status"] = "not_mentioned"
            
            # Quality checks on the answer
            answer = validated_response["answer"]
            
            # Check for hallucination indicators
            hallucination_flags = []
            if "typically" in answer.lower() or "usually" in answer.lower():
                hallucination_flags.append("generic_language")
            if "most policies" in answer.lower() or "generally" in answer.lower():
                hallucination_flags.append("generalization")
            
            # Add warning if potential hallucination detected
            if hallucination_flags:
                logging.warning(f"[VALIDATION] Potential hallucination detected: {hallucination_flags}")
                validated_response["validation_warnings"] = hallucination_flags
            
            # Ensure answer quality
            if len(answer.strip()) < 10:
                validated_response["answer"] = "Insufficient information in provided clauses to answer this question."
                validated_response["confidence_score"] = 0.1
            
            return validated_response
            
        except json.JSONDecodeError as e:
            logging.error(f"[JSON ERROR] Failed to parse JSON from LLM: {e}")
            logging.error(f"[JSON ERROR] Raw response: {response[:200]}...")
            
            # Fallback parsing - extract answer from text
            fallback_answer = self._extract_answer_from_text(response)
            
            return {
                "answer": fallback_answer,
                "confidence_score": 0.3,  # Low confidence for fallback
                "clause_references": [],
                "coverage_status": "not_mentioned",
                "parse_error": "JSON parsing failed, used fallback extraction"
            }
    
    def _extract_answer_from_text(self, text: str) -> str:
        """Fallback method to extract answer when JSON parsing fails"""
        lines = text.split('\n')
        
        # Look for answer-like content
        for line in lines:
            line = line.strip()
            if line and not line.startswith('#') and len(line) > 20:
                # Clean up the line
                if line.startswith('"') and line.endswith('"'):
                    line = line[1:-1]
                return line
        
        # If no good line found, return the first substantial paragraph
        if len(text.strip()) > 50:
            return text.strip()[:500] + "..." if len(text) > 500 else text.strip()
        
        return "Unable to extract answer from LLM response"

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

@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": asyncio.get_event_loop().time()}

@app.post("/hackrx/run", response_model=BatchQueryResponse)
async def process_batch_queries(request: BatchQueryRequest):
    try:
        logging.info(f"[API] Received request with {len(request.questions)} questions")
        
        # No timeout - let it complete properly for maximum accuracy
        start_time = asyncio.get_event_loop().time()
        answers = await query_system.process_batch_queries(request.documents, request.questions)
        
        end_time = asyncio.get_event_loop().time()
        total_api_time = end_time - start_time
        
        logging.info(f"[API] Completed processing in {total_api_time:.2f}s, returning {len(answers)} answers")
        return BatchQueryResponse(answers=answers)
        
    except Exception as e:
        logging.error(f"[API] Error in batch processing: {e}")
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logging.error(f"[ERROR] Unhandled Exception: {exc}")
    return JSONResponse(status_code=500, content={"error": "Internal server error", "detail": str(exc)})

if __name__ == "__main__":
    uvicorn.run("app:app", host=settings.api_host, port=settings.api_port, reload=settings.debug)