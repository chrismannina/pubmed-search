import os
import datetime
import time
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from pydantic import HttpUrl
from typing import List, Optional, Literal
from contextlib import asynccontextmanager # For lifespan management
from dotenv import load_dotenv

# --- Import Logic --- 
# Assuming search.py is in the same directory
from search import (
    Author as LogicAuthor,
    PubMedArticle as LogicPubMedArticle,
    generate_mesh_terms,
    search_pubmed_keywords,
    ensure_articles_in_db,
    semantic_rank_articles,
    hybrid_rank_rrf,
    DEFAULT_LLM_MESH_MODEL,
    DEFAULT_EMBED_CONTENT_MODE,
    OPENAI_EMBED_MODEL # Need the specific embedding model name
)
from Bio import Entrez
from openai import OpenAI
import chromadb

# --- Global Variables & Clients --- 
# These will be initialized during startup
config = {}
openai_client: Optional[OpenAI] = None
chroma_client: Optional[chromadb.PersistentClient] = None
chroma_collection: Optional[chromadb.Collection] = None

# --- API Request/Response Models --- 
# (Keep the models defined earlier: SearchRequest, AuthorAPI, SearchResultItem, SearchResponse)
class SearchRequest(BaseModel):
    query: str = Field(..., description="The user's natural language search query.")
    ranking_mode: Literal["pubmed", "semantic", "hybrid"] = Field(default="hybrid", description="The ranking strategy to use.")
    min_year: Optional[int] = Field(default=None, description="Optional minimum publication year to filter results before ranking.")
    top_k: int = Field(default=10, gt=0, le=50, description="Number of top results to return.")
    max_pubmed_results: int = Field(default=100, gt=0, le=500, description="Maximum results from initial PubMed search.")
    # Allow overriding certain configs per request
    llm_mesh_model: Optional[str] = Field(default=None, description="Override LLM model for MeSH generation.")
    embed_content_mode: Optional[Literal["abstract", "title_abstract"]] = Field(default=None, description="Override content embedding mode.")

class AuthorAPI(BaseModel):
    lastName: Optional[str] = None
    foreName: Optional[str] = None
    initials: Optional[str] = None
    affiliation: Optional[str] = None
    display_name: str

class SearchResultItem(BaseModel):
    pmid: str
    rank: int
    score: Optional[float] = None
    title: Optional[str] = None
    abstract_snippet: Optional[str] = None
    authors: List[AuthorAPI] = []
    journal: Optional[str] = None
    pubDate: Optional[str] = None
    publicationTypes: List[str] = []
    meshHeadings: List[str] = []
    keywords: List[str] = []
    language: Optional[str] = None
    doi: Optional[str] = None
    pubmed_url: str # Keep as str for API response consistency
    doi_url: Optional[str] = None # Use str here for simplicity in the model

class SearchResponse(BaseModel):
    query: str
    ranking_mode: str
    min_year_filter: Optional[int]
    generated_mesh_terms: Optional[List[str]] = None
    pubmed_query_used: Optional[str] = None
    results_count: int
    results: List[SearchResultItem]

# --- Lifespan Management for Startup/Shutdown --- 
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Load config and initialize clients
    print("API Starting up...")
    global config, openai_client, chroma_client, chroma_collection
    
    load_dotenv()
    config['NCBI_EMAIL'] = os.getenv("NCBI_EMAIL", "your.email@example.com")
    config['OPENAI_API_KEY'] = os.getenv("OPENAI_API_KEY")
    config['CHROMA_DB_PATH'] = "./chroma_db"
    config['COLLECTION_NAME'] = "pubmed_articles_openai"
    config['LLM_MESH_MODEL'] = DEFAULT_LLM_MESH_MODEL
    config['EMBED_CONTENT_MODE'] = DEFAULT_EMBED_CONTENT_MODE
    config['OPENAI_EMBED_MODEL'] = OPENAI_EMBED_MODEL

    if not config['OPENAI_API_KEY']:
        raise ValueError("OPENAI_API_KEY not found in environment variables.")
    if config['NCBI_EMAIL'] == "your.email@example.com":
        print("Warning: Using default NCBI email. Set NCBI_EMAIL in .env")

    print("Initializing OpenAI client...")
    openai_client = OpenAI(api_key=config['OPENAI_API_KEY'])
    
    print(f"Initializing persistent ChromaDB client at: {config['CHROMA_DB_PATH']}")
    chroma_client = chromadb.PersistentClient(path=config['CHROMA_DB_PATH'])
    
    print(f"Getting or creating ChromaDB collection: {config['COLLECTION_NAME']}")
    try:
        chroma_collection = chroma_client.get_or_create_collection(
            name=config['COLLECTION_NAME'],
            metadata={"hnsw:space": "cosine"}
        )
        print("ChromaDB collection ready.")
    except Exception as e:
        print(f"FATAL: Could not initialize ChromaDB collection: {e}")
        # Decide how to handle - maybe raise error to stop API?
        raise RuntimeError("ChromaDB initialization failed") from e

    print("API Startup complete.")
    yield
    # Shutdown: Cleanup (if necessary)
    print("API Shutting down...")


# --- FastAPI Application Setup --- 
app = FastAPI(
    title="PubMed Semantic Search API",
    description="API for performing MeSH-refined semantic search on PubMed articles.",
    version="0.1.0",
    lifespan=lifespan # Add lifespan manager
)

# --- API Endpoints ---

@app.get("/", include_in_schema=False)
async def root():
    return {"message": "PubMed Search API is running."}

@app.post("/search", response_model=SearchResponse)
async def perform_search(request: SearchRequest):
    """Performs MeSH-refined PubMed search with semantic/hybrid ranking."""
    start_time = time.time()
    print(f"Received search request: Query='{request.query}', Mode='{request.ranking_mode}', Year>='{request.min_year}', K='{request.top_k}'")

    # Ensure clients are initialized (should be by lifespan)
    if not openai_client or not chroma_collection:
        raise HTTPException(status_code=500, detail="API clients not initialized.")

    # Get config overrides or defaults
    llm_mesh_model = request.llm_mesh_model or config['LLM_MESH_MODEL']
    embed_content_mode = request.embed_content_mode or config['EMBED_CONTENT_MODE']
    entrez_email = config['NCBI_EMAIL']
    embed_model = config['OPENAI_EMBED_MODEL']

    # --- Core Logic Execution --- 
    # 0. Generate MeSH terms
    generated_mesh_terms = generate_mesh_terms(request.query, openai_client, model=llm_mesh_model)

    # 1. Initial Keyword Search
    initial_pmids_ordered, pubmed_query_used = search_pubmed_keywords(
        request.query, 
        mesh_terms=generated_mesh_terms, 
        max_results=request.max_pubmed_results,
        entrez_email=entrez_email
    )
    if not initial_pmids_ordered:
        # Return empty response if no initial results
        return SearchResponse(
            query=request.query, ranking_mode=request.ranking_mode, min_year_filter=request.min_year,
            generated_mesh_terms=generated_mesh_terms, pubmed_query_used=pubmed_query_used, 
            results_count=0, results=[]
        )

    # 2. Ensure Articles are in DB
    pmids_in_db = ensure_articles_in_db(
        initial_pmids_ordered, 
        chroma_collection, 
        openai_client, 
        embed_model, 
        embed_content_mode, 
        entrez_email
    )
    if not pmids_in_db:
        # Return empty response if no articles could be processed
        return SearchResponse(
            query=request.query, ranking_mode=request.ranking_mode, min_year_filter=request.min_year,
            generated_mesh_terms=generated_mesh_terms, pubmed_query_used=pubmed_query_used, 
            results_count=0, results=[]
        )

    # 3. Optional Pre-Ranking Date Filter
    eligible_pmids_for_ranking = pmids_in_db
    if request.min_year:
        print(f"Applying pre-ranking date filter: Year >= {request.min_year}")
        # Fetch metadata needed for filtering
        metadata_for_filtering = {}
        try:
            filter_data = chroma_collection.get(ids=pmids_in_db, include=['metadatas'])
            if filter_data and filter_data.get('ids'):
                 for i, pmid in enumerate(filter_data['ids']):
                     meta = filter_data['metadatas'][i] if i < len(filter_data.get('metadatas', [])) else {}
                     metadata_for_filtering[pmid] = meta
        except Exception as e:
            print(f"Warning: Could not retrieve metadata for date filtering: {e}. Filter skipped.")
        
        if metadata_for_filtering:
             temp_eligible_pmids = []
             for pmid in pmids_in_db:
                 meta = metadata_for_filtering.get(pmid)
                 keep = True
                 if meta and meta.get('pub_date'):
                     try:
                         article_year = int(meta['pub_date'].split('-')[0])
                         if article_year < request.min_year:
                             keep = False
                     except: pass
                 if keep: temp_eligible_pmids.append(pmid)
             eligible_pmids_for_ranking = temp_eligible_pmids
             print(f"{len(eligible_pmids_for_ranking)} articles meet date criteria.")

    if not eligible_pmids_for_ranking:
        return SearchResponse(
            query=request.query, ranking_mode=request.ranking_mode, min_year_filter=request.min_year,
            generated_mesh_terms=generated_mesh_terms, pubmed_query_used=pubmed_query_used,
            results_count=0, results=[] # No results after date filter
        )

    # Filter the original ordered list based on eligibility
    pubmed_ranked_eligible_pmids = [pmid for pmid in initial_pmids_ordered if pmid in eligible_pmids_for_ranking]

    # 4. Perform Ranking
    final_ranked_pmid_scores = [] # List of (pmid, score)
    
    if request.ranking_mode == "pubmed":
        print(f"Using PubMed relevance ranking on {len(pubmed_ranked_eligible_pmids)} eligible articles...")
        final_ranked_pmid_scores = [(pmid, i+1.0) for i, pmid in enumerate(pubmed_ranked_eligible_pmids[:request.top_k])]
    
    elif request.ranking_mode == "semantic":
        print(f"Performing semantic ranking on {len(eligible_pmids_for_ranking)} eligible articles...")
        final_ranked_pmid_scores = semantic_rank_articles(
            request.query, eligible_pmids_for_ranking, chroma_collection, request.top_k, openai_client, embed_model
        )
    
    elif request.ranking_mode == "hybrid":
        print(f"Performing hybrid ranking (RRF) on {len(eligible_pmids_for_ranking)} eligible articles...")
        print(f"  Getting semantic scores for {len(eligible_pmids_for_ranking)} articles...")
        all_semantic_results = semantic_rank_articles(
            request.query, eligible_pmids_for_ranking, chroma_collection, len(eligible_pmids_for_ranking), openai_client, embed_model
        )
        if not all_semantic_results:
             print("  Warning: Failed to get semantic scores for RRF. Falling back to PubMed ranking.")
             final_ranked_pmid_scores = [(pmid, i+1.0) for i, pmid in enumerate(pubmed_ranked_eligible_pmids[:request.top_k])]
        else:
             rrf_results = hybrid_rank_rrf(pubmed_ranked_eligible_pmids, all_semantic_results)
             final_ranked_pmid_scores = rrf_results[:request.top_k]

    # 5. Fetch Metadata for Final Results and Format Response
    final_pmids = [pmid for pmid, _ in final_ranked_pmid_scores]
    search_result_items: List[SearchResultItem] = []
    display_data = {}
    if final_pmids:
        try:
            top_articles_data = chroma_collection.get(ids=final_pmids, include=['documents', 'metadatas'])
            if top_articles_data and top_articles_data.get('ids'):
                retrieved_metadatas = top_articles_data.get('metadatas', [])
                retrieved_documents = top_articles_data.get('documents', [])
                for i, pmid_retrieved in enumerate(top_articles_data['ids']):
                     metadata = retrieved_metadatas[i] if i < len(retrieved_metadatas) else {}
                     document = retrieved_documents[i] if i < len(retrieved_documents) else ''
                     display_data[pmid_retrieved] = {'metadata': metadata, 'document': document}
        except Exception as e:
            print(f"Warning: Could not retrieve full metadata for display: {e}")

    # Format final response items
    for rank, (pmid, score) in enumerate(final_ranked_pmid_scores):
        article_info = display_data.get(pmid)
        meta = article_info.get('metadata', {}) if article_info else {}
        doc = article_info.get('document', '') if article_info else ''
        
        authors_str = meta.get('authors', '')
        authors_api = [
            AuthorAPI(display_name=name.strip()) 
            for name in authors_str.split(', ') if name.strip()
        ]
        
        doi = meta.get('doi')
        doi_url_str = f"https://doi.org/{doi}" if doi else None
        
        item = SearchResultItem(
            pmid=pmid,
            rank=rank + 1,
            score=score,
            title=meta.get('title'),
            abstract_snippet=doc[:300] + '...' if doc else None,
            authors=authors_api,
            journal=meta.get('journal'),
            pubDate=meta.get('pub_date'),
            publicationTypes=meta.get('publication_types', '').split(', ') if meta.get('publication_types') else [],
            meshHeadings=meta.get('mesh_headings', '').split(', ') if meta.get('mesh_headings') else [],
            keywords=meta.get('keywords', '').split(', ') if meta.get('keywords') else [],
            language=meta.get('language'),
            doi=doi,
            pubmed_url=f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
            doi_url=doi_url_str
        )
        search_result_items.append(item)

    end_time = time.time()
    print(f"Search completed in {end_time - start_time:.2f} seconds.")
    
    return SearchResponse(
        query=request.query,
        ranking_mode=request.ranking_mode,
        min_year_filter=request.min_year,
        generated_mesh_terms=generated_mesh_terms,
        pubmed_query_used=pubmed_query_used,
        results_count=len(search_result_items),
        results=search_result_items
    )

# --- Placeholder for main execution (if needed for direct run - usually done via uvicorn) ---
# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000) 