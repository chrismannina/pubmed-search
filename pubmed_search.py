import requests
from Bio import Entrez
import chromadb
from openai import OpenAI
import os
from dotenv import load_dotenv
import time
import datetime

# --- Configuration ---
load_dotenv() # Load environment variables from .env file

# NCBI requires you to identify yourself.
Entrez.email = os.getenv("NCBI_EMAIL", "your.email@example.com") # Get email from env or use default
if Entrez.email == "your.email@example.com":
    print("Warning: Using default NCBI email. Please set NCBI_EMAIL in your .env file or script.")

# OpenAI API Key and Client
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("OPENAI_API_KEY not found in environment variables. Please set it in your .env file.")
client = OpenAI(api_key=openai_api_key)

# Model for OpenAI embeddings
OPENAI_EMBED_MODEL = "text-embedding-3-small"

# Content to embed ('abstract' or 'title_abstract')
EMBED_CONTENT_MODE = "title_abstract" # Defaulting to title + abstract

# ChromaDB Client and Collection
# Using a persistent client to save the DB to disk
CHROMA_DB_PATH = "./chroma_db"
print(f"Initializing persistent ChromaDB client at: {CHROMA_DB_PATH}")
chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)

# Create a collection (or get it if it already exists)
collection_name = "pubmed_articles_openai" # Changed name slightly for clarity
print(f"Getting or creating ChromaDB collection: {collection_name}")
# Using cosine distance, standard for semantic search
collection = chroma_client.get_or_create_collection(name=collection_name, metadata={"hnsw:space": "cosine"})

# Maximum number of results from initial keyword search
MAX_KEYWORD_RESULTS = 100
# Top K results to return after re-ranking
TOP_K = 10

# LLM Model for MeSH generation (can be changed, e.g., "gpt-4o")
LLM_MESH_MODEL = "gpt-4o" #"gpt-3.5-turbo"

# RRF constant (common value)
RRF_K = 60

# --- Functions ---

def generate_mesh_terms(query: str, llm_client: OpenAI, model: str = LLM_MESH_MODEL) -> list[str]:
    """Uses an LLM to generate relevant MeSH terms from a user query."""
    print(f"Generating MeSH terms for query: '{query}' using {model}...")
    
    # Simple prompt - can be refined significantly
    prompt = (
        f"You are an expert medical librarian assisting with PubMed searches. "
        f"Based on the following user query, please identify the most relevant official MeSH (Medical Subject Headings) terms. "
        f"Return ONLY a comma-separated list of the MeSH terms (maximum 10 terms). Do not include explanations or introductory text. "
        f"Query: \"{query}\""
    )
    
    try:
        response = llm_client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are an expert medical librarian specializing in MeSH terms."}, 
                {"role": "user", "content": prompt}
            ],
            temperature=0.2, # Lower temperature for more focused, predictable output
            max_tokens=100 
        )
        
        raw_terms = response.choices[0].message.content.strip()
        print(f"  LLM Raw Output: {raw_terms}")
        
        # Parse the comma-separated list
        mesh_terms = [term.strip() for term in raw_terms.split(',') if term.strip()]
        
        # Basic validation/cleanup (optional)
        # Remove any terms that are very short or look like fragments
        mesh_terms = [term for term in mesh_terms if len(term) > 2]
        
        print(f"  Generated MeSH terms: {mesh_terms}")
        return mesh_terms
        
    except Exception as e:
        print(f"Error generating MeSH terms: {e}")
        return [] # Return empty list on error

def search_pubmed_keywords(query: str, mesh_terms: list[str] | None = None, max_results: int = MAX_KEYWORD_RESULTS) -> list[str]:
    """Performs a keyword search on PubMed, prioritizing MeSH terms if available."""
    
    final_query = ""
    query_description = ""
    
    if mesh_terms:
        # Prioritize using only MeSH terms if available
        # Format: (term1[MeSH Terms] OR term2[MeSH Terms] ...)
        mesh_query_part = " OR ".join([f'\"{term}\"[MeSH Terms]' for term in mesh_terms])
        if mesh_query_part:
             final_query = mesh_query_part
             query_description = "using generated MeSH terms"
             print(f"Constructed PubMed query based on MeSH terms.")
        else:
             print("Warning: MeSH terms were provided but resulted in an empty query part. Falling back to original query.")
             final_query = query
             query_description = "using original query (MeSH generation failed)"
    else:
        # Fallback to original query if no MeSH terms generated
        print("No MeSH terms generated or provided. Using original query text.")
        final_query = query
        query_description = "using original query"

    print(f"Executing PubMed query ({query_description}): {final_query}")
    
    try:
        handle = Entrez.esearch(db="pubmed", term=final_query, retmax=str(max_results), sort="relevance")
        record = Entrez.read(handle)
        handle.close()
        pmids = record.get("IdList", [])
        print(f"Found {len(pmids)} potential PMIDs {query_description}.")
        
        # If MeSH query returned few/no results, optionally try original query as fallback?
        # Let's keep it simple for now: trust the MeSH query if it was attempted.
        # Adding a fallback here might re-introduce noise.
        
        return pmids
    except Exception as e:
        print(f"Error during PubMed keyword search ({query_description}): {e}")
        # If the MeSH query failed, definitely try the original query as a fallback
        if mesh_terms and final_query != query: # Check if we actually used MeSH
             print(f"MeSH-based query failed. Falling back to original query: {query}")
             try:
                  handle = Entrez.esearch(db="pubmed", term=query, retmax=str(max_results), sort="relevance")
                  record = Entrez.read(handle)
                  handle.close()
                  pmids = record.get("IdList", [])
                  print(f"Found {len(pmids)} potential PMIDs using fallback original query.")
                  return pmids
             except Exception as e2:
                  print(f"Error during fallback PubMed keyword search: {e2}")
                  return []
        else:
             # Original query failed, or MeSH query was same as original and failed
             return []

def parse_pubmed_date(date_info: dict) -> str | None:
    """Helper function to parse various PubMed date formats."""
    try:
        year = int(date_info.get('Year'))
        month_str = date_info.get('Month')
        day_str = date_info.get('Day')

        # Attempt to convert month string/number to number
        month_map = { 'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6,
                      'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12 }
        if month_str in month_map:
            month = month_map[month_str]
        elif month_str and month_str.isdigit():
            month = int(month_str)
        else:
            month = 1 # Default to January if month is missing/invalid

        day = int(day_str) if day_str and day_str.isdigit() else 1 # Default to 1st if day is missing/invalid

        # Basic validation for day based on month
        if month in [4, 6, 9, 11] and day > 30:
            day = 30
        elif month == 2:
            is_leap = (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0)
            max_feb_day = 29 if is_leap else 28
            if day > max_feb_day:
                day = max_feb_day
        elif day > 31:
            day = 31

        return f"{year:04d}-{month:02d}-{day:02d}"
    except (ValueError, TypeError, AttributeError):
        # Handle cases where Year, Month, or Day are missing or not parsable
        # Try to get just the MedlineDate if PubDate fails
        medline_date = date_info.get('MedlineDate')
        if medline_date and isinstance(medline_date, str):
            # Basic parsing for YYYY format or YYYY Mon format
            parts = medline_date.split()
            if parts[0].isdigit() and len(parts[0]) == 4:
                year = int(parts[0])
                month = 1
                if len(parts) > 1:
                     month_str_medline = parts[1][:3]
                     if month_str_medline in month_map:
                         month = month_map[month_str_medline]
                return f"{year:04d}-{month:02d}-01" # Default to 1st day
        return None # Could not parse

def fetch_abstracts(pmids: list[str]) -> dict[str, dict[str, str | list[str] | None]]:
    """Fetches details (title, abstract, date, authors, journal, MeSH) for PMIDs.

    Args:
        pmids: A list of PubMed IDs.

    Returns:
        A dictionary where keys are PMIDs and values are dictionaries
        containing 'title', 'abstract', 'pub_date', 'authors', 'journal', 'mesh_headings'.
    """
    print(f"Fetching details for {len(pmids)} PMIDs...")
    articles = {}
    batch_size = 100 # Keep batch size reasonable for Entrez
    for i in range(0, len(pmids), batch_size):
        batch_pmids = pmids[i:i + batch_size]
        print(f"  Fetching batch {i//batch_size + 1} ({len(batch_pmids)} PMIDs)...")
        try:
            handle = Entrez.efetch(db="pubmed", id=batch_pmids, rettype="medline", retmode="xml")
            records = Entrez.read(handle)
            handle.close()

            for record in records.get('PubmedArticle', []):
                medline_citation = record.get('MedlineCitation', {})
                if not medline_citation:
                    continue

                pmid = str(medline_citation.get('PMID'))
                article = medline_citation.get('Article', {})
                title = article.get('ArticleTitle', 'No Title Available')

                # --- Extract Abstract ---
                abstract_list = article.get('Abstract', {}).get('AbstractText', [])
                if isinstance(abstract_list, list):
                    abstract = " ".join(abstract_list)
                else:
                    abstract = str(abstract_list)
                if not abstract:
                    other_abstract_list = article.get('OtherAbstract', [])
                    if other_abstract_list and isinstance(other_abstract_list, list) and len(other_abstract_list) > 0:
                        abstract_texts = other_abstract_list[0].get('AbstractText', [])
                        if isinstance(abstract_texts, list):
                             abstract = " ".join(abstract_texts)
                        else:
                             abstract = str(abstract_texts)
                
                # Only proceed if we have an abstract, as it's needed for embedding
                if not abstract:
                    print(f"  Warning: No abstract found for PMID: {pmid}. Skipping storage.")
                    continue

                # --- Extract Date ---
                journal_info = article.get('Journal', {})
                pub_date_info = journal_info.get('JournalIssue', {}).get('PubDate', {})
                pub_date = parse_pubmed_date(pub_date_info)

                # --- Extract Journal Title ---
                journal_title = journal_info.get('Title', 'No Journal Title')

                # --- Extract Authors ---
                author_list_data = article.get('AuthorList', [])
                authors = []
                if author_list_data and isinstance(author_list_data, list):
                    for author_info in author_list_data:
                        last_name = author_info.get('LastName')
                        fore_name = author_info.get('ForeName')
                        initials = author_info.get('Initials')
                        name_parts = [n for n in [fore_name, last_name] if n] # Filter out None/empty
                        if name_parts:
                            authors.append(" ".join(name_parts))
                        elif initials: # Fallback if only initials are present (rare)
                             authors.append(f"{initials}.")

                # --- Extract MeSH Headings ---
                mesh_heading_list = medline_citation.get('MeshHeadingList', [])
                mesh_headings = []
                if mesh_heading_list and isinstance(mesh_heading_list, list):
                    for mesh_item in mesh_heading_list:
                        descriptor = mesh_item.get('DescriptorName')
                        if descriptor:
                            # Sometimes descriptor is an object with 'UI' and text content
                            if hasattr(descriptor, 'attributes') and 'UI' in descriptor.attributes:
                                mesh_headings.append(str(descriptor)) # Get the text content
                            else:
                                 mesh_headings.append(str(descriptor))


                # --- Store Article ---
                articles[pmid] = {
                    'title': title,
                    'abstract': abstract,
                    'pub_date': pub_date,
                    'journal': journal_title,
                    'authors': authors, # List of strings
                    'mesh_headings': mesh_headings # List of strings
                }

            # Be nice to NCBI servers
            time.sleep(0.34) # NCBI recommends >= 3 requests/sec

        except Exception as e:
            print(f"Error fetching or parsing batch starting at index {i}: {e}")
            # Optionally add more robust error handling / retries

    print(f"Successfully fetched details for {len(articles)} articles with abstracts.")
    return articles

def get_openai_embeddings(texts: list[str], model: str = OPENAI_EMBED_MODEL) -> list[list[float]]:
    """Generates embeddings for a list of texts using OpenAI API."""
    if not texts:
        return []
    try:
        # Replace newlines for OpenAI API
        texts = [text.replace("\n", " ") for text in texts]
        response = client.embeddings.create(input=texts, model=model)
        return [embedding.embedding for embedding in response.data]
    except Exception as e:
        print(f"Error getting OpenAI embeddings: {e}")
        # Handle API errors (e.g., rate limits, invalid key)
        return []

# --- Function to manage ChromaDB persistence ---
def ensure_articles_in_db(pmids_to_check: list[str], collection: chromadb.Collection) -> list[str]:
    """Checks ChromaDB for PMIDs, fetches missing ones, embeds, and adds them.

    Args:
        pmids_to_check: List of PMIDs from the initial keyword search.
        collection: The ChromaDB collection object.

    Returns:
        List of PMIDs from the input that are confirmed to be in the DB
        (either pre-existing or newly added). Returns empty list on critical error.
    """
    if not pmids_to_check:
        return []

    print(f"Checking persistence for {len(pmids_to_check)} PMIDs...")
    try:
        existing_in_db = collection.get(ids=pmids_to_check, include=[]) # Only need IDs
        existing_ids = set(existing_in_db['ids'])
        print(f"Found {len(existing_ids)} PMIDs already in the persistent DB.")
    except Exception as e:
        print(f"Error checking existing documents in ChromaDB: {e}")
        # Decide how to handle: assume none exist? Abort? For now, assume none.
        existing_ids = set()

    pmids_to_fetch = [pmid for pmid in pmids_to_check if pmid not in existing_ids]

    newly_added_ids = set()
    if pmids_to_fetch:
        print(f"Fetching details for {len(pmids_to_fetch)} new PMIDs...")
        new_articles_data = fetch_abstracts(pmids_to_fetch)

        if new_articles_data:
            new_pmids = list(new_articles_data.keys())
            
            # --- Prepare documents for embedding based on config ---
            documents = []
            print(f"Preparing documents for embedding using mode: {EMBED_CONTENT_MODE}")
            for pmid in new_pmids:
                title = new_articles_data[pmid].get('title', '')
                abstract = new_articles_data[pmid].get('abstract', '')
                if EMBED_CONTENT_MODE == "title_abstract":
                    # Combine title and abstract, ensuring space if both exist
                    doc_text = f"{title}. {abstract}".strip()
                    documents.append(doc_text if doc_text != "." else "") # Handle empty title/abstract case
                else: # Default to abstract only
                    documents.append(abstract if abstract else "") # Handle empty abstract

            # Filter out articles where the document text ended up empty
            valid_indices = [i for i, doc in enumerate(documents) if doc]
            if len(valid_indices) < len(new_pmids):
                print(f"  Warning: Skipping {len(new_pmids) - len(valid_indices)} articles with empty content for embedding.")
            
            # Keep only data corresponding to valid documents
            new_pmids = [new_pmids[i] for i in valid_indices]
            documents = [documents[i] for i in valid_indices]
            if not new_pmids: # Check if any valid articles remain
                 print("No valid articles remain after filtering empty content.")
                 # We still need to return the existing IDs confirmed earlier
                 confirmed_available_ids = list(existing_ids)
                 final_available_pmids = [pmid for pmid in pmids_to_check if pmid in confirmed_available_ids]
                 return final_available_pmids

            # --- Prepare metadata (only for valid articles) ---
            metadatas = []
            for pmid in new_pmids: # Iterate through the filtered new_pmids
                # Access original data using the valid pmid
                article_data = new_articles_data[pmid] 
                meta = {
                    'title': article_data['title'],
                    'pub_date': article_data['pub_date'],
                    'journal': article_data['journal'],
                    'authors': ", ".join(article_data.get('authors', [])),
                    'mesh_headings': ", ".join(article_data.get('mesh_headings', []))
                }
                metadatas.append({k: v for k, v in meta.items() if v is not None})

            print(f"Generating embeddings for {len(new_pmids)} new articles...")
            new_embeddings = get_openai_embeddings(documents)

            if new_embeddings and len(new_embeddings) == len(new_pmids):
                print(f"Adding {len(new_pmids)} new articles to ChromaDB...")
                try:
                    collection.add(
                        embeddings=new_embeddings,
                        metadatas=metadatas,
                        documents=documents, # Store the combined text if title_abstract mode
                        ids=new_pmids
                    )
                    newly_added_ids.update(new_pmids)
                except Exception as e:
                    print(f"Error adding new documents to ChromaDB: {e}")
            else:
                print("Failed to generate embeddings for new articles. They won't be added.")
        else:
             print("Failed to fetch details for new PMIDs.")
    else:
        print("No new PMIDs to fetch.")

    # Return the list of PMIDs confirmed available (original + newly added)
    confirmed_available_ids = list(existing_ids.union(newly_added_ids))
    final_available_pmids = [pmid for pmid in pmids_to_check if pmid in confirmed_available_ids]

    print(f"{len(final_available_pmids)} PMIDs are now available in ChromaDB for ranking.")
    return final_available_pmids

# --- Remove old search_and_rank_chroma ---
# We replace it with ensure_articles_in_db and semantic_rank_articles

def semantic_rank_articles(query: str, pmids_to_rank: list[str], collection: chromadb.Collection, top_k: int = TOP_K) -> list[tuple[str, float]]:
    """Performs semantic search against specified PMIDs in ChromaDB using query.

    Uses collection.query, filtering by the provided PMIDs, and relies on
    ChromaDB's distance calculation.

    Args:
        query: The user's search query.
        pmids_to_rank: The list of PMIDs to search within (guaranteed to be in DB).
        collection: The ChromaDB collection object.
        top_k: The number of results to return.

    Returns:
        A list of tuples (pmid, similarity_score), sorted by score.
    """
    if not pmids_to_rank:
        print("No PMIDs provided for semantic ranking.")
        return []

    print("Generating embedding for the query...")
    query_embedding_list = get_openai_embeddings([query])
    if not query_embedding_list:
        print("Failed to generate embedding for the query. Cannot perform semantic search.")
        return []
    # query_embedding = query_embedding_list[0] # We pass the list to query

    # --- DEBUG --- 
    print(f"Debug: Type of query_embedding_list: {type(query_embedding_list)}")
    if isinstance(query_embedding_list, list) and len(query_embedding_list) > 0:
        print(f"Debug: Type of first element: {type(query_embedding_list[0])}")
        if isinstance(query_embedding_list[0], list):
            print(f"Debug: Length of first embedding vector: {len(query_embedding_list[0])}")
    # --- END DEBUG ---

    # The number of results to fetch depends on the context:
    # - If called for final display (top_k is small), limit results.
    # - If called for RRF (top_k is large, == len(pmids_to_rank)), fetch all.
    # We query slightly more than needed and filter client-side.
    num_results_to_request_query = min(len(pmids_to_rank), max(top_k * 2, 50))
    # However, if top_k *is* len(pmids_to_rank), we need to ensure we query at least that many
    if top_k == len(pmids_to_rank):
         num_results_to_request_query = max(num_results_to_request_query, top_k)

    # The number of results we ultimately return
    num_results_to_return = top_k 

    print(f"Querying ChromaDB (requesting up to {num_results_to_request_query}) within {len(pmids_to_rank)} PMIDs to return top {num_results_to_return}...")
    try:
        # Use collection.query, specifying the query embedding.
        # Query broadly and filter client-side.
        results = collection.query(
            query_embeddings=query_embedding_list, # Pass the list containing one embedding
            n_results=num_results_to_request_query, # Use the calculated limit
            include=['distances'] # Only need distances for ranking
        )

        # Filter results client-side to only include those from our target PMID list
        ranked_results = []
        if results and results.get('ids') and results.get('distances'):
            result_ids = results['ids'][0]
            result_distances = results['distances'][0]

            pmids_to_rank_set = set(pmids_to_rank) # For faster lookup

            for pmid, distance in zip(result_ids, result_distances):
                if pmid in pmids_to_rank_set:
                    # Cosine distance: 0 is identical, higher is less similar. Convert to similarity.
                    similarity_score = 1.0 - distance
                    ranked_results.append((pmid, similarity_score))

            # Sort the filtered results by score descending
            ranked_results.sort(key=lambda x: x[1], reverse=True)

            # Return the top_k (num_results_to_return) from the filtered and sorted list
            return ranked_results[:num_results_to_return]
        else:
            print("Warning: ChromaDB query returned unexpected or empty results.")
            return []

    except Exception as e:
        print(f"Error querying ChromaDB: {e}")
        return []

def hybrid_rank_rrf(pubmed_ranked_pmids: list[str], semantic_ranked_results: list[tuple[str, float]], k: int = RRF_K) -> list[tuple[str, float]]:
    """
    Combines PubMed relevance rank and semantic similarity rank using Reciprocal Rank Fusion (RRF).

    Args:
        pubmed_ranked_pmids: List of PMIDs sorted by PubMed relevance.
        semantic_ranked_results: List of (pmid, similarity_score) sorted by semantic similarity.
        k: RRF weighting constant (default 60).

    Returns:
        List of (pmid, rrf_score) sorted by RRF score descending.
    """
    print(f"Performing Reciprocal Rank Fusion (k={k})...")
    rrf_scores = {}

    # Create rank dictionaries for fast lookup
    pubmed_ranks = {pmid: rank + 1 for rank, pmid in enumerate(pubmed_ranked_pmids)}
    # Create semantic ranks dict from the semantic results
    semantic_ranks = {pmid: rank + 1 for rank, (pmid, _) in enumerate(semantic_ranked_results)}

    # Get the union of all PMIDs that appear in either ranking
    all_pmids = set(pubmed_ranked_pmids) | set(p[0] for p in semantic_ranked_results)

    for pmid in all_pmids:
        score = 0.0
        # Add score based on PubMed rank
        rank_pubmed = pubmed_ranks.get(pmid)
        if rank_pubmed:
            score += 1.0 / (k + rank_pubmed)

        # Add score based on semantic rank
        rank_semantic = semantic_ranks.get(pmid)
        if rank_semantic:
            score += 1.0 / (k + rank_semantic)

        # Only include PMIDs that have at least one score
        if score > 0:
            rrf_scores[pmid] = score

    # Sort PMIDs by RRF score descending
    sorted_results = sorted(rrf_scores.items(), key=lambda item: item[1], reverse=True)

    print(f"RRF calculation complete for {len(sorted_results)} articles.")
    return sorted_results

def main():
    """Main function allowing selection of ranking modes and pre-ranking date filter."""
    user_query = input("Enter your medical query: ")
    if not user_query:
        print("Query cannot be empty.")
        return

    # Get user choice for ranking mode
    while True:
        print("\nSelect Ranking Mode:")
        print("  1: PubMed Relevance (Keyword/MeSH search order)")
        print("  2: Semantic Similarity (Vector search ranking)")
        print("  3: Hybrid (Reciprocal Rank Fusion of 1 & 2)")
        mode_choice = input("Enter choice (1, 2, or 3): ")
        if mode_choice == '1':
            ranking_mode = "pubmed"
            break
        elif mode_choice == '2':
            ranking_mode = "semantic"
            break
        elif mode_choice == '3':
            ranking_mode = "hybrid"
            break
        else:
            print("Invalid choice. Please enter 1, 2, or 3.")

    print(f"\n--- Starting search with mode: {ranking_mode} ---")

    # 0. Generate MeSH terms
    generated_mesh_terms = generate_mesh_terms(user_query, client)

    # 1. Initial Keyword Search (Get PubMed relevance-ranked PMIDs)
    initial_pmids_ordered = search_pubmed_keywords(user_query, mesh_terms=generated_mesh_terms)
    if not initial_pmids_ordered:
        print("No results found from initial PubMed keyword search.")
        return

    # 2. Ensure Articles are in DB (and get the subset available)
    pmids_in_db = ensure_articles_in_db(initial_pmids_ordered, collection)
    if not pmids_in_db:
        print("Could not ensure any articles were available in the vector database.")
        print("\n--- Initial Keyword Search PMIDs (Unable to process further) --- ")
        for i, pmid in enumerate(initial_pmids_ordered[:TOP_K]):
            print(f"{i+1}. PMID: {pmid}")
        return

    # --- Optional Pre-Ranking Date Filter ---
    eligible_pmids_for_ranking = pmids_in_db # Default to all available PMIDs
    apply_date_filter = input("Apply a date filter BEFORE ranking (e.g., only rank articles from a certain year onwards)? (y/n): ").lower()
    
    if apply_date_filter == 'y':
        while True:
            try:
                min_year_str = input(f"Enter the minimum publication year for ranking (e.g., {datetime.datetime.now().year - 5}): ")
                min_year = int(min_year_str)
                if 1900 < min_year <= datetime.datetime.now().year:
                    break
                else:
                    print("Please enter a reasonable year.")
            except ValueError:
                print("Invalid input. Please enter a year as a number.")
        
        print(f"Fetching metadata for date filtering {len(pmids_in_db)} articles...")
        metadata_for_filtering = {}
        try:
            filter_data = collection.get(ids=pmids_in_db, include=['metadatas'])
            if filter_data and filter_data.get('ids'):
                 for i, pmid in enumerate(filter_data['ids']):
                     meta = filter_data['metadatas'][i] if i < len(filter_data.get('metadatas', [])) else {}
                     metadata_for_filtering[pmid] = meta
        except Exception as e:
            print(f"Warning: Could not retrieve metadata for date filtering: {e}. Proceeding without date filter.")
            # Keep eligible_pmids_for_ranking as pmids_in_db if metadata fetch fails
        
        if metadata_for_filtering: # Proceed only if metadata was fetched
             print(f"Applying date filter: Keeping articles from year {min_year} onwards for ranking...")
             temp_eligible_pmids = []
             for pmid in pmids_in_db:
                 meta = metadata_for_filtering.get(pmid)
                 keep_article = True # Default to keeping if date is missing/invalid
                 if meta:
                     pub_date_str = meta.get('pub_date') # Expecting YYYY-MM-DD
                     if pub_date_str:
                         try:
                             article_year = int(pub_date_str.split('-')[0])
                             if article_year < min_year:
                                 keep_article = False
                         except (IndexError, ValueError):
                              pass # Keep if date format is invalid
                 if keep_article:
                     temp_eligible_pmids.append(pmid)
             
             eligible_pmids_for_ranking = temp_eligible_pmids
             print(f"Date filter applied. {len(eligible_pmids_for_ranking)} articles eligible for ranking.")

    if not eligible_pmids_for_ranking:
         print("No articles match the date filter criteria.")
         return

    # Filter the original ordered list to only those eligible for ranking
    pubmed_ranked_eligible_pmids = [pmid for pmid in initial_pmids_ordered if pmid in eligible_pmids_for_ranking]

    # 3. Perform Ranking based on selected mode using the eligible PMIDs
    final_ranked_results = [] # This will store list of (pmid, score) tuples

    if ranking_mode == "pubmed":
        print(f"Using PubMed relevance ranking on {len(pubmed_ranked_eligible_pmids)} eligible articles...")
        # Use the filtered, ordered list; score is the rank (1-based)
        final_ranked_results = [(pmid, i+1) for i, pmid in enumerate(pubmed_ranked_eligible_pmids[:TOP_K])]
        print(f"Displaying top {len(final_ranked_results)} results based on PubMed order.")

    elif ranking_mode == "semantic":
        print(f"Performing semantic ranking on {len(eligible_pmids_for_ranking)} eligible articles...")
        # Pass the eligible list to the semantic rank function
        semantic_results = semantic_rank_articles(user_query, eligible_pmids_for_ranking, collection, TOP_K)
        final_ranked_results = semantic_results # Already (pmid, similarity_score)
        print(f"Displaying top {len(final_ranked_results)} results based on semantic similarity.")

    elif ranking_mode == "hybrid":
        print(f"Performing hybrid ranking (RRF) on {len(eligible_pmids_for_ranking)} eligible articles...")
        # A. Get semantic scores for ALL eligible articles for RRF input
        print(f"  Getting semantic scores for {len(eligible_pmids_for_ranking)} eligible articles...")
        all_semantic_results = semantic_rank_articles(user_query, eligible_pmids_for_ranking, collection, len(eligible_pmids_for_ranking))

        if not all_semantic_results:
             print("  Warning: Failed to get semantic scores for RRF. Falling back to PubMed ranking.")
             # Fallback to PubMed ranking using the eligible list
             final_ranked_results = [(pmid, i+1) for i, pmid in enumerate(pubmed_ranked_eligible_pmids[:TOP_K])]
        else:
             # B. Perform RRF using the PubMed ordered eligible list and all semantic results
             rrf_results = hybrid_rank_rrf(pubmed_ranked_eligible_pmids, all_semantic_results)
             # C. Take Top K from RRF results
             final_ranked_results = rrf_results[:TOP_K] # RRF returns (pmid, rrf_score)
             print(f"Displaying top {len(final_ranked_results)} results based on hybrid RRF score.")

    # 4. Display Results (using the final ranked list)
    print(f"\n--- Top {len(final_ranked_results)} Results ({ranking_mode.capitalize()} Ranking) --- ")
    if not final_ranked_results:
        print("No articles found after ranking.")
        # Fallback logic already handled if pmids_in_db is empty or ranking fails
    else:
        # Fetch full metadata for the final ranked results for display
        top_pmids = [pmid for pmid, score in final_ranked_results]
        display_data = {}
        if top_pmids:
             try:
                 top_articles_data = collection.get(
                     ids=top_pmids,
                     include=['documents', 'metadatas']
                 )
                 if top_articles_data and top_articles_data.get('ids'):
                     retrieved_metadatas = top_articles_data.get('metadatas', [])
                     retrieved_documents = top_articles_data.get('documents', [])
                     for i, pmid_retrieved in enumerate(top_articles_data['ids']):
                          metadata = retrieved_metadatas[i] if i < len(retrieved_metadatas) else {}
                          document = retrieved_documents[i] if i < len(retrieved_documents) else ''
                          display_data[pmid_retrieved] = {'metadata': metadata, 'document': document}

                 if len(display_data) != len(top_pmids):
                     print(f"Warning: Could not retrieve metadata for all {len(top_pmids)} ranked PMIDs. Retrieved {len(display_data)}.")

             except Exception as e:
                 print(f"Warning: Could not retrieve full metadata for display: {e}")

        # Display loop
        for i, (pmid, score) in enumerate(final_ranked_results):
            article_info = display_data.get(pmid)
            title, pub_date, journal, authors_str, mesh_str, abstract_snippet = ('Data unavailable',) * 6
            abstract = ''

            if article_info:
                 meta = article_info.get('metadata', {})
                 abstract = article_info.get('document', '')
                 title = meta.get('title', 'Title unavailable')
                 pub_date = meta.get('pub_date', 'Date unavailable')
                 journal = meta.get('journal', 'Journal unavailable')
                 authors_str = meta.get('authors', '')
                 mesh_str = meta.get('mesh_headings', '')
                 abstract_snippet = abstract[:250] if abstract else ''
            else:
                 print(f"  (Metadata for PMID {pmid} could not be retrieved or was missing)")

            authors = authors_str.split(', ') if authors_str else ['Authors unavailable']
            mesh = mesh_str.split(', ') if mesh_str else ['MeSH unavailable']

            score_display = ""
            if ranking_mode == "semantic":
                score_display = f"Score: {score:.4f}"
            elif ranking_mode == "hybrid":
                 score_display = f"RRF Score: {score:.4f}"
            elif ranking_mode == "pubmed":
                 score_display = f"Rank: {score}"

            print(f"{i+1}. PMID: {pmid} ({score_display}, Date: {pub_date})")
            print(f"   Title: {title}")
            print(f"   Journal: {journal}")
            if authors and authors[0]: print(f"   Authors: {(', '.join(authors[:3])) + (' et al.' if len(authors) > 3 else '')}")
            if abstract_snippet:
                print(f"   Abstract: {abstract_snippet}...")
            if mesh and mesh[0]: print(f"   MeSH: {(', '.join(mesh[:5])) + ('...' if len(mesh) > 5 else '')}")
            print()

if __name__ == "__main__":
    main() 