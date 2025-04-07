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
    """Performs a keyword search on PubMed, optionally refined by MeSH terms."""
    
    # Construct the final PubMed query string
    final_query = query # Start with the original query
    
    if mesh_terms:
        # Add the MeSH terms using OR within the group, and AND with the original query
        # Format: (original query) AND (term1[MeSH Terms] OR term2[MeSH Terms] ...)
        mesh_query_part = " OR ".join([f"{term}[MeSH Terms]" for term in mesh_terms])
        if mesh_query_part:
             final_query = f"({query}) AND ({mesh_query_part})"
        print(f"Refining keyword search with MeSH terms.")
    else:
        print("Performing keyword search without MeSH term refinement.")

    print(f"Executing PubMed query: {final_query}")
    
    try:
        handle = Entrez.esearch(db="pubmed", term=final_query, retmax=str(max_results), sort="relevance")
        record = Entrez.read(handle)
        handle.close()
        pmids = record.get("IdList", [])
        print(f"Found {len(pmids)} potential PMIDs using the refined query.")
        return pmids
    except Exception as e:
        print(f"Error during PubMed keyword search: {e}")
        # Fallback: If the complex query fails, maybe try the original query?
        if mesh_terms: # Only fallback if we were using mesh terms
             print(f"Refined query failed. Falling back to original query: {query}")
             try:
                  handle = Entrez.esearch(db="pubmed", term=query, retmax=str(max_results), sort="relevance")
                  record = Entrez.read(handle)
                  handle.close()
                  pmids = record.get("IdList", [])
                  print(f"Found {len(pmids)} potential PMIDs using fallback query.")
                  return pmids
             except Exception as e2:
                  print(f"Error during fallback PubMed keyword search: {e2}")
                  return []
        else:
            return [] # Original query already failed

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
            documents = [new_articles_data[pmid]['abstract'] for pmid in new_pmids]
            # Prepare metadata, ensuring complex fields are serializable (e.g., lists to strings)
            metadatas = []
            for pmid in new_pmids:
                meta = {
                    'title': new_articles_data[pmid]['title'],
                    'pub_date': new_articles_data[pmid]['pub_date'],
                    'journal': new_articles_data[pmid]['journal'],
                    # Convert lists to comma-separated strings for Chroma metadata
                    'authors': ", ".join(new_articles_data[pmid].get('authors', [])),
                    'mesh_headings': ", ".join(new_articles_data[pmid].get('mesh_headings', []))
                }
                 # Filter out None values before adding
                metadatas.append({k: v for k, v in meta.items() if v is not None})


            print(f"Generating embeddings for {len(new_pmids)} new articles...")
            new_embeddings = get_openai_embeddings(documents)

            if new_embeddings and len(new_embeddings) == len(new_pmids):
                print(f"Adding {len(new_pmids)} new articles to ChromaDB...")
                try:
                    collection.add(
                        embeddings=new_embeddings,
                        metadatas=metadatas,
                        documents=documents,
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

    # Return the list of PMIDs from the original check list that are now confirmed available
    confirmed_available_ids = list(existing_ids.union(newly_added_ids))
    # Filter this list to only include PMIDs that were in the original pmid_to_check list
    # This handles cases where fetching might fail for some requested IDs
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

    # Ensure n_results doesn't exceed the number of PMIDs we are ranking
    num_results_to_request = min(top_k, len(pmids_to_rank))

    print(f"Querying ChromaDB within {len(pmids_to_rank)} PMIDs for top {num_results_to_request} results...")
    try:
        # Use collection.query, specifying the query embedding.
        # Query broadly and filter client-side.
        results = collection.query(
            query_embeddings=query_embedding_list, # Pass the list containing one embedding
            # Removed redundant n_results=num_results_to_request
            # Removed where clause as we filter based on pmids_to_rank set later
            # Query slightly more than top_k globally to increase chances of getting relevant PMIDs.
            n_results=min(len(pmids_to_rank), max(top_k * 2, 50)),
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

            # Return the top_k from the filtered and sorted list
            return ranked_results[:top_k]
        else:
            print("Warning: ChromaDB query returned unexpected or empty results.")
            return []

    except Exception as e:
        print(f"Error querying ChromaDB: {e}")
        return []

def main():
    """Main function to run the enhanced hybrid search with persistence and MeSH term generation."""
    user_query = input("Enter your medical query: ")

    if not user_query:
        print("Query cannot be empty.")
        return
        
    # 0. Generate MeSH terms from the query using LLM
    generated_mesh_terms = generate_mesh_terms(user_query, client) # Pass the initialized OpenAI client

    # 1. Initial Keyword Search (potentially refined with MeSH terms)
    initial_pmids = search_pubmed_keywords(user_query, mesh_terms=generated_mesh_terms)
    if not initial_pmids:
        print("No results found from initial PubMed keyword search (with/without MeSH refinement).")
        return

    # 2. Ensure Articles are in Persistent DB (Fetch/Embed/Add if needed)
    pmids_in_db = ensure_articles_in_db(initial_pmids, collection)

    if not pmids_in_db:
        print("Could not ensure any articles were available in the vector database.")
        print("\n--- Initial Keyword Search PMIDs --- ")
        for i, pmid in enumerate(initial_pmids[:TOP_K]):
             print(f"{i+1}. PMID: {pmid}")
        return

    # 3. Semantic Ranking within the available PMIDs
    ranked_results = semantic_rank_articles(user_query, pmids_in_db, collection, TOP_K)

    # 4. Display Results
    print(f"\n--- Top {len(ranked_results)} Semantic Search Results --- ")
    if not ranked_results:
        print("No articles found after semantic ranking.")
        # Fallback: Show top PMIDs from keyword search that are in the DB but weren't ranked
        print("\n--- Available Articles (from Keyword Search, Unranked) ---")
        try:
            fallback_data = collection.get(ids=pmids_in_db[:TOP_K], include=['metadatas'])
            if fallback_data and fallback_data.get('ids'):
                 for i, pmid in enumerate(fallback_data['ids']):
                     meta = fallback_data['metadatas'][i] if fallback_data.get('metadatas') else {}
                     title = meta.get('title', 'Title unavailable')
                     pub_date = meta.get('pub_date', 'Date unavailable')
                     print(f"{i+1}. PMID: {pmid} (Date: {pub_date}) - {title}")
            else:
                print("(Could not retrieve fallback metadata)")
        except Exception as fallback_e:
             print(f"(Error retrieving fallback metadata: {fallback_e})")

    else:
        # Fetch full metadata for the top ranked results for display
        top_pmids = [pmid for pmid, score in ranked_results]
        try:
            top_articles_data = collection.get(
                ids=top_pmids,
                include=['documents', 'metadatas'] # Get abstract (document) and other metadata
            )
            # Create a lookup dict for easier access
            display_data = {pmid: {'metadata': top_articles_data['metadatas'][i],
                                    'document': top_articles_data['documents'][i]} 
                            for i, pmid in enumerate(top_articles_data['ids'])}

        except Exception as e:
            print(f"Warning: Could not retrieve full metadata for display: {e}")
            display_data = {}

        # Display ranked results with enriched data
        for i, (pmid, score) in enumerate(ranked_results):
            article_info = display_data.get(pmid)
            if article_info:
                meta = article_info.get('metadata', {})
                abstract = article_info.get('document', '')
                title = meta.get('title', 'Title unavailable')
                pub_date = meta.get('pub_date', 'Date unavailable')
                journal = meta.get('journal', 'Journal unavailable')
                authors = meta.get('authors', 'Authors unavailable').split(', ') # Split back from string
                mesh = meta.get('mesh_headings', 'MeSH unavailable').split(', ') # Split back from string
                abstract_snippet = abstract[:250] if abstract else ''
            else:
                # Fallback if metadata fetch failed for this specific PMID
                title, pub_date, journal, authors, mesh, abstract_snippet = ('Data unavailable',) * 6

            print(f"{i+1}. PMID: {pmid} (Score: {score:.4f}, Date: {pub_date})")
            print(f"   Title: {title}")
            print(f"   Journal: {journal}")
            if authors and authors[0]: print(f"   Authors: {(', '.join(authors[:3])) + (' et al.' if len(authors) > 3 else '')}")
            if abstract_snippet:
                print(f"   Abstract: {abstract_snippet}...")
            if mesh and mesh[0]: print(f"   MeSH: {(', '.join(mesh[:5])) + ('...' if len(mesh) > 5 else '')}")
            print()

if __name__ == "__main__":
    main() 