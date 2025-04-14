import requests
from Bio import Entrez
import chromadb
from openai import OpenAI
import os
import time
import datetime
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, HttpUrl

# --- Pydantic Models (Keep definitions here for now, or move to a dedicated models.py) ---

class Author(BaseModel):
    lastName: Optional[str] = None
    foreName: Optional[str] = None
    initials: Optional[str] = None
    affiliation: Optional[str] = None

    def display_name(self) -> str:
        parts = [self.foreName, self.lastName]
        name = " ".join(p for p in parts if p)
        return name if name else self.initials or "Unknown Author"

class PubMedArticle(BaseModel):
    pmid: str
    doi: Optional[str] = None
    title: Optional[str] = None
    abstract: Optional[str] = None
    authors: List[Author] = []
    journalTitle: Optional[str] = None
    journalISOAbbreviation: Optional[str] = None
    volume: Optional[str] = None
    issue: Optional[str] = None
    pagination: Optional[str] = None
    pubDate: Optional[str] = None
    publicationTypes: List[str] = []
    meshHeadings: List[str] = []
    keywords: List[str] = []
    language: Optional[str] = None

    @property
    def pubmed_url(self) -> str: # Keep as string
        return f"https://pubmed.ncbi.nlm.nih.gov/{self.pmid}/"

# --- Configuration Constants (moved from main script, might move to a config module later) ---
# Defaults, can be overridden by API request
DEFAULT_MAX_KEYWORD_RESULTS = 100
DEFAULT_TOP_K = 10
DEFAULT_LLM_MESH_MODEL = "gpt-4o"
DEFAULT_EMBED_CONTENT_MODE = "title_abstract"
RRF_K = 60

# --- Core Functions ---

def generate_mesh_terms(query: str, llm_client: OpenAI, model: str = DEFAULT_LLM_MESH_MODEL) -> list[str]:
    """Uses an LLM to generate relevant MeSH terms from a user query."""
    print(f"Generating MeSH terms for query: '{query}' using {model}...")
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
            temperature=0.2, max_tokens=100 
        )
        raw_terms = response.choices[0].message.content.strip()
        print(f"  LLM Raw Output: {raw_terms}")
        mesh_terms = [term.strip() for term in raw_terms.split(',') if term.strip() and len(term) > 2]
        print(f"  Generated MeSH terms: {mesh_terms}")
        return mesh_terms
    except Exception as e:
        print(f"Error generating MeSH terms: {e}")
        return []

def search_pubmed_keywords(query: str, mesh_terms: list[str] | None, max_results: int, entrez_email: str) -> tuple[list[str], str]:
    """Performs a keyword search, combining query text and MeSH Major Topics with fallbacks."""
    Entrez.email = entrez_email
    pmids = []
    final_query_used = query # Default to original query
    min_results_threshold = 20 # Threshold to trigger fallback

    # --- Attempt 1: Query Text AND MeSH Major Topics ---
    if mesh_terms:
        major_topic_query_part = " OR ".join([f'\"{term}\"[MeSH Major Topic]' for term in mesh_terms])
        if major_topic_query_part:
            refined_query = f"({query}) AND ({major_topic_query_part})"
            print(f"Executing PubMed query (Attempt 1: Text AND MeSH Major Topics): {refined_query}")
            try:
                handle = Entrez.esearch(db="pubmed", term=refined_query, retmax=str(max_results), sort="relevance")
                record = Entrez.read(handle)
                handle.close()
                pmids = record.get("IdList", [])
                final_query_used = refined_query
                print(f"Found {len(pmids)} potential PMIDs.")
                # If enough results, return them
                if len(pmids) >= min_results_threshold:
                    return pmids, final_query_used
                else:
                    print(f"Attempt 1 yielded only {len(pmids)} results (threshold: {min_results_threshold}). Trying broader MeSH Terms...")
            except Exception as e:
                print(f"Error during PubMed search (Attempt 1): {e}. Trying broader MeSH Terms...")
                # Proceed to Attempt 2
        else:
             print("Warning: MeSH terms provided but resulted in an empty Major Topic query part. Skipping Attempt 1.")

    # --- Attempt 2: Query Text AND MeSH Terms (Broader) ---
    # Triggered if Attempt 1 failed, yielded too few results, or MeSH terms existed but query part was empty
    if mesh_terms and (not pmids or len(pmids) < min_results_threshold): # Check if we need to run this attempt
        terms_query_part = " OR ".join([f'\"{term}\"[MeSH Terms]' for term in mesh_terms])
        if terms_query_part:
            refined_query_broad = f"({query}) AND ({terms_query_part})"
            # Avoid re-running if it's identical to a failed previous query (unlikely here)
            if refined_query_broad != final_query_used: 
                print(f"Executing PubMed query (Attempt 2: Text AND MeSH Terms): {refined_query_broad}")
                try:
                    handle = Entrez.esearch(db="pubmed", term=refined_query_broad, retmax=str(max_results), sort="relevance")
                    record = Entrez.read(handle)
                    handle.close()
                    pmids = record.get("IdList", [])
                    final_query_used = refined_query_broad
                    print(f"Found {len(pmids)} potential PMIDs.")
                    # If enough results now, return them
                    if len(pmids) >= min_results_threshold:
                        return pmids, final_query_used
                    else:
                        print(f"Attempt 2 yielded only {len(pmids)} results. Falling back to original query text...")
                except Exception as e:
                    print(f"Error during PubMed search (Attempt 2): {e}. Falling back to original query text...")
                    # Proceed to Attempt 3
            else:
                 print("Broad MeSH query is same as a previous attempt. Falling back to original query text...")
        else:
            print("Warning: MeSH terms provided but resulted in an empty MeSH Terms query part. Falling back to original query text.")

    # --- Attempt 3: Original Query Text Only ---
    # Triggered if MeSH terms weren't generated, or Attempts 1 & 2 failed or yielded too few results
    if not pmids or len(pmids) < min_results_threshold: # Check if we need to run this fallback
        # Avoid re-running if original query was already tried and failed
        if final_query_used != query or not pmids: # Check if original query differs or pmids is empty
            print(f"Executing PubMed query (Attempt 3: Original Text Only): {query}")
            try:
                handle = Entrez.esearch(db="pubmed", term=query, retmax=str(max_results), sort="relevance")
                record = Entrez.read(handle)
                handle.close()
                pmids = record.get("IdList", [])
                final_query_used = query
                print(f"Found {len(pmids)} potential PMIDs using only original query text.")
            except Exception as e:
                print(f"Error during PubMed search (Attempt 3 - Original Query): {e}")
                pmids = [] # Ensure empty on final failure
        else:
             print("Original query already attempted or results were insufficient. Returning current results.")
             
    # Return whatever PMIDs we have at the end, and the last query tried
    return pmids, final_query_used

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

def fetch_abstracts(pmids: list[str], entrez_email: str) -> Dict[str, PubMedArticle]:
    """Fetches full article details for PMIDs, returns Pydantic models."""
    Entrez.email = entrez_email # Set email for this call
    print(f"Fetching details for {len(pmids)} PMIDs...")
    articles: Dict[str, PubMedArticle] = {}
    batch_size = 100
    for i in range(0, len(pmids), batch_size):
        batch_pmids = pmids[i:i + batch_size]
        print(f"  Fetching batch {i//batch_size + 1} ({len(batch_pmids)} PMIDs)...")
        try:
            handle = Entrez.efetch(db="pubmed", id=batch_pmids, rettype="medline", retmode="xml")
            records = Entrez.read(handle)
            handle.close()

            for record in records.get('PubmedArticle', []):
                try:
                    medline_citation = record.get('MedlineCitation', {})
                    if not medline_citation:
                        continue
                    
                    article_info = medline_citation.get('Article', {})
                    journal_info = article_info.get('Journal', {})
                    journal_issue = journal_info.get('JournalIssue', {})
                    
                    pmid = str(medline_citation.get('PMID'))
                    
                    # --- Basic Info --- 
                    title = article_info.get('ArticleTitle', None)
                    language = article_info.get('Language', [None])[0] # Takes first language
                    pagination = article_info.get('Pagination', {}).get('MedlinePgn', None)
                    
                    # --- Abstract --- 
                    abstract = None
                    abstract_list = article_info.get('Abstract', {}).get('AbstractText', [])
                    if isinstance(abstract_list, list):
                        abstract = " ".join(str(a) for a in abstract_list) # Ensure all parts are strings
                    elif abstract_list: # Handle case where it might be a single string directly
                        abstract = str(abstract_list)
                    if not abstract:
                        other_abstract_list = article_info.get('OtherAbstract', [])
                        if other_abstract_list and isinstance(other_abstract_list, list) and len(other_abstract_list) > 0:
                            abstract_texts = other_abstract_list[0].get('AbstractText', [])
                            if isinstance(abstract_texts, list):
                                 abstract = " ".join(str(a) for a in abstract_texts)
                            elif abstract_texts:
                                 abstract = str(abstract_texts)
                    
                    # Skip article if no abstract found (required for embedding)
                    if not abstract:
                        # print(f"  Warning: No abstract found for PMID: {pmid}. Skipping storage.") # Keep this internal maybe?
                        continue

                    # --- Date --- 
                    pub_date_info = journal_issue.get('PubDate', {})
                    pub_date = parse_pubmed_date(pub_date_info)
                    
                    # --- Journal --- 
                    journal_title = journal_info.get('Title', None)
                    journal_iso = journal_info.get('ISOAbbreviation', None)
                    volume = journal_issue.get('Volume', None)
                    issue = journal_issue.get('Issue', None)
                    
                    # --- Authors --- 
                    authors_data = article_info.get('AuthorList', [])
                    authors = []
                    if isinstance(authors_data, list):
                        for author_info in authors_data:
                            affiliation = None
                            affil_list = author_info.get('AffiliationInfo', [])
                            if affil_list and isinstance(affil_list, list):
                                affiliation = affil_list[0].get('Affiliation', None)
                                
                            authors.append(Author(
                                lastName=author_info.get('LastName'),
                                foreName=author_info.get('ForeName'),
                                initials=author_info.get('Initials'),
                                affiliation=affiliation
                            ))
                            
                    # --- Publication Types --- 
                    pub_types_list = article_info.get('PublicationTypeList', [])
                    pub_types = [str(pt) for pt in pub_types_list if pt]
                    
                    # --- MeSH Headings --- 
                    mesh_list = medline_citation.get('MeshHeadingList', [])
                    mesh_headings = []
                    if isinstance(mesh_list, list):
                        for mesh_item in mesh_list:
                            descriptor = mesh_item.get('DescriptorName')
                            if descriptor:
                                mesh_headings.append(str(descriptor))
                                
                    # --- Keywords --- 
                    keyword_list_data = medline_citation.get('KeywordList', [])
                    keywords = []
                    if isinstance(keyword_list_data, list) and len(keyword_list_data) > 0:
                         actual_keywords = keyword_list_data[0] 
                         if isinstance(actual_keywords, list):
                             keywords = [str(kw).strip() for kw in actual_keywords if str(kw).strip()]

                    # --- DOI --- 
                    doi = None
                    article_ids = record.get('PubmedData', {}).get('ArticleIdList', [])
                    if isinstance(article_ids, list):
                         for article_id in article_ids:
                              if hasattr(article_id, 'attributes') and article_id.attributes.get('IdType') == 'doi':
                                   doi = str(article_id)
                                   break

                    # --- Create Pydantic Model Instance --- 
                    article_model = PubMedArticle(
                        pmid=pmid,
                        doi=doi,
                        title=title,
                        abstract=abstract,
                        authors=authors,
                        journalTitle=journal_title,
                        journalISOAbbreviation=journal_iso,
                        volume=volume,
                        issue=issue,
                        pagination=pagination,
                        pubDate=pub_date,
                        publicationTypes=pub_types,
                        meshHeadings=mesh_headings,
                        keywords=keywords,
                        language=language
                    )
                    articles[pmid] = article_model
                
                except Exception as parse_e:
                     pmid_err = record.get('MedlineCitation', {}).get('PMID', 'UNKNOWN')
                     print(f"  Error parsing record for PMID {pmid_err}: {parse_e}")
                     continue

            time.sleep(0.34)

        except Exception as e:
            print(f"Error fetching or processing batch starting at index {i}: {e}")

    print(f"Successfully processed details for {len(articles)} articles with abstracts.")
    return articles

def get_openai_embeddings(texts: list[str], openai_client: OpenAI, model: str) -> list[list[float]]:
    """Generates embeddings using OpenAI API."""
    if not texts:
        return []
    try:
        texts = [text.replace("\n", " ") for text in texts]
        response = openai_client.embeddings.create(input=texts, model=model)
        return [embedding.embedding for embedding in response.data]
    except Exception as e:
        print(f"Error getting OpenAI embeddings: {e}")
        return []

def ensure_articles_in_db(pmids_to_check: list[str], collection: chromadb.Collection, openai_client: OpenAI, embed_model: str, embed_content_mode: str, entrez_email: str) -> list[str]:
    """Checks ChromaDB, fetches missing, embeds, and adds them."""
    if not pmids_to_check:
        return []

    print(f"Checking persistence for {len(pmids_to_check)} PMIDs...")
    try:
        existing_in_db = collection.get(ids=pmids_to_check, include=[]) # Only need IDs
        existing_ids = set(existing_in_db['ids'])
        print(f"Found {len(existing_ids)} PMIDs already in the persistent DB.")
    except Exception as e:
        print(f"Error checking existing documents in ChromaDB: {e}")
        existing_ids = set()

    pmids_to_fetch = [pmid for pmid in pmids_to_check if pmid not in existing_ids]
    newly_added_ids = set()

    if pmids_to_fetch:
        print(f"Fetching details for {len(pmids_to_fetch)} new PMIDs...")
        # Pass Entrez email
        new_articles_data = fetch_abstracts(pmids_to_fetch, entrez_email)

        if new_articles_data:
            new_pmids = list(new_articles_data.keys())
            documents = []
            print(f"Preparing documents for embedding using mode: {embed_content_mode}")
            for pmid in new_pmids:
                article_model = new_articles_data[pmid]
                title = article_model.title or ''
                abstract = article_model.abstract or ''
                if embed_content_mode == "title_abstract":
                    doc_text = f"{title}. {abstract}".strip()
                    documents.append(doc_text if doc_text != "." else "")
                else:
                    documents.append(abstract if abstract else "")
            
            valid_indices = [i for i, doc in enumerate(documents) if doc]
            if len(valid_indices) < len(new_pmids):
                print(f"  Warning: Skipping {len(new_pmids) - len(valid_indices)} articles with empty content for embedding.")
            
            new_pmids = [new_pmids[i] for i in valid_indices]
            documents = [documents[i] for i in valid_indices]
            
            if new_pmids: # Proceed only if we have valid articles
                metadatas = []
                for pmid in new_pmids:
                    article_model: PubMedArticle = new_articles_data[pmid]
                    meta = {
                        'title': article_model.title,
                        'pub_date': article_model.pubDate,
                        'journal': article_model.journalTitle,
                        'authors': ", ".join(author.display_name() for author in article_model.authors),
                        'mesh_headings': ", ".join(article_model.meshHeadings),
                        'keywords': ", ".join(article_model.keywords),
                        'publication_types': ", ".join(article_model.publicationTypes),
                        'language': article_model.language,
                        'doi': article_model.doi
                    }
                    metadatas.append({k: v for k, v in meta.items() if v is not None})

                print(f"Generating embeddings for {len(new_pmids)} new articles...")
                # Pass openai client and model
                new_embeddings = get_openai_embeddings(documents, openai_client, embed_model)

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
            # else clause for 'if not new_pmids' already handled returning existing earlier
        else:
             print("Failed to fetch details for new PMIDs.")
    else:
        print("No new PMIDs to fetch.")

    confirmed_available_ids = list(existing_ids.union(newly_added_ids))
    final_available_pmids = [pmid for pmid in pmids_to_check if pmid in confirmed_available_ids]
    print(f"{len(final_available_pmids)} PMIDs are now available in ChromaDB for ranking.")
    return final_available_pmids

def semantic_rank_articles(query: str, pmids_to_rank: list[str], collection: chromadb.Collection, top_k: int, openai_client: OpenAI, embed_model: str) -> list[tuple[str, float]]:
    """Performs semantic search against specified PMIDs."""
    if not pmids_to_rank:
        print("No PMIDs provided for semantic ranking.")
        return []

    print("Generating embedding for the query...")
    # Pass client and model
    query_embedding_list = get_openai_embeddings([query], openai_client, embed_model)
    if not query_embedding_list:
        print("Failed to generate embedding for query.")
        return []
    
    # Debugging removed for brevity

    num_results_to_request_query = min(len(pmids_to_rank), max(top_k * 2, 50))
    if top_k == len(pmids_to_rank):
         num_results_to_request_query = max(num_results_to_request_query, top_k)
    num_results_to_return = top_k 

    print(f"Querying ChromaDB (requesting up to {num_results_to_request_query}) to return top {num_results_to_return}...")
    try:
        results = collection.query(
            query_embeddings=query_embedding_list,
            n_results=num_results_to_request_query,
            include=['distances']
        )

        ranked_results = []
        if results and results.get('ids') and results.get('distances'):
            result_ids = results['ids'][0]
            result_distances = results['distances'][0]
            pmids_to_rank_set = set(pmids_to_rank)
            for pmid, distance in zip(result_ids, result_distances):
                if pmid in pmids_to_rank_set:
                    similarity_score = 1.0 - distance
                    ranked_results.append((pmid, similarity_score))
            ranked_results.sort(key=lambda x: x[1], reverse=True)
            return ranked_results[:num_results_to_return]
        else:
            print("Warning: ChromaDB query returned unexpected results.")
            return []
    except Exception as e:
        print(f"Error querying ChromaDB: {e}")
        return []

def hybrid_rank_rrf(pubmed_ranked_pmids: list[str], semantic_ranked_results: list[tuple[str, float]], k: int = RRF_K) -> list[tuple[str, float]]:
    """Combines ranks using Reciprocal Rank Fusion (RRF)."""
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

# --- Remove Main Function --- 
# The main execution logic will now be in api.py 