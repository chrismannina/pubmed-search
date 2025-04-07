# PubMed Semantic Search with MeSH Refinement

This script demonstrates an enhanced PubMed search approach.
It combines:
1.  **LLM-powered MeSH Term Generation:** Translates a natural language query into relevant MeSH terms using an OpenAI model (e.g., GPT-4o).
2.  **MeSH-Refined Keyword Search:** Uses the generated MeSH terms to perform a more targeted initial keyword search on PubMed via the NCBI Entrez API.
3.  **Semantic Ranking:** Embeds the abstracts of the retrieved articles using OpenAI embeddings and ranks them based on semantic similarity to the original user query.
4.  **Persistent Vector Store:** Uses ChromaDB to store article embeddings and metadata, reducing redundant API calls and processing for previously seen articles.

## Setup

1.  **Clone/Download:** Get the code.
2.  **Environment File:** Create a file named `.env` in the project root.
3.  **API Key:** Add your OpenAI API key to the `.env` file:
    ```dotenv
    OPENAI_API_KEY='your_actual_openai_api_key_here'
    ```
    Replace `'your_actual_openai_api_key_here'` with your real key.
4.  **NCBI Email (Recommended):** Add your email address to the `.env` file. NCBI requires this for using the Entrez API.
    ```dotenv
    NCBI_EMAIL='your.actual.email@example.org'
    ```
    If not set, the script will use a placeholder and show a warning.
5.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *(Installs `requests`, `biopython`, `openai`, `chromadb`, `python-dotenv`, `numpy`)*

## Usage

Run the script from your terminal:

```bash
python pubmed_search.py 
```

The script will prompt you to enter a medical query.

Example:
```
Enter your medical query: treatment options for metastatic melanoma
```

It will then:
1.  Generate relevant MeSH terms for the query using the specified LLM.
2.  Perform a refined keyword search on PubMed using the original query AND the generated MeSH terms.
3.  Check the local ChromaDB vector store for the retrieved PMIDs.
4.  Fetch details (abstract, title, date, authors, etc.) for any *new* PMIDs from PubMed.
5.  Generate embeddings for the new articles using the specified OpenAI model.
6.  Add the new articles (embeddings, metadata) to the persistent ChromaDB store.
7.  Generate an embedding for the original user query.
8.  Perform a semantic search within the relevant articles (those from the keyword search now present in ChromaDB) based on cosine similarity between the query embedding and article embeddings.
9.  Display the top K results (default is 10) with PMID, title, similarity score, publication date, journal, authors, MeSH terms, and abstract snippet.

## Configuration

You can adjust the following parameters at the top of `pubmed_search.py`:

*   `OPENAI_EMBED_MODEL`: The OpenAI embedding model to use (default: `"text-embedding-3-small"`).
*   `LLM_MESH_MODEL`: The OpenAI chat model used for MeSH term generation (default: `"gpt-4o"`).
*   `MAX_KEYWORD_RESULTS`: The maximum number of articles to retrieve from the initial MeSH-refined keyword search (default: 100).
*   `TOP_K`: The final number of top-ranked articles to display after semantic ranking (default: 10).
*   `CHROMA_DB_PATH`: The directory where the persistent ChromaDB vector store is saved (default: `"./chroma_db"`).
*   `collection_name`: The name for the ChromaDB collection (default: `"pubmed_articles_openai"`). 