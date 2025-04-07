# Project Roadmap: Enhanced PubMed Semantic Search

This document outlines potential future improvements and features for the PubMed Semantic Search tool.

## Core Search & Retrieval Enhancements

*   **LLM Query Expansion:**
    *   Experiment with having the LLM generate multiple variations of the user query to potentially broaden the initial MeSH/keyword search and capture different facets of the topic.
    *   Refine the LLM prompt for MeSH term generation for higher accuracy and relevance.
*   **Hybrid Ranking Strategies:**
    *   Investigate combining the semantic similarity score (from embeddings) with the relevance score returned by the initial PubMed keyword/MeSH search (e.g., using Reciprocal Rank Fusion - RRF) for potentially more robust ranking.
*   **Advanced Filtering (Post-Search):**
    *   Allow users to filter the *semantically ranked* results by metadata fields (e.g., Publication Date range, Journal Title, Authors, specific MeSH Headings).
    *   Implement re-ranking based on these filters.
*   **Metadata Enrichment:**
    *   Extract additional potentially useful metadata from PubMed records (e.g., Publication Types, Grant Information, DOI).
    *   Include direct links to the articles on PubMed (e.g., `https://pubmed.ncbi.nlm.nih.gov/{pmid}/`).

## Backend & Architecture Improvements

*   **Framework Migration (FastAPI):**
    *   Convert the script into a FastAPI application to provide a proper API structure.
    *   This enables easier integration with a frontend and allows for more structured request/response handling.
*   **Data Modeling (Pydantic):**
    *   Define Pydantic models for API requests (e.g., search query, filter parameters) and responses (e.g., article details, search results) to ensure data validation and clear API contracts.
*   **Framework Integration (LlamaIndex):**
    *   Evaluate using LlamaIndex to manage the overall workflow:
        *   Document loading (fetching from PubMed).
        *   Embedding generation.
        *   Vector store interaction (ChromaDB).
        *   Potentially handling the LLM query expansion/MeSH generation steps within a LlamaIndex pipeline.
        *   Simplifies swapping components (e.g., different embedding models, vector stores).
*   **Robust Error Handling:**
    *   Implement more comprehensive error handling for API calls (NCBI, OpenAI) and database interactions.
*   **Configuration Management:**
    *   Improve configuration (e.g., using a dedicated config file or more structured environment variables).

## Frontend & User Experience

*   **Web Interface:**
    *   Develop a simple web frontend (e.g., using Streamlit, React, Vue) to interact with the FastAPI backend.
*   **User Controls:**
    *   Allow users to configure parameters like the number of initial keyword results (`MAX_KEYWORD_RESULTS`) and the final number of semantic results (`TOP_K`).
    *   Provide options for applying metadata filters (see Core Enhancements).
*   **Transparency:**
    *   Display the generated MeSH terms and the final PubMed query string used for the initial search to the user for clarity.
*   **Result Display:**
    *   Improve the formatting and presentation of search results.
    *   Highlight matched keywords or semantically relevant snippets within the abstract.

## Evaluation & Optimization

*   **Performance Monitoring:** Track API usage, execution time, and database size.
*   **Relevance Evaluation:** Develop methods to evaluate the quality and relevance of search results compared to standard PubMed or other baselines.
