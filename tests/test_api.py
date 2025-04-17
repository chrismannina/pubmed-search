# Placeholder for API endpoint tests
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

# Adjust the import path based on your project structure
# If api.py is in the root, this should work when running with 'python -m pytest'
from api import app

client = TestClient(app)


def test_read_root():
    # Rewritten docstring cleanly
    """Test the root endpoint '/'."""
    response = client.get("/")
    assert response.status_code == 200
    # Basic check for content type, more robust than exact JSON match for now
    assert "application/json" in response.headers["content-type"]
    # We can add back more specific JSON checks later if needed
    # assert response.json() == {\"message\": \"PubMed Search API is running.\"}


# TODO: Add tests for the /search endpoint in api.py
# These will require mocking external calls (OpenAI, Entrez, ChromaDB)

# --- Integration Tests for /search ---

# TODO: Add more integration tests for the /search endpoint:
#   - Test ranking_mode="pubmed"
#   - Test ranking_mode="semantic"
#   - Test with min_year date filter (successful filtering)
#   - Test with min_year date filter resulting in no eligible articles
#   - Test error case: Initial PubMed search yields no results
#   - Test error case: ensure_articles_in_db yields no results (fetch/embed failure)
#   - Test error case: semantic ranking fails during hybrid mode (fallback to pubmed)
#   - Test error case: ChromaDB .get fails during final metadata fetch
#   - Test variations: Different top_k, max_pubmed_results, override models/modes


def test_search_endpoint_success_hybrid():
    """Test a successful call to /search with hybrid ranking."""
    with patch(
        "api.config",
        new={
            "LLM_MESH_MODEL": "test-mesh-model",
            "EMBED_CONTENT_MODE": "title_abstract",
            "NCBI_EMAIL": "test@example.com",
            "OPENAI_EMBED_MODEL": "test-embed-model",
        },
    ) as mock_config, patch("api.openai_client") as mock_openai, patch(
        "api.chroma_collection"
    ) as mock_collection, patch(
        "api.hybrid_rank_rrf"
    ) as mock_rrf, patch(
        "api.semantic_rank_articles"
    ) as mock_semantic, patch(
        "api.ensure_articles_in_db"
    ) as mock_ensure, patch(
        "api.search_pubmed_keywords"
    ) as mock_search_pubmed, patch(
        "api.generate_mesh_terms"
    ) as mock_generate_mesh:
        # --- Mock Configuration ---
        test_query = "test disease"
        mock_generate_mesh.return_value = ["Disease MeSH"]
        mock_search_pubmed.return_value = (
            ["1", "2", "3"],
            '(test disease) AND ("Disease MeSH"[MeSH Major Topic])',
        )  # PMIDs and query used
        mock_ensure.return_value = ["1", "2", "3"]  # All PMIDs available
        # Mock semantic ranking (needed for RRF)
        mock_semantic.return_value = [("2", 0.9), ("1", 0.8), ("3", 0.7)]
        # Mock RRF result
        mock_rrf.return_value = [("2", 0.8), ("1", 0.75), ("3", 0.6)]
        # Mock chroma_collection.get for final metadata fetch
        # Provide more complete metadata matching the structure after ensure_articles_in_db
        mock_collection.get.return_value = {
            "ids": ["2", "1", "3"],
            "documents": [
                "Title 2. Doc 2",
                "Title 1. Doc 1",
                "Title 3. Doc 3",
            ],  # Assume title+abstract embed mode
            "metadatas": [
                {
                    "title": "Title 2",
                    "pub_date": "2023-01-01",
                    "journal": "Journal B",
                    "authors": "Doe J, Test T",
                    "mesh_headings": "MeSH B, MeSH A",
                    "keywords": "KW2, KW1",
                    "publication_types": "Review",
                    "language": "eng",
                    "doi": "10.doi/2",
                },
                {
                    "title": "Title 1",
                    "pub_date": "2022-01-01",
                    "journal": "Journal A",
                    "authors": "Smith A",
                    "mesh_headings": "MeSH A",
                    "keywords": "KW1",
                    "publication_types": "Journal Article",
                    "language": "eng",
                    "doi": "10.doi/1",
                },
                {
                    "title": "Title 3",
                    "pub_date": "2021-01-01",
                    "journal": "Journal C",
                    "authors": "Ray B",
                    "mesh_headings": "MeSH C",
                    "keywords": "",
                    "publication_types": "Clinical Trial",
                    "language": "fre",
                    "doi": "10.doi/3",
                },
            ],
        }

        # --- API Call ---
        request_data = {"query": test_query, "ranking_mode": "hybrid", "top_k": 3}
        response = client.post("/search", json=request_data)

        # --- Assertions ---
        assert response.status_code == 200
        data = response.json()

        assert data["query"] == test_query
        assert data["ranking_mode"] == "hybrid"
        assert data["generated_mesh_terms"] == ["Disease MeSH"]
        assert data["results_count"] == 3
        assert len(data["results"]) == 3
        # Check order
        assert data["results"][0]["pmid"] == "2"
        assert data["results"][1]["pmid"] == "1"
        assert data["results"][2]["pmid"] == "3"
        # Check detailed fields of first result
        res1 = data["results"][0]
        assert res1["title"] == "Title 2"
        assert res1["rank"] == 1
        assert res1["score"] == pytest.approx(0.8)
        assert res1["pubDate"] == "2023-01-01"
        assert len(res1["authors"]) == 2
        assert res1["authors"][0]["display_name"] == "Doe J"
        assert res1["publicationTypes"] == ["Review"]
        assert res1["meshHeadings"] == ["MeSH B", "MeSH A"]
        assert res1["doi"] == "10.doi/2"
        assert res1["doi_url"] == "https://doi.org/10.doi/2"


# TODO: Add more integration tests for other ranking modes, date filter, error cases, etc.

# Remove the placeholder test
# def test_example_api():
#     assert True
