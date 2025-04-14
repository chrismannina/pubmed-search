# Placeholder for search logic tests
import pytest
from search import parse_pubmed_date # Import the function to test
from unittest.mock import patch, MagicMock
from openai import OpenAI # Import OpenAI for type hinting if needed

# Import functions to test
from search import generate_mesh_terms, hybrid_rank_rrf, get_openai_embeddings, search_pubmed_keywords

# Test cases for parse_pubmed_date
@pytest.mark.parametrize("input_dict, expected_output", [
    # Standard cases
    ({'Year': '2023', 'Month': '05', 'Day': '15'}, "2023-05-15"),
    ({'Year': '2024', 'Month': 'Feb', 'Day': '29'}, "2024-02-29"), # Leap year
    ({'Year': '2023', 'Month': 'Feb', 'Day': '29'}, "2023-02-28"), # Non-leap year
    # Missing day/month
    ({'Year': '2022', 'Month': 'Jul'}, "2022-07-01"),
    ({'Year': '2021'}, "2021-01-01"),
    # MedlineDate fallback
    ({'MedlineDate': '2020 Spring'}, None), # Cannot parse fully
    ({'MedlineDate': '2019 Nov-Dec'}, "2019-11-01"), # Takes first month
    ({'MedlineDate': '2018'}, "2018-01-01"),
    # Invalid/Edge cases
    ({}, None),
    ({'Year': 'abc'}, None),
    ({'Year': '2023', 'Month': '13'}, "2023-01-01"), # Invalid month defaults to Jan
    ({'Year': '2023', 'Month': 'Apr', 'Day': '31'}, "2023-04-30"), # Invalid day clamped
])
def test_parse_pubmed_date(input_dict, expected_output):
    assert parse_pubmed_date(input_dict) == expected_output

# --- Tests for generate_mesh_terms --- 

# Mock OpenAI client for testing generate_mesh_terms
@pytest.fixture
def mock_openai_client_chat():
    mock_client = MagicMock(spec=OpenAI)
    # Explicitly create nested mocks
    mock_completions = MagicMock()
    mock_chat = MagicMock()
    mock_chat.completions = mock_completions
    mock_client.chat = mock_chat
    
    # Configure the mock response structure
    mock_response = MagicMock()
    mock_choice = MagicMock()
    mock_message = MagicMock()
    mock_message.content = "Term1, Term2, Another MeSH Term, short"
    mock_choice.message = mock_message
    mock_response.choices = [mock_choice]
    # Set return value on the correct mock method
    mock_client.chat.completions.create.return_value = mock_response
    return mock_client

def test_generate_mesh_terms_success(mock_openai_client_chat):
    """Test successful MeSH term generation and parsing."""
    query = "test query"
    expected_terms = ["Term1", "Term2", "Another MeSH Term", "short"]
    
    generated_terms = generate_mesh_terms(query, mock_openai_client_chat)
    
    assert generated_terms == expected_terms
    # Check if the mock was called correctly
    mock_openai_client_chat.chat.completions.create.assert_called_once()
    call_args, call_kwargs = mock_openai_client_chat.chat.completions.create.call_args
    assert call_kwargs['model'] is not None # Check model was passed
    assert query in call_kwargs['messages'][1]['content'] # Check query in user message

def test_generate_mesh_terms_api_error(mock_openai_client_chat):
    """Test handling of OpenAI API errors."""
    query = "error query"
    # Simulate an API error
    mock_openai_client_chat.chat.completions.create.side_effect = Exception("API Error")
    
    generated_terms = generate_mesh_terms(query, mock_openai_client_chat)
    
    assert generated_terms == [] # Expect empty list on error

def test_generate_mesh_terms_empty_response(mock_openai_client_chat):
    """Test handling of empty or non-comma-separated LLM responses."""
    query = "empty response query"
    mock_response = MagicMock()
    mock_choice = MagicMock()
    mock_message = MagicMock()
    mock_message.content = "  " # Empty content
    mock_choice.message = mock_message
    mock_response.choices = [mock_choice]
    mock_openai_client_chat.chat.completions.create.return_value = mock_response

    generated_terms = generate_mesh_terms(query, mock_openai_client_chat)
    assert generated_terms == []

# --- Tests for get_openai_embeddings --- 

@pytest.fixture
def mock_openai_client_embed():
    mock_client = MagicMock(spec=OpenAI)
    # Explicitly create nested mocks
    mock_embeddings_obj = MagicMock()
    mock_client.embeddings = mock_embeddings_obj
    
    # Configure mock response
    mock_embedding1 = MagicMock()
    mock_embedding1.embedding = [0.1, 0.2, 0.3]
    mock_embedding2 = MagicMock()
    mock_embedding2.embedding = [0.4, 0.5, 0.6]
    mock_response = MagicMock()
    mock_response.data = [mock_embedding1, mock_embedding2]
    # Set return value on the correct mock method
    mock_client.embeddings.create.return_value = mock_response
    return mock_client

def test_get_openai_embeddings_success(mock_openai_client_embed):
    texts = ["text 1", "text 2"]
    model = "test-embed-model"
    expected_embeddings = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
    
    embeddings = get_openai_embeddings(texts, mock_openai_client_embed, model)
    
    assert embeddings == expected_embeddings
    mock_openai_client_embed.embeddings.create.assert_called_once_with(input=texts, model=model)

def test_get_openai_embeddings_api_error(mock_openai_client_embed):
    mock_openai_client_embed.embeddings.create.side_effect = Exception("Embed API Error")
    embeddings = get_openai_embeddings(["text 1"], mock_openai_client_embed, "model")
    assert embeddings == []

def test_get_openai_embeddings_empty_input(mock_openai_client_embed):
    embeddings = get_openai_embeddings([], mock_openai_client_embed, "model")
    assert embeddings == []
    mock_openai_client_embed.embeddings.create.assert_not_called()

# --- Tests for search_pubmed_keywords --- 

# Use patch context manager for Entrez methods
@patch('search.Entrez.read')
@patch('search.Entrez.esearch')
def test_search_pubmed_keywords_mesh_major_success(mock_esearch, mock_read):
    """Test successful search using Attempt 1 (Text + MeSH Major)."""
    # Mock esearch to return a MagicMock (simulating a handle)
    mock_handle = MagicMock()
    mock_esearch.return_value = mock_handle
    mock_read.return_value = {'IdList': ['1', '2', '3'] * 10} # Simulate >= 20 results
    
    query = "original query text"
    mesh_terms = ["TermA", "TermB"]
    max_results = 50
    email = "test@example.com"
    
    expected_query = '(original query text) AND ("TermA"[MeSH Major Topic] OR "TermB"[MeSH Major Topic])'
    expected_pmids = ['1', '2', '3'] * 10
    
    pmids, query_used = search_pubmed_keywords(query, mesh_terms, max_results, email)
    
    assert pmids == expected_pmids
    assert query_used == expected_query
    mock_esearch.assert_called_once_with(db="pubmed", term=expected_query, retmax=str(max_results), sort="relevance")
    mock_read.assert_called_once_with(mock_handle) # Check read was called with the mock handle
    mock_handle.close.assert_called_once() # Check that handle.close() was called

@patch('search.Entrez.read')
@patch('search.Entrez.esearch')
def test_search_pubmed_keywords_fallback_to_mesh_terms(mock_esearch, mock_read):
    """Test fallback to Attempt 2 (Text + MeSH Terms) when Attempt 1 yields few results."""
    # Mock esearch to return a MagicMock (simulating a handle)
    mock_handle = MagicMock()
    mock_esearch.return_value = mock_handle
    # Configure mock for two calls: first fails threshold, second succeeds
    mock_read_attempt1 = {'IdList': ['1', '2']} # Too few results
    mock_read_attempt2 = {'IdList': ['10', '11', '12'] * 10} # Enough results
    mock_read.side_effect = [mock_read_attempt1, mock_read_attempt2]
    
    query = "original query text"
    mesh_terms = ["TermA", "TermB"]
    max_results = 50
    email = "test@example.com"
    
    expected_query_attempt1 = '(original query text) AND ("TermA"[MeSH Major Topic] OR "TermB"[MeSH Major Topic])'
    expected_query_attempt2 = '(original query text) AND ("TermA"[MeSH Terms] OR "TermB"[MeSH Terms])'
    expected_pmids = ['10', '11', '12'] * 10
    
    pmids, query_used = search_pubmed_keywords(query, mesh_terms, max_results, email)
    
    assert pmids == expected_pmids
    assert query_used == expected_query_attempt2
    assert mock_esearch.call_count == 2
    assert mock_read.call_count == 2
    # Check args of the second esearch call
    mock_esearch.assert_called_with(db="pubmed", term=expected_query_attempt2, retmax=str(max_results), sort="relevance")
    # Check handle.close was called (at least once for each successful esearch call)
    assert mock_handle.close.call_count == 2

@patch('search.Entrez.read')
@patch('search.Entrez.esearch')
def test_search_pubmed_keywords_fallback_to_original_query(mock_esearch, mock_read):
    """Test fallback to Attempt 3 (Original Text) when Attempts 1 & 2 yield few results."""
    # Mock esearch to return a MagicMock (simulating a handle)
    mock_handle = MagicMock()
    mock_esearch.return_value = mock_handle
    # Mock read for three calls
    mock_read_attempt1 = {'IdList': ['1']} # Too few
    mock_read_attempt2 = {'IdList': ['2']} # Still too few
    mock_read_attempt3 = {'IdList': ['100', '101']} # Final result
    mock_read.side_effect = [mock_read_attempt1, mock_read_attempt2, mock_read_attempt3]
    
    query = "original query text"
    mesh_terms = ["TermA"]
    max_results = 50
    email = "test@example.com"
    
    expected_pmids = ['100', '101']
    
    pmids, query_used = search_pubmed_keywords(query, mesh_terms, max_results, email)
    
    assert pmids == expected_pmids
    assert query_used == query # Should have fallen back to original query text
    assert mock_esearch.call_count == 3
    assert mock_read.call_count == 3
    # Check args of the third esearch call
    mock_esearch.assert_called_with(db="pubmed", term=query, retmax=str(max_results), sort="relevance")
    assert mock_handle.close.call_count == 3

@patch('search.Entrez.esearch')
def test_search_pubmed_keywords_no_mesh(mock_esearch):
    """Test search when no mesh terms are provided."""
    # Configure mocks - only expect one call
    # Mock esearch to return a MagicMock (simulating a handle)
    mock_handle = MagicMock()
    mock_esearch.return_value = mock_handle
    mock_read_instance = MagicMock(return_value={'IdList': ['200']})
    with patch('search.Entrez.read', mock_read_instance):
        
        query = "original query text"
        mesh_terms = None
        max_results = 50
        email = "test@example.com"
        
        expected_pmids = ['200']
        
        pmids, query_used = search_pubmed_keywords(query, mesh_terms, max_results, email)
        
        assert pmids == expected_pmids
        assert query_used == query
        mock_esearch.assert_called_once_with(db="pubmed", term=query, retmax=str(max_results), sort="relevance")
        mock_read_instance.assert_called_once_with(mock_handle)
        mock_handle.close.assert_called_once()

# --- Tests for hybrid_rank_rrf --- 

def test_hybrid_rank_rrf_basic():
    """Test basic RRF calculation."""
    pubmed_ranked = ['1', '2', '3', '4']
    semantic_ranked = [('2', 0.9), ('1', 0.8), ('4', 0.7), ('3', 0.6)] # (pmid, score)
    k = 60
    
    # Expected scores (approx): 
    # PMID 1: 1/(60+1) + 1/(60+2) = 0.01639 + 0.01613 = 0.03252
    # PMID 2: 1/(60+2) + 1/(60+1) = 0.01613 + 0.01639 = 0.03252
    # PMID 3: 1/(60+3) + 1/(60+4) = 0.01587 + 0.01563 = 0.03150
    # PMID 4: 1/(60+4) + 1/(60+3) = 0.01563 + 0.01587 = 0.03150
    # Expected order: 1, 2 (tied), 4, 3 (tied) - exact order of ties may vary
    
    rrf_results = hybrid_rank_rrf(pubmed_ranked, semantic_ranked, k)
    
    assert len(rrf_results) == 4
    # Check that the top scores are higher than lower scores
    assert rrf_results[0][1] > rrf_results[2][1] # Score of 1/2 should be > score of 3/4
    # Check the PMIDs are present
    assert set(r[0] for r in rrf_results) == {'1', '2', '3', '4'}
    # Check relative order of non-tied items
    assert rrf_results[0][0] in {'1', '2'}
    assert rrf_results[1][0] in {'1', '2'}
    assert rrf_results[2][0] in {'3', '4'}
    assert rrf_results[3][0] in {'3', '4'}
    # Check scores are roughly correct (allow for floating point)
    assert rrf_results[0][1] == pytest.approx(1.0/(k+1) + 1.0/(k+2))
    assert rrf_results[2][1] == pytest.approx(1.0/(k+3) + 1.0/(k+4))

def test_hybrid_rank_rrf_disjoint():
    """Test RRF when lists have different items."""
    pubmed_ranked = ['1', '2']
    semantic_ranked = [('2', 0.9), ('3', 0.8)] # PMID 1 only in pubmed, 3 only in semantic
    k = 1
    
    # Expected scores:
    # PMID 1: 1/(1+1) = 0.5
    # PMID 2: 1/(1+2) + 1/(1+1) = 0.333 + 0.5 = 0.833
    # PMID 3: 1/(1+2) = 0.333
    # Order: 2, 1, 3
    
    rrf_results = hybrid_rank_rrf(pubmed_ranked, semantic_ranked, k)
    
    assert len(rrf_results) == 3
    assert [r[0] for r in rrf_results] == ['2', '1', '3']
    assert rrf_results[0][1] == pytest.approx(1.0/3 + 1.0/2)
    assert rrf_results[1][1] == pytest.approx(1.0/2)
    assert rrf_results[2][1] == pytest.approx(1.0/3)

# TODO: Add tests for semantic_rank_articles (requires mocking ChromaDB query)
# TODO: Add tests for ensure_articles_in_db (complex, requires mocking fetch, embed, ChromaDB add/get)

# Remove the old example test
# def test_example():
#     assert True 