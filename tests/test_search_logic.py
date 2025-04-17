# Placeholder for search logic tests
import pytest
from search import parse_pubmed_date # Import the function to test
from unittest.mock import patch, MagicMock
from openai import OpenAI # Import OpenAI for type hinting if needed
import chromadb # <--- Import chromadb

# Import functions to test
from search import generate_mesh_terms, hybrid_rank_rrf, get_openai_embeddings, search_pubmed_keywords, fetch_abstracts, semantic_rank_articles, ensure_articles_in_db, PubMedArticle, Author

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

# --- Tests for semantic_rank_articles --- 

# Mock ChromaDB collection for semantic ranking tests
@pytest.fixture
def mock_chroma_collection_query():
    mock_collection = MagicMock(spec=chromadb.Collection)
    # Simulate response from collection.query
    mock_response = {
        'ids': [['pmid1', 'pmid3', 'pmid2']], # Note: Order might not match input
        'distances': [[0.1, 0.3, 0.2]], # Lower distance = more similar
        'metadatas': None, # Not needed for this func's logic
        'embeddings': None,
        'documents': None
    }
    mock_collection.query.return_value = mock_response
    return mock_collection

# Use patch for get_openai_embeddings within semantic_rank_articles tests
@patch('search.get_openai_embeddings')
def test_semantic_rank_articles_success(mock_get_embed, mock_chroma_collection_query):
    """Test successful semantic ranking and sorting."""
    mock_get_embed.return_value = [[0.1, 0.9]] # Mock query embedding
    mock_openai_client = MagicMock() # Dummy client needed for signature
    
    query = "semantic query"
    pmids_to_rank = ['pmid1', 'pmid2', 'pmid3']
    top_k = 3
    embed_model = "test-model"
    
    # Expected: pmid1 (dist 0.1 -> score 0.9), pmid2 (dist 0.2 -> score 0.8), pmid3 (dist 0.3 -> score 0.7)
    expected_ranking = [('pmid1', pytest.approx(0.9)), ('pmid2', pytest.approx(0.8)), ('pmid3', pytest.approx(0.7))]
    
    ranked_results = semantic_rank_articles(
        query, pmids_to_rank, mock_chroma_collection_query, top_k, mock_openai_client, embed_model
    )
    
    assert ranked_results == expected_ranking
    # Verify mocks were called
    mock_get_embed.assert_called_once_with([query], mock_openai_client, embed_model)
    mock_chroma_collection_query.query.assert_called_once()
    call_args, call_kwargs = mock_chroma_collection_query.query.call_args
    assert call_kwargs['query_embeddings'] == [[0.1, 0.9]]

@patch('search.get_openai_embeddings')
def test_semantic_rank_articles_empty_input(mock_get_embed, mock_chroma_collection_query):
    """Test semantic ranking with empty input PMIDs."""
    mock_openai_client = MagicMock()
    ranked_results = semantic_rank_articles(
        "query", [], mock_chroma_collection_query, 3, mock_openai_client, "model"
    )
    assert ranked_results == []
    mock_get_embed.assert_not_called()
    mock_chroma_collection_query.query.assert_not_called()

@patch('search.get_openai_embeddings')
def test_semantic_rank_articles_embed_fail(mock_get_embed, mock_chroma_collection_query):
    """Test semantic ranking when query embedding fails."""
    mock_get_embed.return_value = [] # Simulate embedding failure
    mock_openai_client = MagicMock()
    ranked_results = semantic_rank_articles(
        "query", ['1'], mock_chroma_collection_query, 3, mock_openai_client, "model"
    )
    assert ranked_results == []
    mock_chroma_collection_query.query.assert_not_called()

@patch('search.get_openai_embeddings')
def test_semantic_rank_articles_chroma_fail(mock_get_embed, mock_chroma_collection_query):
    """Test semantic ranking when ChromaDB query fails."""
    mock_get_embed.return_value = [[0.1]]
    mock_chroma_collection_query.query.side_effect = Exception("DB Error")
    mock_openai_client = MagicMock()
    ranked_results = semantic_rank_articles(
        "query", ['1'], mock_chroma_collection_query, 3, mock_openai_client, "model"
    )
    assert ranked_results == []
    mock_chroma_collection_query.query.assert_called_once()

@patch('search.get_openai_embeddings')
def test_semantic_rank_articles_top_k_limit(mock_get_embed, mock_chroma_collection_query):
    """Test that top_k correctly limits results."""
    mock_get_embed.return_value = [[0.1]]
    # Make query return more results than top_k
    mock_response = {
        'ids': [['pmid1', 'pmid3', 'pmid2', 'pmid4']], 
        'distances': [[0.1, 0.3, 0.2, 0.4]], 
        # ... other keys ...
    }
    mock_chroma_collection_query.query.return_value = mock_response
    mock_openai_client = MagicMock()

    ranked_results = semantic_rank_articles(
        "query", ['pmid1', 'pmid2', 'pmid3', 'pmid4'], mock_chroma_collection_query, 2, mock_openai_client, "model"
    )
    # Expected: pmid1 (score 0.9), pmid2 (score 0.8)
    assert len(ranked_results) == 2
    assert ranked_results[0][0] == 'pmid1'
    assert ranked_results[1][0] == 'pmid2'

@patch('search.get_openai_embeddings')
def test_semantic_rank_articles_filters_pmids(mock_get_embed, mock_chroma_collection_query):
    """Test that results are filtered by pmids_to_rank."""
    mock_get_embed.return_value = [[0.1]]
    # Make query return PMIDs not in pmids_to_rank
    mock_response = {
        'ids': [['pmid1', 'pmid99', 'pmid2']], 
        'distances': [[0.1, 0.15, 0.2]], 
        # ... other keys ...
    }
    mock_chroma_collection_query.query.return_value = mock_response
    mock_openai_client = MagicMock()
    
    ranked_results = semantic_rank_articles(
        "query", ['pmid1', 'pmid2'], mock_chroma_collection_query, 3, mock_openai_client, "model"
    )
    # Expected: pmid1 (score 0.9), pmid2 (score 0.8). pmid99 should be excluded.
    assert len(ranked_results) == 2
    assert ranked_results[0][0] == 'pmid1'
    assert ranked_results[1][0] == 'pmid2'

# Remove the old example test
# def test_example():
#     assert True  

# --- Mocks for Entrez --- 

# Sample data mimicking Entrez.read output for fetch_abstracts
SAMPLE_EFETCH_RECORD_COMPLETE = {
    'PubmedArticle': [{
        'MedlineCitation': {
            'PMID': '123',
            'Article': {
                'ArticleTitle': 'Test Title 1',
                'Language': ['eng'],
                'Pagination': {'MedlinePgn': '1-10'},
                'Abstract': {'AbstractText': ['Abstract content.']},
                'AuthorList': [{
                    'LastName': 'Smith', 'ForeName': 'John', 'Initials': 'J'
                }],
                'Journal': {
                    'Title': 'Journal of Testing', 
                    'ISOAbbreviation': 'J Test',
                    'JournalIssue': {'Volume': '10', 'Issue': '2', 'PubDate': {'Year': '2022', 'Month': 'Mar'}}
                },
                'PublicationTypeList': ['Journal Article', 'Review']
            },
            'MeshHeadingList': [{'DescriptorName': 'MeSH Term 1'}],
            'KeywordList': [['Keyword1', 'Keyword2']] # Nested list structure
        },
        'PubmedData': {
            'ArticleIdList': [{'IdType': 'pubmed', '#text': '123'}, {'IdType': 'doi', '#text': '10.1234/test.doi'}]
        }
    }]
}

SAMPLE_EFETCH_RECORD_MINIMAL = {
    'PubmedArticle': [{
        'MedlineCitation': {
            'PMID': '456',
            'Article': {
                'ArticleTitle': 'Minimal Title',
                'Abstract': {'AbstractText': ['Minimal abstract.']},
                'Journal': {'JournalIssue': {'PubDate': {'MedlineDate': '2021'}}}
            }
        },
        'PubmedData': {}
    }]
}

SAMPLE_EFETCH_RECORD_NO_ABSTRACT = {
    'PubmedArticle': [{
        'MedlineCitation': {
            'PMID': '789',
            'Article': {
                'ArticleTitle': 'No Abstract Title',
                 'Journal': {'JournalIssue': {'PubDate': {'Year': '2023'}}}
            }
        },
         'PubmedData': {}
    }]
}

# --- Tests for fetch_abstracts --- 

@patch('search.Entrez.read')
@patch('search.Entrez.efetch')
def test_fetch_abstracts_success_complete(mock_efetch, mock_read):
    """Test fetching a complete article record."""
    mock_efetch.return_value = MagicMock() # Simulate handle
    mock_read.return_value = SAMPLE_EFETCH_RECORD_COMPLETE
    
    pmids = ['123']
    email = "test@example.com"
    
    articles = fetch_abstracts(pmids, email)
    
    mock_efetch.assert_called_once_with(db="pubmed", id=pmids, rettype="medline", retmode="xml")
    mock_read.assert_called_once()
    assert '123' in articles
    article = articles['123']
    assert article.pmid == '123'
    assert article.title == 'Test Title 1'
    assert article.abstract == 'Abstract content.'
    assert article.pubDate == '2022-03-01'
    assert article.doi == '10.1234/test.doi'
    assert article.journalTitle == 'Journal of Testing'
    assert len(article.authors) == 1
    assert article.authors[0].lastName == 'Smith'
    assert article.publicationTypes == ['Journal Article', 'Review']
    assert article.meshHeadings == ['MeSH Term 1']
    assert article.keywords == ['Keyword1', 'Keyword2']

@patch('search.Entrez.read')
@patch('search.Entrez.efetch')
def test_fetch_abstracts_success_minimal(mock_efetch, mock_read):
    """Test fetching a record with minimal data."""
    mock_efetch.return_value = MagicMock()
    mock_read.return_value = SAMPLE_EFETCH_RECORD_MINIMAL
    
    pmids = ['456']
    email = "test@example.com"
    articles = fetch_abstracts(pmids, email)
    
    assert '456' in articles
    article = articles['456']
    assert article.title == 'Minimal Title'
    assert article.abstract == 'Minimal abstract.'
    assert article.pubDate == '2021-01-01' # Parsed from MedlineDate
    assert article.doi is None
    assert article.authors == []
    assert article.keywords == []

@patch('search.Entrez.read')
@patch('search.Entrez.efetch')
def test_fetch_abstracts_no_abstract_skipped(mock_efetch, mock_read):
    """Test that articles without abstracts are skipped."""
    mock_efetch.return_value = MagicMock()
    # Combine records, one with abstract, one without
    combined_records = {
        'PubmedArticle': 
            SAMPLE_EFETCH_RECORD_MINIMAL['PubmedArticle'] + 
            SAMPLE_EFETCH_RECORD_NO_ABSTRACT['PubmedArticle']
    }
    mock_read.return_value = combined_records
    
    pmids = ['456', '789']
    email = "test@example.com"
    articles = fetch_abstracts(pmids, email)
    
    assert '456' in articles # Minimal article should be present
    assert '789' not in articles # Article with no abstract should be skipped
    assert len(articles) == 1

@patch('search.Entrez.efetch')
def test_fetch_abstracts_efetch_error(mock_efetch):
    """Test handling of Entrez.efetch errors."""
    mock_efetch.side_effect = Exception("EFetch failed")
    articles = fetch_abstracts(['111'], "test@example.com")
    assert articles == {}  

# --- Tests for ensure_articles_in_db --- 

# Mock ChromaDB collection for ensure_articles_in_db tests
@pytest.fixture
def mock_chroma_collection_ensure():
    mock_collection = MagicMock(spec=chromadb.Collection)
    # We'll configure get/add responses within each test
    return mock_collection

# Mock fetch_abstracts response 
@pytest.fixture
def mock_fetched_articles():
    # Simulate returning data for PMIDs '2' and '3'
    return {
        '2': PubMedArticle(pmid='2', title='Title 2', abstract='Abstract 2', authors=[Author(lastName='Doe')], pubDate='2023-01-01'),
        '3': PubMedArticle(pmid='3', title='Title 3', abstract='Abstract 3', authors=[Author(lastName='Ray')], pubDate='2024-01-01')
    }

# Mock get_openai_embeddings response
@pytest.fixture
def mock_embeddings_ensure():
    return [[0.1]*10], [[0.2]*10] # Two dummy embeddings

# Patch necessary functions within the tests for ensure_articles_in_db
@patch('search.fetch_abstracts')
@patch('search.get_openai_embeddings')
def test_ensure_articles_all_exist(mock_get_embed, mock_fetch, mock_chroma_collection_ensure):
    """Test case where all requested PMIDs already exist in DB."""
    pmids_to_check = ['1', '2', '3']
    mock_chroma_collection_ensure.get.return_value = {'ids': ['1', '2', '3']} # Simulate all exist
    mock_openai_client = MagicMock()
    
    result = ensure_articles_in_db(pmids_to_check, mock_chroma_collection_ensure, mock_openai_client, "m", "a", "e")
    
    assert result == ['1', '2', '3'] # Should return the original list
    mock_chroma_collection_ensure.get.assert_called_once_with(ids=pmids_to_check, include=[])
    # Crucially, fetch, embed, and add should NOT have been called
    mock_fetch.assert_not_called()
    mock_get_embed.assert_not_called()
    mock_chroma_collection_ensure.add.assert_not_called()

@patch('search.fetch_abstracts')
@patch('search.get_openai_embeddings')
def test_ensure_articles_none_exist(mock_get_embed, mock_fetch, mock_chroma_collection_ensure, mock_fetched_articles, mock_embeddings_ensure):
    """Test case where no PMIDs exist and need to be fetched, embedded, added."""
    pmids_to_check = ['2', '3']
    mock_chroma_collection_ensure.get.return_value = {'ids': []} # Simulate none exist
    mock_fetch.return_value = mock_fetched_articles # Simulate fetch success for 2, 3
    mock_get_embed.return_value = mock_embeddings_ensure # Simulate embed success
    mock_chroma_collection_ensure.add.return_value = None # Simulate add success
    mock_openai_client = MagicMock()
    embed_model = "test-model"
    embed_mode = "title_abstract" # Match default or pass explicitly
    email = "test@example.com"

    result = ensure_articles_in_db(pmids_to_check, mock_chroma_collection_ensure, mock_openai_client, embed_model, embed_mode, email)
    
    assert set(result) == {'2', '3'} # Order might vary slightly depending on set union
    mock_chroma_collection_ensure.get.assert_called_once_with(ids=pmids_to_check, include=[])
    mock_fetch.assert_called_once_with(['2', '3'], email)
    # Check that embedding was called with prepared documents
    mock_get_embed.assert_called_once()
    call_args, _ = mock_get_embed.call_args
    assert call_args[0] == ['Title 2. Abstract 2', 'Title 3. Abstract 3'] # Check combined text
    assert call_args[2] == embed_model # Check model passed
    # Check that add was called correctly
    mock_chroma_collection_ensure.add.assert_called_once()
    _, add_kwargs = mock_chroma_collection_ensure.add.call_args
    assert add_kwargs['ids'] == ['2', '3']
    assert add_kwargs['embeddings'] == mock_embeddings_ensure
    assert add_kwargs['documents'] == ['Title 2. Abstract 2', 'Title 3. Abstract 3']
    assert len(add_kwargs['metadatas']) == 2 # Check metadatas were prepared

@patch('search.fetch_abstracts')
@patch('search.get_openai_embeddings')
def test_ensure_articles_some_exist(mock_get_embed, mock_fetch, mock_chroma_collection_ensure, mock_fetched_articles, mock_embeddings_ensure):
    """Test case where some PMIDs exist, others need fetching."""
    pmids_to_check = ['1', '2', '3'] # 1 exists, 2 and 3 need fetching
    mock_chroma_collection_ensure.get.return_value = {'ids': ['1']} 
    mock_fetch.return_value = mock_fetched_articles # Fetches 2, 3
    mock_get_embed.return_value = mock_embeddings_ensure
    mock_openai_client = MagicMock()
    email = "test@example.com"

    result = ensure_articles_in_db(pmids_to_check, mock_chroma_collection_ensure, mock_openai_client, "m", "title_abstract", email)

    assert set(result) == {'1', '2', '3'}
    mock_chroma_collection_ensure.get.assert_called_once_with(ids=pmids_to_check, include=[])
    # Fetch only called for missing PMIDs
    mock_fetch.assert_called_once_with(['2', '3'], email)
    # Embed only called for fetched PMIDs
    mock_get_embed.assert_called_once()
    call_args, _ = mock_get_embed.call_args
    assert call_args[0] == ['Title 2. Abstract 2', 'Title 3. Abstract 3']
    # Add only called for fetched PMIDs
    mock_chroma_collection_ensure.add.assert_called_once()
    _, add_kwargs = mock_chroma_collection_ensure.add.call_args
    assert add_kwargs['ids'] == ['2', '3']

@patch('search.fetch_abstracts')
@patch('search.get_openai_embeddings')
def test_ensure_articles_fetch_fails(mock_get_embed, mock_fetch, mock_chroma_collection_ensure):
    """Test case where fetching new articles fails."""
    pmids_to_check = ['1', '2'] # 1 exists, 2 needs fetching
    mock_chroma_collection_ensure.get.return_value = {'ids': ['1']}
    mock_fetch.return_value = {} # Simulate fetch returning nothing
    mock_openai_client = MagicMock()

    result = ensure_articles_in_db(pmids_to_check, mock_chroma_collection_ensure, mock_openai_client, "m", "a", "e")

    assert result == ['1'] # Only the existing one should be returned
    mock_fetch.assert_called_once_with(['2'], "e")
    mock_get_embed.assert_not_called()
    mock_chroma_collection_ensure.add.assert_not_called()

@patch('search.fetch_abstracts')
@patch('search.get_openai_embeddings')
def test_ensure_articles_embed_fails(mock_get_embed, mock_fetch, mock_chroma_collection_ensure, mock_fetched_articles):
    """Test case where embedding new articles fails."""
    pmids_to_check = ['1', '2'] # 1 exists, 2 needs fetching
    mock_chroma_collection_ensure.get.return_value = {'ids': ['1']}
    # Only need to fetch article 2 for this test
    mock_fetch.return_value = {'2': mock_fetched_articles['2']}
    mock_get_embed.return_value = [] # Simulate embedding failure
    mock_openai_client = MagicMock()

    result = ensure_articles_in_db(pmids_to_check, mock_chroma_collection_ensure, mock_openai_client, "m", "title_abstract", "e")

    assert result == ['1'] # Only the existing one should be returned
    mock_fetch.assert_called_once_with(['2'], "e")
    mock_get_embed.assert_called_once()
    mock_chroma_collection_ensure.add.assert_not_called()

@patch('search.fetch_abstracts')
@patch('search.get_openai_embeddings')
def test_ensure_articles_add_fails(mock_get_embed, mock_fetch, mock_chroma_collection_ensure, mock_fetched_articles, mock_embeddings_ensure):
    """Test case where adding to ChromaDB fails."""
    pmids_to_check = ['1', '2'] # 1 exists, 2 needs fetching
    mock_chroma_collection_ensure.get.return_value = {'ids': ['1']}
    mock_fetch.return_value = {'2': mock_fetched_articles['2']} # Fetch only 2
    mock_get_embed.return_value = [mock_embeddings_ensure[0]] # Embed only 2
    mock_chroma_collection_ensure.add.side_effect = Exception("Add failed") # Simulate add failure
    mock_openai_client = MagicMock()

    result = ensure_articles_in_db(pmids_to_check, mock_chroma_collection_ensure, mock_openai_client, "m", "title_abstract", "e")

    assert result == ['1'] # Only the existing one should be returned
    mock_fetch.assert_called_once_with(['2'], "e")
    mock_get_embed.assert_called_once()
    mock_chroma_collection_ensure.add.assert_called_once()  