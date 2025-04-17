import requests
import json
import os
import datetime
import re

# --- Configuration ---
API_BASE_URL = "http://127.0.0.1:8000"  # Assuming the API runs locally on port 8000
SEARCH_ENDPOINT = f"{API_BASE_URL}/search"
OUTPUT_DIR = "live_query_results"
# Get query from environment variable or use default
DEFAULT_QUERY = "What are recent findings on robotic-assisted laparoscopic surgery?"
QUERY = os.getenv('LIVE_QUERY_STRING', DEFAULT_QUERY)
RANKING_MODES = ["pubmed", "semantic", "hybrid"]
TOP_K = 20  # Number of results to request for each mode

# --- Helper Function ---
def create_slug(text, max_length=50):
    """Creates a filesystem-friendly slug from a string."""
    s = text.lower()
    s = re.sub(r'[^a-z0-9\s-]', '', s)  # Remove non-alphanumeric characters except spaces and hyphens
    s = re.sub(r'[\s-]+', '_', s)      # Replace spaces and hyphens with underscores
    s = s.strip('_')                   # Remove leading/trailing underscores
    return s[:max_length]              # Truncate if too long

# --- Main Execution ---
if __name__ == "__main__":
    print(f"Starting live query test for: \"{QUERY}\"")
    print(f"API Endpoint: {SEARCH_ENDPOINT}")
    print(f"Output Directory: {OUTPUT_DIR}")
    print("-" * 30)

    # Create output directory if it doesn't exist
    try:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        print(f"Ensured output directory exists: {OUTPUT_DIR}")
    except OSError as e:
        print(f"Error creating directory {OUTPUT_DIR}: {e}")
        exit(1)

    # Generate a timestamp and query slug for filenames
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    query_slug = create_slug(QUERY)
    print(f"Timestamp for this run: {timestamp}")
    print(f"Query slug: {query_slug}")
    print("-" * 30)


    all_successful = True
    for mode in RANKING_MODES:
        print(f"Testing Ranking Mode: '{mode}'...")

        request_data = {
            "query": QUERY,
            "ranking_mode": mode,
            "top_k": TOP_K
            # Add other parameters like min_year if needed
            # "min_year": 2020
        }

        try:
            response = requests.post(SEARCH_ENDPOINT, json=request_data, timeout=120) # Increased timeout for potentially long semantic searches
            response.raise_for_status()  # Raise an exception for bad status codes (4xx or 5xx)

            print(f"  Successfully received response (Status: {response.status_code})")
            results_data = response.json()

            # Define filenames
            base_filename = f"results_{timestamp}_{query_slug}_{mode}"
            json_filename = os.path.join(OUTPUT_DIR, f"{base_filename}.json")
            pubmed_query_filename = os.path.join(OUTPUT_DIR, f"{base_filename}_pubmed_query.txt")

            # Save full JSON response
            try:
                with open(json_filename, 'w', encoding='utf-8') as f_json:
                    json.dump(results_data, f_json, indent=4, ensure_ascii=False)
                print(f"  Saved full JSON response to: {json_filename}")
            except IOError as e:
                print(f"  Error saving JSON file {json_filename}: {e}")
                all_successful = False

            # Save PubMed query used
            pubmed_query_used = results_data.get("pubmed_query_used", "N/A")
            try:
                with open(pubmed_query_filename, 'w', encoding='utf-8') as f_query:
                    f_query.write(f"Query sent to API: {QUERY}\n")
                    f_query.write(f"Ranking Mode: {mode}\n")
                    f_query.write(f"Top K requested: {TOP_K}\n\n")
                    f_query.write("PubMed Query Used (from API response):\n")
                    f_query.write("-" * 20 + "\n")
                    f_query.write(pubmed_query_used + "\n")
                print(f"  Saved PubMed query used to: {pubmed_query_filename}")
            except IOError as e:
                print(f"  Error saving PubMed query file {pubmed_query_filename}: {e}")
                all_successful = False

        except requests.exceptions.Timeout:
            print(f"  Error: Request timed out for mode '{mode}'.")
            all_successful = False
        except requests.exceptions.RequestException as e:
            print(f"  Error making request for mode '{mode}': {e}")
            # Attempt to print response body if available, might contain FastAPI error details
            if hasattr(e, 'response') and e.response is not None:
                print(f"  Response status: {e.response.status_code}")
                try:
                    print(f"  Response body: {e.response.text}")
                except Exception:
                    print("  Response body could not be decoded.")
            all_successful = False
        except json.JSONDecodeError:
             print(f"  Error: Could not decode JSON response for mode '{mode}'.")
             all_successful = False

        print("-" * 30)

    print("Live query test finished.")
    if not all_successful:
        print("Note: One or more requests or file operations failed.") 