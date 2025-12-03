"""
nDCG Evaluation Script for Azure AI Search with Azure OpenAI embeddings
"""
import os
import json
import math
import hashlib
import time
from typing import List, Dict
from dotenv import load_dotenv
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.models import VectorizedQuery
from openai import AzureOpenAI  # Using AzureOpenAI specifically

# ---- Configuration ----

load_dotenv()
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
if not AZURE_OPENAI_ENDPOINT:
    raise ValueError("AZURE_OPENAI_ENDPOINT must be set")
if AZURE_OPENAI_ENDPOINT.endswith("/"):
    AZURE_OPENAI_ENDPOINT = AZURE_OPENAI_ENDPOINT.rstrip("/")

AZURE_OPENAI_KEY = os.getenv("AZURE_OPENAI_API_KEY")
if not AZURE_OPENAI_KEY:
    raise ValueError("AZURE_OPENAI_API_KEY must be set")

EMBED_DEPLOYMENT = os.getenv("AZURE_OPENAI_EMBED_DEPLOYMENT")
if not EMBED_DEPLOYMENT:
    raise ValueError("AZURE_OPENAI_EMBED_DEPLOYMENT must be set to your embedding deployment name")

SEARCH_ENDPOINT = os.environ["AZURE_SEARCH_ENDPOINT"]
SEARCH_KEY = os.environ["AZURE_SEARCH_API_KEY"]
INDEX_NAME = os.environ["AZURE_SEARCH_INDEX"]

K_LIST = [1, 3, 10, 30]          # <<< EDIT HERE: set all k you want to evaluate
K = max(K_LIST)              
USE_SEMANTIC = True
SEMANTIC_CONFIG_NAME = "cms-hospital-semantic-config" # <<< EDIT HERE: set it up semantic config in the index

JUDGE_MODEL = os.environ.get("AZURE_AI_MODEL_DEPLOYMENT_NAME", "gpt-4.1-mini")
MAX_CHARS_FOR_JUDGE = 1200  # cap chunk text length for judging

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
if not os.path.exists(RESULTS_DIR):
    os.makedirs(RESULTS_DIR, exist_ok=True)

# Persistent cache file 
CACHE_FILE = os.path.join(RESULTS_DIR, "judge_cache.json")

# Timestamped report file per run
RUN_TIMESTAMP = time.strftime("%Y%m%dT%H%M%S")
REPORT_FILE = os.path.join(RESULTS_DIR, f"ndcg_report_{RUN_TIMESTAMP}.json")
REPORT_FILE_LATEST = os.path.join(RESULTS_DIR, "ndcg_report.json")
# Timestamped cache snapshot for this run.persistent CACHE_FILE continues to be updated
CACHE_FILE_TS = os.path.join(RESULTS_DIR, f"judge_cache_{RUN_TIMESTAMP}.json")
# Ensure cache file exists so first run creates it in results folder
if not os.path.exists(CACHE_FILE):
    try:
        with open(CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump({}, f, ensure_ascii=False)
        print(f"Debug - Created empty judge cache at: {CACHE_FILE}")
    except Exception as e:
        print(f"Debug - Failed to create initial cache file: {e}")

# ---- Initialize Clients ----
def create_openai_client() -> AzureOpenAI:
    """Create Azure OpenAI client with proper configuration"""
    print(f"Debug - Initializing Azure OpenAI client with endpoint: {AZURE_OPENAI_ENDPOINT}")
    print(f"Debug - Using embedding deployment: {EMBED_DEPLOYMENT}")
    
    try:
        client = AzureOpenAI(
            api_key=AZURE_OPENAI_KEY,
            api_version="2024-02-15-preview",
            azure_endpoint=AZURE_OPENAI_ENDPOINT
        )
        
        # Test the client
        print("Debug - Testing embedding...")
        response = client.embeddings.create(
            model=EMBED_DEPLOYMENT,
            input=["Test embedding functionality"]
        )
        print(f"Debug - Successfully generated test embedding of size: {len(response.data[0].embedding)}")
        return client
    except Exception as e:
        print(f"Debug - Error creating Azure OpenAI client: {str(e)}")
        print(f"Debug - Full error details: {repr(e)}")
        raise

openai_client = create_openai_client()
search_client = SearchClient(SEARCH_ENDPOINT, INDEX_NAME, AzureKeyCredential(SEARCH_KEY))

def embed_text(text: str) -> List[float]:
    """Generate embeddings for a text using Azure OpenAI"""
    try:
        response = openai_client.embeddings.create(
            model=EMBED_DEPLOYMENT,
            input=[text]
        )
        return response.data[0].embedding
    except Exception as e:
        print(f"Debug - Embedding error for text: {text[:100]}...")
        print(f"Debug - Error: {str(e)}")
        raise

def hybrid_search(query: str, k: int = K) -> List[Dict]:
    """Perform hybrid search (vector + optional semantic) on the index"""
    try:
        print(f"Debug - Performing semantic search for query: {query}")
        params = {
            "top": k,
            "select": "id,title,page,section,content",
            "query_type": "semantic",
            "semantic_configuration_name": SEMANTIC_CONFIG_NAME
        }

        print("Debug - Executing semantic search with parameters:", json.dumps(params, default=str))

        query_vector = openai_client.embeddings.create(input = [query], model=EMBED_DEPLOYMENT).data[0].embedding
        raw_vector_query = VectorizedQuery(vector=query_vector, k_nearest_neighbors=3, fields="contentVector")

        results = search_client.search(
            search_text=query,
            vector_queries=[raw_vector_query],
            **params
        )

        documents = []
        for doc in results:
            documents.append({
                "id": doc["id"],
                "title": doc.get("title", ""),
                "page": doc.get("page", ""),
                "section": doc.get("section", ""),
                "content": doc.get("content", "")
            })

        print(f"Debug - Retrieved {len(documents)} documents")
        return documents
        
    except Exception as e:
        print(f"Debug - Search error: {str(e)}")
        print(f"Debug - Full error details: {repr(e)}")
        raise

def judge_relevance(query: str, passage: str) -> int:
    """Judge the relevance of a passage to a query using Azure OpenAI"""
    system_prompt = (
        "You are a precise information retrieval judge. "
        "Given a USER query and a CANDIDATE passage, assign a single integer relevance score:\n"
        "3 = highly relevant and directly answers or is exactly on topic\n"
        "2 = relevant and useful, but not perfectly focused\n"
        "1 = partially relevant or tangential\n"
        "0 = not relevant\n"
        "Return ONLY the integer 0, 1, 2, or 3."
    )
    
    passage = passage[:MAX_CHARS_FOR_JUDGE] 
    response = openai_client.chat.completions.create(
        model=JUDGE_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"QUERY:\n{query}\n\nCANDIDATE PASSAGE:\n{passage}\n\nScore:"}
        ],
        temperature=0,
        max_tokens=4
    )
    
    text = response.choices[0].message.content.strip()
    return next((int(ch) for ch in text if ch in "0123"), 0)

# ---- Cache Management ----
def load_cache() -> Dict[str, int]:
    """Load the judge cache from file"""
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}

def save_cache(cache: Dict[str, int]):
    """Save the judge cache to file"""
    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)

def get_cache_key(query: str, doc_id: str) -> str:
    """Generate a cache key for a query-document pair"""
    return f"{hashlib.sha256(query.encode()).hexdigest()[:16]}::{doc_id}"

# ---- nDCG Calculation ----
def calculate_dcg(labels: List[int]) -> float:
    """Calculate DCG for a list of relevance labels using gains (2^rel - 1) and log2(i+1) discounts."""
    # i starts at 1; discount = log2(i+1)  
    return sum(((2**rel) - 1) / math.log2(i + 1) for i, rel in enumerate(labels, start=1))

def calculate_ndcg(labels: List[int], k: int) -> float:
    """Calculate nDCG@k for a list of relevance labels """
    if not labels or k <= 0:
        return 0.0

    # DCG on the retrieved top-k
    dcg_k = calculate_dcg(labels[:k])

    # IDCG from the best possible ordering drawn from ALL judgments available
    ideal_labels = sorted(labels, reverse=True)[:k]
    idcg_k = calculate_dcg(ideal_labels)

    return (dcg_k / idcg_k) if idcg_k > 0 else 0.0

# ---- Evaluation Queries ----
EVAL_QUERIES = [
    "acceptable standards of practice for Nuclear Medicine Services",
    "how is the weather today in New York"
] #<<< EDIT HERE: change or add your evaluation queries

def run_evaluation():
    """Run the complete evaluation process"""
    print("\n=== Starting nDCG Evaluation ===\n")
    
    # Test search client connection
    try:
        print("Debug - Testing search client connection...")
        test_results = search_client.search("test", top=1)
        list(test_results)  # Force execution of the search
        print("Debug - Search client connection successful")
    except Exception as e:
        print(f"Error - Failed to connect to search service: {str(e)}")
        raise
    
    cache = load_cache()
    results = []
    
    for query in EVAL_QUERIES:
        print(f"\nProcessing query: {query}")
        try:
            # Get search results
            hits = hybrid_search(query, k=K)
            print(f"Retrieved {len(hits)} results")
            
            if not hits:
                print(f"Warning - No results found for query: {query}")
                continue
            
            # getJudge relevance
            labels = []
            for hit in hits:
                try:
                    cache_key = get_cache_key(query, hit["id"])
                    if cache_key in cache:
                        score = cache[cache_key]
                        print(f"Using cached relevance score: {score}") # if it finds ID in cache then it will use it instead of calling the model again for the judge
                    else:
                        score = judge_relevance(query, hit["content"])
                        cache[cache_key] = score
                        print(f"New relevance score: {score}")
                    labels.append(score)
                except Exception as e:
                    print(f"Warning - Failed to judge document {hit.get('id')}: {str(e)}")
                    labels.append(0)  # Use 0 score for failed judgments
            
            # Calculate nDCG for each k in K_LIST
            ndcg_by_k = {f"ndcg@{kk}": calculate_ndcg(labels, kk) for kk in K_LIST}
            results.append({
                "query": query,
                "labels": labels,
                "metrics": ndcg_by_k
            })
            print(" | ".join([f"{k}: {v:.3f}" for k, v in ndcg_by_k.items()]))

            save_cache(cache)
            time.sleep(0.2)  
            
        except Exception as e:
            print(f"Error processing query '{query}': {str(e)}")
            print("Continuing with next query...")
    
    # Calculate and display summary
    if results:
        mean_ndcg = {f"ndcg@{kk}": 0.0 for kk in K_LIST}
        for kk in K_LIST:
            vals = [r["metrics"][f"ndcg@{kk}"] for r in results]
            mean_ndcg[f"ndcg@{kk}"] = sum(vals) / len(vals)
        print(f"\n=== Evaluation Complete ===")
        for kk in K_LIST:
            print(f"Mean ndcg@{kk}: {mean_ndcg[f'ndcg@{kk}']:.3f}")
    else:
        print("\n=== Evaluation Complete ===")
        print(f"No results found for any query.")

    report = {
        "k_list": K_LIST,
        "results": results,
        "mean_ndcg": mean_ndcg if results else {}
    }
    try:
        with open(REPORT_FILE, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

        try:
            with open(REPORT_FILE_LATEST, "w", encoding="utf-8") as f2:
                json.dump(report, f2, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"Warning - Failed to write latest report copy: {e}")
        try:
            with open(CACHE_FILE_TS, "w", encoding="utf-8") as f3:
                try:
                    current_cache = load_cache()
                except Exception:
                    current_cache = {}
                json.dump(current_cache, f3, ensure_ascii=False, indent=2)
            print(f"Timestamped judge cache saved to: {CACHE_FILE_TS}")
        except Exception as e:
            print(f"Warning - Failed to write timestamped judge cache: {e}")

        print(f"Detailed results saved to: {REPORT_FILE}")
        print(f"Latest report (overwritten): {REPORT_FILE_LATEST}")
        print(f"Judge cache saved to: {CACHE_FILE}")
    except Exception as e:
        print(f"Failed to save report or cache: {e}")

if __name__ == "__main__":
    run_evaluation()
