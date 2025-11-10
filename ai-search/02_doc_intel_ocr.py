import os
import json
import uuid
import requests
from typing import List, Dict, Any
from dotenv import load_dotenv
from tqdm import tqdm
from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from openai import AzureOpenAI

load_dotenv()

# Document Intelligence settings
DOCINTEL_ENDPOINT = os.environ["AZURE_DOCINTEL_ENDPOINT"]
DOCINTEL_KEY = os.environ["AZURE_DOCINTEL_KEY"]

# Azure Search settings
SEARCH_ENDPOINT = os.environ["AZURE_SEARCH_ENDPOINT"]
SEARCH_KEY = os.environ["AZURE_SEARCH_API_KEY"]
SEARCH_INDEX = os.environ["AZURE_SEARCH_INDEX"]

# Azure OpenAI settings
OPENAI_ENDPOINT = os.environ["AZURE_OPENAI_ENDPOINT"]
OPENAI_KEY = os.environ["AZURE_OPENAI_API_KEY"]

# File settings
PDF_URL = "https://www.cms.gov/Regulations-and-Guidance/Guidance/Manuals/downloads/som107ap_a_hospitals.pdf"
PDF_FILE = "cms_hospital_manual.pdf"
JSONL_OUT = "cms_chunks_docintel.jsonl"

# Initialize clients
doc_client = DocumentIntelligenceClient(
    endpoint=DOCINTEL_ENDPOINT,
    credential=AzureKeyCredential(DOCINTEL_KEY)
)

search_client = SearchClient(
    endpoint=SEARCH_ENDPOINT,
    index_name=SEARCH_INDEX,
    credential=AzureKeyCredential(SEARCH_KEY)
)

openai_client = AzureOpenAI(
    api_key=OPENAI_KEY,
    api_version="2023-05-15",
    azure_endpoint=OPENAI_ENDPOINT
)

def download_pdf(url, out_path):
    if not os.path.exists(out_path):
        print(f"⬇️ Downloading {url} ...")
        r = requests.get(url, timeout=300)
        r.raise_for_status()
        with open(out_path, "wb") as f:
            f.write(r.content)
        print(f"✅ Saved to {out_path}")
    else:
        print(f"{out_path} already exists — skipping download.")
    return out_path

def get_embeddings(texts: List[str]) -> List[List[float]]:
    """Generate embeddings for a list of texts using Azure OpenAI."""
    embeddings = []
    
    for text in tqdm(texts, desc="Generating embeddings"):
        response = openai_client.embeddings.create(
            model="text-embedding-3-small",  #  deployed model name
            input=text
        )
        embeddings.append(response.data[0].embedding)
    
    return embeddings

def extract_docintelligence_chunks(pdf_path: str, chunk_size: int = 1200, overlap: int = 200) -> List[Dict[str, Any]]:
    """Extract and process chunks from a PDF using Azure Document Intelligence, with overlap."""
    print(f"Processing {pdf_path} with Document Intelligence...")

    with open(pdf_path, "rb") as f:
        file_content = f.read()
        poller = doc_client.begin_analyze_document(
            model_id="prebuilt-layout",
            body=file_content)
    result = poller.result()

    chunks = []
    texts_for_embeddings = []

    for p in tqdm(result.paragraphs, desc="Extracting paragraphs"):
        if not p.content or not p.bounding_regions:
            continue

        page_no = p.bounding_regions[0].page_number
        text = p.content.strip()

        # chunk long paragraphs with overlap
        start = 0
        chunk_num = 1
        while start < len(text):
            end = start + chunk_size
            segment = text[start:end]
            if segment.strip():
                doc = {
                    "id": f"cms-{page_no}-{uuid.uuid4().hex[:8]}",
                    "source": "CMS_SOM_Hospitals",
                    "title": "CMS State Operations Manual - Appendix A (Hospitals)",
                    "page": page_no,
                    "section": f"chunk_{chunk_num}",
                    "content": segment
                }
                chunks.append(doc)
                texts_for_embeddings.append(segment)
                chunk_num += 1
            start += chunk_size - overlap  # chunk_size minus overlap

    # Generate embeddings for all chunks
    print("\nGenerating embeddings for chunks...")
    embeddings = get_embeddings(texts_for_embeddings)

    for chunk, embedding in zip(chunks, embeddings):
        chunk["contentVector"] = embedding

    return chunks

def upload_to_search(documents: List[Dict[str, Any]], batch_size: int = 50):
    """Upload documents to Azure Search in batches."""
    print(f"\nUploading {len(documents)} documents to search index...")
    
    # Upload in batches
    for i in tqdm(range(0, len(documents), batch_size), desc="Uploading batches"):
        batch = documents[i:i + batch_size]
        try:
            results = search_client.upload_documents(documents=batch)
            success_count = sum(1 for r in results if r.succeeded)
            if success_count < len(results):
                print(f"\nWarning: {len(results) - success_count} documents in the batch failed to upload")
                for result in results:
                    if not result.succeeded:
                        print(f"Document {result.key} failed: {result.error_message}")
        except Exception as e:
            print(f"\nError uploading batch: {str(e)}")

if __name__ == "__main__":
    # Validate environment variables
    required_vars = [
        "AZURE_DOCINTEL_ENDPOINT", "AZURE_DOCINTEL_KEY",
        "AZURE_SEARCH_ENDPOINT", "AZURE_SEARCH_API_KEY", "AZURE_SEARCH_INDEX",
        "AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_API_KEY"
    ]
    
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    if missing_vars:
        raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")
    
    try:
        # Download PDF if needed - goes to root. check that the URL is still working
        download_pdf(PDF_URL, PDF_FILE)
        
        # Process PDF and generate chunks with embeddings
        chunks = extract_docintelligence_chunks(PDF_FILE)
        
        # Save chunks to JSONL
        with open(JSONL_OUT, "w", encoding="utf-8") as f:
            for c in chunks:
                # Create a copy without the embedding for JSONL storage
                chunk_for_json = c.copy()
                chunk_for_json.pop("contentVector", None)
                f.write(json.dumps(chunk_for_json, ensure_ascii=False) + "\n")
        print(f"✅ Wrote {len(chunks)} chunks to {JSONL_OUT}")
        
        # Upload chunks to Azure Search
        upload_to_search(chunks)
        print("\n✅ Processing and indexing complete!")
    except Exception as e:
        print(f"\n❌ Error during processing: {str(e)}")
        raise
