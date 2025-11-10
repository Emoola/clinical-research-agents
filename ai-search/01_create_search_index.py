#!/usr/bin/env python3
"""
A simple script to create or update an Azure AI Search index.
Requires the following environment variables:
- AZURE_SEARCH_ENDPOINT: Your Azure AI Search service endpoint
- AZURE_SEARCH_API_KEY: Your Azure AI Search admin API key
- AZURE_SEARCH_INDEX: Name of the index to create/update (optional, defaults to 'default-index')
"""

import os
from typing import Optional
from dotenv import load_dotenv
from azure.core.credentials import AzureKeyCredential
from azure.search.documents.indexes import SearchIndexClient
from azure.core.exceptions import ResourceNotFoundError
from azure.search.documents.indexes.models import (
    SearchIndex,
    SimpleField,
    SearchableField,
    SearchFieldDataType,
    VectorSearch,
    HnswAlgorithmConfiguration,
    CorsOptions,
)

def load_environment():
    """Load and validate environment variables."""
    load_dotenv()
    
    endpoint = os.getenv("AZURE_SEARCH_ENDPOINT")
    api_key = os.getenv("AZURE_SEARCH_API_KEY")
    index_name = os.getenv("AZURE_SEARCH_INDEX", "default-index")
    
    if not endpoint or not api_key:
        raise ValueError(
            "Missing required environment variables. Please ensure AZURE_SEARCH_ENDPOINT "
            "and AZURE_SEARCH_API_KEY are set in your .env file."
        )
    
    return endpoint, api_key, index_name

def create_index(endpoint: str, key: str, index_name: str, vector_dimension: int = 1536):
    """Create a new search index with vector search capability.
    
    Args:
        endpoint: Azure Search service endpoint
        key: Azure Search API key
        index_name: Name of the index to create
        vector_dimension: Dimension of the vector embeddings (default: 1536 for text-embedding-3-small)
    """
    client = SearchIndexClient(endpoint, AzureKeyCredential(key))
    
    # Define the index using raw definition that matches Azure Search REST API
    # Define the index definition
    index_definition = {
        "name": index_name,
        "fields": [
            {
                "name": "id",
                "type": "Edm.String",
                "key": True,
                "filterable": True
            },
            {
                "name": "source",
                "type": "Edm.String",
                "filterable": True,
                "facetable": True
            },
            {
                "name": "title",
                "type": "Edm.String",
                "searchable": True,
                "filterable": True,
                "sortable": True
            },
            {
                "name": "page",
                "type": "Edm.Int32",
                "filterable": True,
                "sortable": True
            },
            {
                "name": "section",
                "type": "Edm.String",
                "searchable": True
            },
            {
                "name": "content",
                "type": "Edm.String",
                "searchable": True
            },
            {
                "name": "contentVector",
                "type": "Collection(Edm.Single)",
                "searchable": True,
                "filterable": False,
                "sortable": False,
                "facetable": False,
                "dimensions": vector_dimension,
                "vectorSearchConfiguration": "default-config",
                "vectorSearchProfile": "default-profile"
            }
        ],
        "vectorSearch": {
            "algorithms": [
                {
                    "name": "default-config",
                    "kind": "hnsw",
                    "parameters": {
                        "m": 4,
                        "efConstruction": 400,
                        "efSearch": 500,
                        "metric": "cosine"
                    }
                }
            ],
            "profiles": [
                {
                    "name": "default-profile",
                    "algorithm": "default-config"
                }
            ]
        },
        "corsOptions": {
            "allowedOrigins": ["*"],
            "maxAgeInSeconds": 60
        }
    }
    # Create the index
    try:
        print(f"Creating index '{index_name}'...")
        index = SearchIndex.deserialize(index_definition)
        client.create_or_update_index(index)
        print(f"Successfully created/updated index '{index_name}'")
        
        # Print index details
        created_index = client.get_index(index_name)
        print("\nIndex details:")
        print(f"Name: {created_index.name}")
        print(f"Fields: {[field.name for field in created_index.fields]}")
        print("Vector search enabled with dimension:", vector_dimension)
            
    except Exception as e:
        print(f"Error creating/updating index: {str(e)}")
        raise

def main():
    """Main entry point."""
    # Load environment variables
    endpoint, api_key, index_name = load_environment()
    
    # Create index with vector search
    create_index(
        endpoint=endpoint,
        key=api_key,
        index_name=index_name,
        vector_dimension=1536  # Using text-embedding-3-small dimension
    )

if __name__ == "__main__":
    main()