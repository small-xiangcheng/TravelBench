#!/usr/bin/env python3
"""
Precompute embeddings for all cached tool calls using remote embedding service.
Run this script before starting the system to prepare embeddings for fast similarity search.

Usage:
    python -m travelbench.simulators.precompute_embeddings --cache-dir /path/to/cache
    
Environment Variables:
    EMBEDDING_SERVICE_URL: URL of the embedding service (e.g., http://localhost:8001/v1)
    EMBEDDING_MODEL_NAME: Model name for the embedding service (e.g., embedding)
"""

import os
import json
import logging
from typing import Optional

import numpy as np
from openai import OpenAI

from ..core.tools import get_sandbox_cache_manager
from ..core.config import DEFAULT_CACHE_DIR

def precompute_embeddings(cache_dir: Optional[str] = None,
                         embedding_service_url: str = None,
                         embedding_model_name: str = None,
                         batch_size: int = 1024):
    """
    Precompute embeddings for all cached tool calls using remote embedding service.
    This should be run once before starting the system to prepare embeddings.
    
    Args:
        cache_dir: Directory containing tool caches
        embedding_service_url: URL of the embedding service
        embedding_model_name: Model name for the embedding service
        batch_size: Batch size for encoding (larger = faster but more memory)
    
    Returns:
        Dict with statistics about precomputed embeddings
    """
    logger = logging.getLogger(__name__)
    logger.info("Starting embedding precomputation using remote API...")
    
    # Read from environment variables if not provided
    if embedding_service_url is None:
        embedding_service_url = os.getenv('EMBEDDING_SERVICE_URL')
    if embedding_model_name is None:
        embedding_model_name = os.getenv('EMBEDDING_MODEL_NAME')
    
    if not embedding_service_url or not embedding_model_name:
        raise ValueError(
            "Embedding service configuration not found. Please set environment variables:\n"
            "  export EMBEDDING_SERVICE_URL='http://localhost:8001/v1'\n"
            "  export EMBEDDING_MODEL_NAME='embedding'"
        )
    
    logger.info(f"Using embedding service: {embedding_service_url}")
    logger.info(f"Model name: {embedding_model_name}")
    
    # Initialize OpenAI client for embedding service
    client = OpenAI(
        api_key="dummy-key",  # vLLM doesn't require a real API key
        base_url=embedding_service_url
    )
    
    # Use DEFAULT_CACHE_DIR if cache_dir is not provided
    cache_dir = cache_dir or DEFAULT_CACHE_DIR
    cache_manager = get_sandbox_cache_manager(cache_dir)
    
    stats = {}
    
    # Process each tool's cache
    for filename in os.listdir(cache_dir):
        if filename.endswith('_cache.json'):
            tool_name = filename[:-11]  # Remove '_cache.json'
            logger.info(f"Processing {tool_name}...")
            
            try:
                # Load tool cache
                cache_manager._ensure_tool_cache(tool_name)
                cache_entries = cache_manager._caches.get(tool_name, {})
                
                if not cache_entries:
                    logger.warning(f"No cache entries for {tool_name}")
                    continue
                
                # Extract params and results
                params_list = []
                results_list = []
                
                for cache_key, result in cache_entries.items():
                    # Parse cache_key to extract params
                    try:
                        parsed_key = json.loads(cache_key)
                        params_str = None
                        
                        if isinstance(parsed_key, list):
                            for item in parsed_key:
                                if isinstance(item, list) and len(item) == 2:
                                    if item[0] == "params":
                                        params_str = item[1]
                                        break
                        
                        if params_str:
                            params_list.append(params_str)
                            results_list.append(result)
                    except (json.JSONDecodeError, TypeError, IndexError):
                        continue
                
                if not params_list:
                    logger.warning(f"No valid params found for {tool_name}")
                    continue
                
                # Convert params to text for embedding
                def params_to_text(params_str):
                    try:
                        params_dict = json.loads(params_str)
                        text_parts = []
                        for key, value in params_dict.items():
                            text_parts.append(f"{key}: {value}")
                        return " | ".join(text_parts)
                    except:
                        return params_str
                
                texts = [params_to_text(p) for p in params_list]
                
                # Compute embeddings in batches using API
                logger.info(f"Computing embeddings for {len(texts)} examples...")
                all_embeddings = []
                
                for i in range(0, len(texts), batch_size):
                    batch = texts[i:i+batch_size]
                    logger.info(f"Processing batch {i//batch_size + 1}/{(len(texts)-1)//batch_size + 1}")
                    
                    try:
                        response = client.embeddings.create(
                            input=batch,
                            model=embedding_model_name
                        )
                        batch_embeddings = [item.embedding for item in response.data]
                        all_embeddings.extend(batch_embeddings)
                    except Exception as e:
                        logger.error(f"Failed to get embeddings for batch: {e}")
                        raise
                
                embeddings = np.array(all_embeddings, dtype=np.float32)
                
                # Save embeddings
                output_file = os.path.join(cache_dir, f"{tool_name}_embeddings.npz")
                np.savez_compressed(
                    output_file,
                    embeddings=embeddings,
                    params=np.array(params_list, dtype=object),
                    results=np.array(results_list, dtype=object)
                )
                
                stats[tool_name] = {
                    'num_embeddings': len(embeddings),
                    'embedding_dim': embeddings.shape[1],
                    'file_size_mb': os.path.getsize(output_file) / (1024 * 1024)
                }
                
                logger.info(f"Saved {len(embeddings)} embeddings for {tool_name} "
                          f"({stats[tool_name]['file_size_mb']:.2f} MB)")
                
            except Exception as e:
                logger.error(f"Failed to process {tool_name}: {e}")
                continue
    
    logger.info(f"Precomputation complete. Processed {len(stats)} tools.")
    return stats

if __name__ == '__main__':
    import argparse
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    parser = argparse.ArgumentParser(
        description='Precompute embeddings for cached tool calls'
    )
    parser.add_argument(
        '--cache-dir',
        default=DEFAULT_CACHE_DIR,
        help='Directory containing tool caches (default: $SANDBOX_CACHE_DIR or ./sandbox_cache)'
    )
    parser.add_argument(
        '--service-url',
        default=None,
        help='Embedding service URL (default: read from EMBEDDING_SERVICE_URL env var)'
    )
    parser.add_argument(
        '--model-name',
        default=None,
        help='Embedding model name (default: read from EMBEDDING_MODEL_NAME env var)'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=1024,
        help='Batch size for encoding (default: 1024)'
    )
    
    args = parser.parse_args()
    
    # Get configuration from args or environment
    service_url = args.service_url or os.getenv('EMBEDDING_SERVICE_URL')
    model_name = args.model_name or os.getenv('EMBEDDING_MODEL_NAME')
    
    print("\n" + "="*70)
    print("Precomputing Embeddings for Tool Simulator (Remote API)")
    print("="*70)
    print(f"Cache directory: {args.cache_dir}")
    print(f"Embedding service: {service_url or 'Not set'}")
    print(f"Model name: {model_name or 'Not set'}")
    print(f"Batch size: {args.batch_size}")
    print("="*70 + "\n")
    
    try:
        stats = precompute_embeddings(
            cache_dir=args.cache_dir,
            embedding_service_url=service_url,
            embedding_model_name=model_name,
            batch_size=args.batch_size
        )
        
        print("\n" + "="*70)
        print("📊 Precomputation Statistics")
        print("="*70)
        
        total_embeddings = 0
        total_size_mb = 0
        
        for tool_name, tool_stats in sorted(stats.items()):
            print(f"\n{tool_name}:")
            print(f"  ✓ Embeddings: {tool_stats['num_embeddings']:,}")
            print(f"  ✓ Dimension: {tool_stats['embedding_dim']}")
            print(f"  ✓ File size: {tool_stats['file_size_mb']:.2f} MB")
            
            total_embeddings += tool_stats['num_embeddings']
            total_size_mb += tool_stats['file_size_mb']
        
        print("\n" + "-"*70)
        print(f"Total: {len(stats)} tools, {total_embeddings:,} embeddings, {total_size_mb:.2f} MB")
        print("="*70)
        
        print("\n✅ Precomputation complete!")
        print(f"📁 Embeddings saved to: {args.cache_dir}/*_embeddings.npz")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
