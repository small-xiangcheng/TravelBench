import os
import json
import random
import logging
import threading
from typing import Dict, Any, List, Optional, Tuple, Iterable

import numpy as np

from ..core.openai_client import OpenAIClient
from ..core.config import OpenAIConfig
from ..core.messages import Message, UserMessage, SystemMessage
from ..core.tools import get_sandbox_cache_manager

# Try to import FAISS for fast similarity search
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    logging.warning("faiss not available. Will use numpy for similarity search (slower).")
from .prompt import (
    TOOL_SIMULATION_SYSTEM_PROMPT,
    EXAMPLES_SECTION_TEMPLATE,
    SINGLE_EXAMPLE_TEMPLATE,
    NO_EXAMPLES_TEMPLATE,
    TOOL_SIMULATION_USER_PROMPT,
    TOOLS_SCHEMAS,
)


def pick_tools_by_name(tool_schemas: List[Dict[str, Any]], names: Iterable[str]) -> List[Dict[str, Any]]:
    """
    Filter tool schemas by tool names.
    
    Args:
        tool_schemas: List of tool schema dicts with format {"type":"function","function":{"name":...}}
        names: Iterable of tool names to filter by.
        
    Returns:
        Filtered list of tool schemas (preserves order, ignores non-existent names).
    """
    name_set = set(names)
    return [
        item for item in tool_schemas
        if item.get("type") == "function"
        and isinstance(item.get("function"), dict)
        and item["function"].get("name") in name_set
    ]




class LLMToolSimulator:
    """
    LLM-based tool simulator that generates realistic tool responses.
    
    Uses cached examples from real tool calls to enhance simulation accuracy.
    Supports both random and similarity-based example retrieval.
    """
    
    def __init__(self, 
                 openai_config: OpenAIConfig,
                 cache_dir: Optional[str] = None,
                 max_examples: int = 8,
                 simulation_temperature: float = 0,
                 use_similarity_retrieval: bool = False,
                 embedding_config: Optional[OpenAIConfig] = None):
        """
        Initialize the LLM tool simulator.
        
        Args:
            openai_config: OpenAI configuration for API calls
            cache_dir: Directory containing tool caches
            max_examples: Maximum number of examples to include in prompt
            simulation_temperature: Temperature for simulation (higher = more creative)
            use_similarity_retrieval: Whether to use similarity-based retrieval (default: False)
            embedding_config: OpenAI configuration for remote embedding service (optional)
        """
        self.openai_client = OpenAIClient(openai_config)
        self.cache_manager = get_sandbox_cache_manager(cache_dir)
        self.cache_dir = cache_dir
        self.max_examples = max_examples
        self.simulation_temperature = simulation_temperature
        self.use_similarity_retrieval = use_similarity_retrieval
        
        # Configure logging
        self.logger = logging.getLogger(__name__)
        
        # Initialize embedding client and FAISS indices if similarity retrieval is enabled
        self.embedding_client = None
        self.faiss_indices: Dict[str, faiss.Index] = {}  # tool_name -> FAISS index
        self.params_mapping: Dict[str, List[str]] = {}  # tool_name -> list of params strings (index mapping)
        self.results_mapping: Dict[str, List[str]] = {}  # tool_name -> list of results (index mapping)
        self.indices_lock = threading.Lock()  # Lock for thread-safe index access
        
        if self.use_similarity_retrieval:
            if embedding_config is None:
                self.logger.warning(
                    "Similarity retrieval requested but no embedding_config provided. "
                    "Falling back to random retrieval."
                )
                self.use_similarity_retrieval = False
            else:
                try:
                    self.logger.info(f"Initializing remote embedding client: {embedding_config.api_base}")
                    self.embedding_client = OpenAIClient(embedding_config)
                    self.logger.info("Remote embedding client initialized successfully")
                    
                    # Load precomputed embeddings and build FAISS indices
                    self._load_all_embeddings()
                except Exception as e:
                    self.logger.error(f"Failed to initialize embedding client: {e}")
                    self.logger.warning("Falling back to random retrieval")
                    self.use_similarity_retrieval = False
        
    def simulate_tool_response(self, tool_name: str, params: str) -> str:
        """
        Simulate a tool response using LLM based on cached examples.
        
        Args:
            tool_name: Name of the tool to simulate
            params: JSON string containing tool parameters
            
        Returns:
            Simulated tool response (raw output, not JSON wrapped)
        """
        try:
            # Parse parameters for context
            try:
                params_dict = json.loads(params)
            except json.JSONDecodeError:
                params_dict = {"raw_params": params}
            # Get examples from cache (pass params for similarity-based retrieval)
            examples = self._get_cache_examples(tool_name, query_params=params)
            # Get tool schema
            tool_schema = self._get_tool_schema(tool_name)
            
            # Build simulation prompt
            messages = self._build_simulation_prompt(tool_name, tool_schema, params_dict, examples)
            # Call LLM for simulation
            response, usage_info = self.openai_client.generate_response(
                messages=messages,
                temperature=self.simulation_temperature
            )
            
            # Return raw simulated result - handle different content types
            if response.content is None:
                simulated_result = ""
            elif isinstance(response.content, str):
                simulated_result = response.content.strip()
            elif isinstance(response.content, list):
                # Convert list to JSON string
                simulated_result = json.dumps(response.content, ensure_ascii=False)
            else:
                # Fallback: convert to string
                simulated_result = str(response.content)
            
            self.logger.info(f"Successfully simulated {tool_name} with {len(examples)} examples")
            return simulated_result
            
        except Exception as e:
            self.logger.error(f"Failed to simulate {tool_name}: {str(e)}")
            # Return error message directly
            return f"Error: LLM simulation failed for {tool_name}: {str(e)}"
    
    def _get_cache_examples(self, tool_name: str, query_params: Optional[str] = None) -> List[Tuple[str, str]]:
        """
        Get examples from tool cache using random or similarity-based retrieval.
        
        Args:
            tool_name: Name of the tool
            query_params: Query parameters for similarity-based retrieval (optional)
            
        Returns:
            List of (params, result) tuples
        """
        try:
            # Ensure cache is loaded
            self.cache_manager._ensure_tool_cache(tool_name)
            
            # Get cached entries
            if tool_name not in self.cache_manager._caches:
                return []
            
            cache_entries = self.cache_manager._caches[tool_name]
            if not cache_entries:
                return []
            
            # Convert cache entries to examples
            examples = []
            for cache_key, result in cache_entries.items():
                # Extract params from cache_key, filtering out 'time' field
                # cache_key format: '[["params","..."],["time","..."]]'
                try:
                    parsed_key = json.loads(cache_key)
                    params_str = None
                    
                    # Extract only the 'params' field
                    if isinstance(parsed_key, list):
                        for item in parsed_key:
                            if isinstance(item, list) and len(item) == 2:
                                if item[0] == "params":
                                    params_str = item[1]
                                    break
                    
                    # If we successfully extracted params, use it; otherwise use original cache_key
                    if params_str:
                        examples.append((params_str, result))
                    else:
                        # Fallback to original cache_key if parsing fails
                        examples.append((cache_key, result))
                        
                except (json.JSONDecodeError, TypeError, IndexError):
                    # If parsing fails, use original cache_key
                    examples.append((cache_key, result))
            # Select examples based on retrieval mode
            if self.use_similarity_retrieval and query_params and self.embedding_client:
                examples = self._get_similar_examples(tool_name, query_params, examples)
            else:
                # Randomly sample up to max_examples
                if len(examples) > self.max_examples:
                    examples = random.sample(examples, self.max_examples)
            
            self.logger.debug(f"Retrieved {len(examples)} examples for {tool_name}")
            return examples
            
        except Exception as e:
            self.logger.warning(f"Failed to get examples for {tool_name}: {str(e)}")
            return []
    
    def _load_all_embeddings(self):
        """
        Load all precomputed embeddings and build FAISS indices.
        This is called once during initialization for fast runtime retrieval.
        """
        self.logger.info("Loading precomputed embeddings and building FAISS indices...")
        for filename in os.listdir(self.cache_dir):
            if filename.endswith('_embeddings.npz'):
                tool_name = filename[:-15]  # Remove '_embeddings.npz'
                embeddings_file = os.path.join(self.cache_dir, filename)
                
                try:
                    # Load precomputed embeddings
                    data = np.load(embeddings_file, allow_pickle=True)
                    embeddings = data['embeddings']  # (N, D) array
                    params_list = data['params'].tolist()  # List of params strings
                    results_list = data['results'].tolist()  # List of results
                    
                    # Build FAISS index
                    dimension = embeddings.shape[1]
                    if FAISS_AVAILABLE:
                        # Use FAISS for fast similarity search
                        index = faiss.IndexFlatIP(dimension)  # Inner product (cosine similarity with normalized vectors)
                        # Normalize embeddings for cosine similarity
                        faiss.normalize_L2(embeddings)
                        index.add(embeddings.astype('float32'))
                        self.faiss_indices[tool_name] = index
                    else:
                        # Fallback: store normalized embeddings for numpy-based search
                        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
                        embeddings = embeddings / norms
                        self.faiss_indices[tool_name] = embeddings  # Store as numpy array
                    
                    self.params_mapping[tool_name] = params_list
                    self.results_mapping[tool_name] = results_list
                    
                    self.logger.info(f"Loaded {len(params_list)} embeddings for {tool_name}")
                    
                except Exception as e:
                    self.logger.warning(f"Failed to load embeddings for {tool_name}: {e}")
        
        self.logger.info(f"Loaded embeddings for {len(self.faiss_indices)} tools")
    
    def _get_similar_examples(self, tool_name: str, query_params: str, 
                            all_examples: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
        """
        Get most similar examples using FAISS-accelerated similarity search.
        
        Args:
            tool_name: Name of the tool
            query_params: Query parameters to find similar examples for
            all_examples: All available examples (not used, kept for compatibility)
            
        Returns:
            List of most similar (params, result) tuples
        """
        try:
            # Check if we have precomputed embeddings for this tool
            if tool_name not in self.faiss_indices:
                self.logger.warning(f"No precomputed embeddings for {tool_name}, falling back to random")
                if len(all_examples) > self.max_examples:
                    return random.sample(all_examples, self.max_examples)
                return all_examples
            
            # Compute embedding for query params using remote service
            query_text = self._params_to_text(query_params)
            embeddings_list, _ = self.embedding_client.create_embeddings(query_text)
            query_embedding = np.array(embeddings_list[0], dtype=np.float32).reshape(1, -1)
            
            # Normalize for cosine similarity
            faiss.normalize_L2(query_embedding)
            
            # Search using FAISS (thread-safe read operation)
            with self.indices_lock:
                if FAISS_AVAILABLE:
                    # FAISS search
                    k = min(self.max_examples, self.faiss_indices[tool_name].ntotal)
                    distances, indices = self.faiss_indices[tool_name].search(query_embedding, k)
                    indices = indices[0]  # Get first row
                    distances = distances[0]
                else:
                    # Numpy fallback
                    embeddings = self.faiss_indices[tool_name]
                    similarities = np.dot(embeddings, query_embedding.T).flatten()
                    k = min(self.max_examples, len(similarities))
                    indices = np.argsort(similarities)[::-1][:k]
                    distances = similarities[indices]
                
                # Get corresponding params and results
                top_examples = []
                for idx, dist in zip(indices, distances):
                    params = self.params_mapping[tool_name][idx]
                    result = self.results_mapping[tool_name][idx]
                    top_examples.append((params, result))
            
            self.logger.debug(
                f"Selected {len(top_examples)} similar examples for {tool_name} "
                f"(top similarity: {distances[0]:.3f})"
            )
            
            return top_examples
            
        except Exception as e:
            self.logger.warning(f"Similarity retrieval failed: {e}. Falling back to random sampling.")
            # Fallback to random sampling
            if len(all_examples) > self.max_examples:
                return random.sample(all_examples, self.max_examples)
            return all_examples
    
    def _params_to_text(self, params_str: str) -> str:
        """
        Convert parameters to text for embedding.
        
        Args:
            params_str: JSON string of parameters
            
        Returns:
            Text representation of parameters
        """
        try:
            params_dict = json.loads(params_str)
            # Create a readable text representation
            text_parts = []
            for key, value in params_dict.items():
                text_parts.append(f"{key}: {value}")
            return " | ".join(text_parts)
        except (json.JSONDecodeError, TypeError):
            # If not valid JSON, return as-is
            return params_str
    
    def _get_tool_schema(self, tool_name: str) -> dict:
        """
        Get tool schema by name from TOOLS_SCHEMAS.
        
        Args:
            tool_name: Name of the tool
            
        Returns:
            Tool schema dict
        """
        tools = pick_tools_by_name(TOOLS_SCHEMAS, [tool_name])
        if tools:
            return tools[0]
        return {'function': {'name': tool_name, 'description': '', 'parameters': {}}}
    
    def _build_simulation_prompt(self, 
                                tool_name: str,
                                tool_schema: dict,
                                params: Dict[str, Any], 
                                examples: List[Tuple[str, str]]) -> List[Message]:
        """
        Build the prompt for LLM simulation.
        
        Args:
            tool_name: Name of the tool to simulate
            tool_schema: Tool schema definition
            params: Tool parameters
            examples: List of (params, result) examples
            
        Returns:
            List of messages for the LLM
        """
        # Extract tool information from schema
        tool_function = tool_schema.get('function', {})
        tool_description = tool_function.get('description', '')
        tool_parameters = tool_function.get('parameters', {})
        
        # Build system prompt using template
        system_content = TOOL_SIMULATION_SYSTEM_PROMPT.format(
            tool_name=tool_name,
            tool_description=tool_description,
            tool_parameters=tool_parameters
        )
        
        # Add examples section if available
        if examples:
            system_content += EXAMPLES_SECTION_TEMPLATE.format(
                num_examples=len(examples[:self.max_examples])
            )
            for i, (example_params, example_result) in enumerate(examples[:self.max_examples], 1):
                system_content += SINGLE_EXAMPLE_TEMPLATE.format(
                    index=i,
                    params=example_params,
                    result=example_result
                )
        else:
            system_content += NO_EXAMPLES_TEMPLATE.format(tool_name=tool_name)
        
        # Build user prompt using template
        user_content = TOOL_SIMULATION_USER_PROMPT.format(
            tool_name=tool_name,
            params_json=json.dumps(params, ensure_ascii=False, indent=2)
        )
        
        messages = [
            SystemMessage(content=system_content),
            UserMessage(content=user_content)
        ]
        
        return messages


# Convenience function for easy usage
def create_llm_simulator(tool_simulator_config: OpenAIConfig,
                        cache_dir: Optional[str] = None,
                        max_examples: int = 8,
                        use_similarity_retrieval: bool = False,
                        embedding_service_url: Optional[str] = None,
                        embedding_model_name: Optional[str] = None) -> LLMToolSimulator:
    """
    Create an LLM simulator with custom parameters.
    
    Args:
        tool_simulator_config: OpenAIConfig for the tool simulator (contains api_key, api_base, model_name, temperature, max_tokens, etc.)
        cache_dir: Directory containing tool caches
        max_examples: Maximum number of examples to include in prompt
        use_similarity_retrieval: Whether to use similarity-based retrieval (default: False)
        embedding_service_url: URL of remote embedding service (optional)
        embedding_model_name: Model name for embedding service (optional)
        
    Returns:
        Configured LLMToolSimulator instance
    """
    # Create embedding config if similarity retrieval is enabled
    embedding_config = None
    if use_similarity_retrieval:
        # Get embedding service configuration from parameters or environment
        final_embedding_url = embedding_service_url or os.getenv('EMBEDDING_SERVICE_URL')
        final_embedding_model = embedding_model_name or os.getenv('EMBEDDING_MODEL_NAME')
        
        # Only create embedding config if both URL and model name are provided
        if final_embedding_url and final_embedding_model:
            embedding_config = OpenAIConfig(
                api_key="Not used",
                api_base=final_embedding_url,
                model_name=final_embedding_model,
                temperature=0.0
            )
        else:
            logging.warning(
                "Similarity retrieval requested but embedding service URL or model name not provided. "
                "Falling back to random retrieval."
            )
    
    return LLMToolSimulator(
        openai_config=tool_simulator_config,
        cache_dir=cache_dir,
        max_examples=max_examples,
        simulation_temperature=tool_simulator_config.temperature,
        use_similarity_retrieval=use_similarity_retrieval,
        embedding_config=embedding_config
    )

if __name__ == '__main__':
    import logging
    
    logging.basicConfig(level=logging.INFO)
    
    # Test simulator
    print("\n" + "="*60)
    print("Testing Tool Simulator")
    print("="*60)
    
    cache_dir = None  # Will use DEFAULT_CACHE_DIR
    
    # Test random retrieval
    print("\n[1] Testing Random Retrieval")
    simulator_random = create_llm_simulator(
        model_name="gpt-41-0414-global",
        use_similarity_retrieval=False,
        cache_dir=cache_dir
    )
    
    tool_name = "travel_search_flights"
    test_params = json.dumps({
        "origin": "北京",
        "destination": "上海",
        "date": "2025-12-15",
        "days": 5
    }, ensure_ascii=False)

    print(f"Simulating {tool_name}...")
    result = simulator_random.simulate_tool_response(tool_name, test_params)
    print(f"Result: {result[:200]}...")

    print("\n[2] Testing FAISS-Accelerated Similarity Retrieval")
    
    simulator_similarity = create_llm_simulator(
        model_name="gpt-41-0414-global",
        use_similarity_retrieval=True,
        cache_dir=cache_dir
    )
    
    print(f"Simulating {tool_name} with similarity search...")
    result = simulator_similarity.simulate_tool_response(tool_name, test_params)
    print(f"Result: {result[:200]}...")
    
    if FAISS_AVAILABLE:
        print("\n✅ Using FAISS for fast similarity search")
    else:
        print("\n⚠️  FAISS not available, using numpy fallback")

