"""
Sandbox cache management system.
Provides persistent caching for sandbox tools.
"""

import os
import json
import threading
import time
import uuid
from typing import Dict, Any, List, Optional
import logging

class SandboxCacheManager:
    """
    Manages persistent caching for sandbox tools.
    
    Args:
        cache_dir: Directory path for storing cache files.
    """
    
    def __init__(self, cache_dir: str):
        self.cache_dir = cache_dir
        self._caches: Dict[str, Dict[str, Any]] = {}
        self._locks: Dict[str, threading.Lock] = {}
        self._missed_locks: Dict[str, threading.Lock] = {}  # Separate locks for missed calls
        self._global_lock = threading.Lock()  # Global lock for creating new tool locks
        self._dirty_tools = set()  # Track which tools need saving
        self._dirty_lock = threading.Lock()  # Lock for dirty_tools set
        
        # Ensure cache directory exists
        self._ensure_cache_dir()
        
        # Setup periodic save (every 30 seconds for dirty caches)
        self._save_timer = None
        self._start_periodic_save()
    
    def _ensure_cache_dir(self):
        """Ensure cache directory exists with proper error handling."""
        try:
            os.makedirs(self.cache_dir, exist_ok=True)
        except OSError as e:
            logging.warning(f"Failed to create cache directory {self.cache_dir}: {e}")
        
    def _start_periodic_save(self):
        """Start periodic saving of dirty caches."""
        def save_dirty():
            # Get dirty tools under lock protection
            with self._dirty_lock:
                dirty_tools_copy = list(self._dirty_tools)
            
            for tool_name in dirty_tools_copy:
                self._save_tool_cache(tool_name)
                with self._dirty_lock:
                    self._dirty_tools.discard(tool_name)
            
            # Schedule next save
            self._save_timer = threading.Timer(30, save_dirty)
            self._save_timer.daemon = True
            self._save_timer.start()
        
        save_dirty()
    
    def get_cache_file_path(self, tool_name: str) -> str:
        """Get cache file path for a tool."""
        return os.path.join(self.cache_dir, f"{tool_name}_cache.json")
    
    def get_missed_calls_file_path(self, tool_name: str) -> str:
        """Get missed calls file path for a tool."""
        return os.path.join(self.cache_dir, f"{tool_name}_missed.json")
    
    def _get_cache_key(self, params: str) -> str:
        """
        Generate cache key from parameters by sorting them.
        
        Args:
            params: JSON string of parameters.
            
        Returns:
            Normalized cache key string.
        """
        try:
            params_dict = json.loads(params)
            if isinstance(params_dict, dict):
                # Sort dictionary keys for consistent ordering
                sorted_items = sorted(params_dict.items())
                return json.dumps(sorted_items, sort_keys=True, separators=(',', ':'), ensure_ascii=False)
            else:
                # If not a dict, return as-is
                return params
        except (json.JSONDecodeError, TypeError):
            # If not valid JSON, return the original string
            return params
    
    def _load_tool_cache(self, tool_name: str) -> Dict[str, Any]:
        """Load cache for a specific tool."""
        cache_file = self.get_cache_file_path(tool_name)
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                logging.warning(f"Failed to load cache for {tool_name}: {e}")
        return {}
    
    def _save_tool_cache(self, tool_name: str):
        """Save cache for a specific tool."""
        if tool_name not in self._caches:
            return
        
        # Ensure cache directory exists
        self._ensure_cache_dir()
        
        cache_file = self.get_cache_file_path(tool_name)
        temp_file = None
        try:
            # Create a copy of the cache under lock protection to avoid
            # "dictionary changed size during iteration" errors
            with self._locks[tool_name]:
                cache_copy = self._caches[tool_name].copy()
            
            # Use temporary file with unique name for atomic write
            temp_file = f"{cache_file}.{uuid.uuid4().hex}.tmp"
            with open(temp_file, 'w', encoding='utf-8') as f:
                json.dump(cache_copy, f, ensure_ascii=False, indent=2)
            os.replace(temp_file, cache_file)
            logging.debug(f"Saved cache for {tool_name}")
        except Exception as e:
            logging.error(f"Failed to save cache for {tool_name}: {e}")
            # Clean up temp file if it exists
            if temp_file and os.path.exists(temp_file):
                try:
                    os.remove(temp_file)
                except Exception:
                    pass
    
    def _ensure_tool_cache(self, tool_name: str):
        """Ensure tool cache is loaded and lock exists."""
        with self._global_lock:
            if tool_name not in self._caches:
                self._caches[tool_name] = self._load_tool_cache(tool_name)
            if tool_name not in self._locks:
                self._locks[tool_name] = threading.Lock()
            if tool_name not in self._missed_locks:
                self._missed_locks[tool_name] = threading.Lock()
    
    def get(self, tool_name: str, params: str) -> Optional[str]:
        """
        Get cached result for tool call.
        
        Args:
            tool_name: Name of the tool.
            params: JSON string of tool parameters.
            
        Returns:
            Cached result string or None if not found.
        """
        self._ensure_tool_cache(tool_name)
        cache_key = self._get_cache_key(params)
        
        with self._locks[tool_name]:
            return self._caches[tool_name].get(cache_key)
    
    def set(self, tool_name: str, params: str, result: str):
        """
        Set cached result for tool call.
        
        Args:
            tool_name: Name of the tool.
            params: JSON string of tool parameters.
            result: Result string to cache.
        """
        self._ensure_tool_cache(tool_name)
        cache_key = self._get_cache_key(params)
        
        with self._locks[tool_name]:
            self._caches[tool_name][cache_key] = result
        
        with self._dirty_lock:
            self._dirty_tools.add(tool_name)
    
    def save_missed_call(self, tool_name: str, params: str, simulated_result: str):
        """
        Save a missed call for later processing.
        
        Args:
            tool_name: Name of the tool.
            params: JSON string of tool parameters.
            simulated_result: Simulated result to save.
        """
        self._ensure_tool_cache(tool_name)
        
        # Use lock to prevent concurrent access
        with self._missed_locks[tool_name]:
            # Ensure cache directory exists (inside lock to prevent race condition)
            self._ensure_cache_dir()
            
            missed_file = self.get_missed_calls_file_path(tool_name)
            
            # Load existing missed calls
            missed_calls = []
            if os.path.exists(missed_file):
                try:
                    with open(missed_file, 'r', encoding='utf-8') as f:
                        missed_calls = json.load(f)
                except Exception as e:
                    logging.warning(f"Failed to load missed calls for {tool_name}: {e}")
            
            # Add new missed call
            missed_calls.append({
                'params': params,
                'simulated_result': simulated_result,
                'timestamp': time.time()
            })
            
            # Save updated missed calls with unique temp file name to avoid conflicts
            try:
                # Use uuid to generate unique temp file name
                temp_file = f"{missed_file}.{uuid.uuid4().hex}.tmp"
                with open(temp_file, 'w', encoding='utf-8') as f:
                    json.dump(missed_calls, f, ensure_ascii=False, indent=2)
                os.replace(temp_file, missed_file)
                logging.debug(f"Saved missed call for {tool_name}")
            except Exception as e:
                logging.error(f"Failed to save missed call for {tool_name}: {e}")
                # Clean up temp file if it exists
                try:
                    if os.path.exists(temp_file):
                        os.remove(temp_file)
                except Exception:
                    pass
    
    def get_missed_calls(self, tool_name: str) -> List[Dict[str, Any]]:
        """Get missed calls for a tool."""
        missed_file = self.get_missed_calls_file_path(tool_name)
        if os.path.exists(missed_file):
            try:
                with open(missed_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                logging.warning(f"Failed to load missed calls for {tool_name}: {e}")
        return []
    
    def clear_missed_calls(self, tool_name: str):
        """Clear missed calls for a tool."""
        missed_file = self.get_missed_calls_file_path(tool_name)
        if os.path.exists(missed_file):
            try:
                os.remove(missed_file)
                logging.debug(f"Cleared missed calls for {tool_name}")
            except Exception as e:
                logging.error(f"Failed to clear missed calls for {tool_name}: {e}")
    
    def force_save_all(self):
        """Force save all dirty caches."""
        with self._dirty_lock:
            dirty_tools_copy = list(self._dirty_tools)
        
        for tool_name in dirty_tools_copy:
            self._save_tool_cache(tool_name)
            with self._dirty_lock:
                self._dirty_tools.discard(tool_name)
    
    def __del__(self):
        """Cleanup on destruction."""
        if hasattr(self, '_save_timer') and self._save_timer:
            self._save_timer.cancel()
        self.force_save_all()

def get_cache_stats(cache_dir: str) -> Dict[str, Dict[str, Any]]:
    """Get statistics about cached data."""
    
    stats = {}
    cache_manager = SandboxCacheManager(cache_dir)
    
    if not os.path.exists(cache_dir):
        return stats
    
    for filename in os.listdir(cache_dir):
        if filename.endswith('_cache.json'):
            tool_name = filename[:-11]  # Remove '_cache.json'
            cache_manager._ensure_tool_cache(tool_name)
            
            cache_size = len(cache_manager._caches.get(tool_name, {}))
            missed_calls = len(cache_manager.get_missed_calls(tool_name))
            
            stats[tool_name] = {
                'cached_calls': cache_size,
                'missed_calls': missed_calls,
                'cache_file': cache_manager.get_cache_file_path(tool_name),
                'missed_file': cache_manager.get_missed_calls_file_path(tool_name)
            }
    
    return stats

def clear_all_missed_calls(cache_dir: str):
    """Clear all missed calls."""
    cache_manager = SandboxCacheManager(cache_dir)
    
    if not os.path.exists(cache_dir):
        return
        
    for filename in os.listdir(cache_dir):
        if filename.endswith('_missed.json'):
            tool_name = filename[:-12]  # Remove '_missed.json'
            cache_manager.clear_missed_calls(tool_name)
