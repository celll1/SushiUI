"""Hash cache utility for caching file hashes with mtime validation"""

import os
import json
import hashlib
from typing import Optional
from pathlib import Path


class HashCache:
    """Cache file hashes to avoid recalculating them on every model load"""

    def __init__(self, cache_file: str = "model_hash_cache.json"):
        """Initialize hash cache

        Args:
            cache_file: Path to cache file (relative to backend directory)
        """
        # Store cache in backend directory
        backend_dir = Path(__file__).parent.parent
        self.cache_path = backend_dir / cache_file
        self.cache = self._load_cache()

    def _load_cache(self) -> dict:
        """Load cache from file"""
        if self.cache_path.exists():
            try:
                with open(self.cache_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                print(f"[HashCache] Failed to load cache: {e}")
                return {}
        return {}

    def _save_cache(self):
        """Save cache to file"""
        try:
            with open(self.cache_path, 'w', encoding='utf-8') as f:
                json.dump(self.cache, f, indent=2)
        except Exception as e:
            print(f"[HashCache] Failed to save cache: {e}")

    def get_hash(self, file_path: str, algorithm: str = "sha256") -> Optional[str]:
        """Get cached hash if file hasn't changed

        Args:
            file_path: Path to file
            algorithm: Hash algorithm (sha256, sha1, md5)

        Returns:
            Cached hash if valid, None if cache miss or file changed
        """
        if not os.path.exists(file_path):
            return None

        # Get file modification time
        mtime = os.path.getmtime(file_path)
        file_size = os.path.getsize(file_path)

        # Create cache key
        cache_key = f"{file_path}:{algorithm}"

        # Check cache
        if cache_key in self.cache:
            cached_entry = self.cache[cache_key]
            cached_mtime = cached_entry.get("mtime")
            cached_size = cached_entry.get("size")

            # Validate cache: both mtime and size must match
            if cached_mtime == mtime and cached_size == file_size:
                return cached_entry.get("hash")

        return None

    def set_hash(self, file_path: str, hash_value: str, algorithm: str = "sha256"):
        """Store hash in cache

        Args:
            file_path: Path to file
            hash_value: Computed hash value
            algorithm: Hash algorithm used
        """
        if not os.path.exists(file_path):
            return

        # Get file metadata
        mtime = os.path.getmtime(file_path)
        file_size = os.path.getsize(file_path)

        # Create cache key
        cache_key = f"{file_path}:{algorithm}"

        # Store in cache
        self.cache[cache_key] = {
            "hash": hash_value,
            "mtime": mtime,
            "size": file_size,
            "algorithm": algorithm
        }

        # Save to disk
        self._save_cache()

    def calculate_and_cache(self, file_path: str, algorithm: str = "sha256") -> str:
        """Calculate hash and cache it, or return cached hash if valid

        Args:
            file_path: Path to file
            algorithm: Hash algorithm to use

        Returns:
            Hash value (from cache or newly calculated)
        """
        # Try to get from cache first
        cached_hash = self.get_hash(file_path, algorithm)
        if cached_hash:
            print(f"[HashCache] Using cached hash for {os.path.basename(file_path)}")
            return cached_hash

        # Calculate new hash
        print(f"[HashCache] Calculating new hash for {os.path.basename(file_path)}")
        hash_value = self._calculate_file_hash(file_path, algorithm)

        # Cache it
        self.set_hash(file_path, hash_value, algorithm)

        return hash_value

    def _calculate_file_hash(self, file_path: str, algorithm: str = "sha256") -> str:
        """Calculate hash of a file

        Args:
            file_path: Path to the file
            algorithm: Hash algorithm to use (sha256, sha1, md5)

        Returns:
            Hexadecimal hash string
        """
        if not os.path.exists(file_path):
            return ""

        hash_obj = hashlib.new(algorithm)

        # Read file in chunks to handle large files
        with open(file_path, 'rb') as f:
            while chunk := f.read(8192):
                hash_obj.update(chunk)

        return hash_obj.hexdigest()

    def clear_cache(self):
        """Clear all cached hashes"""
        self.cache = {}
        self._save_cache()

    def remove_entry(self, file_path: str, algorithm: str = "sha256"):
        """Remove specific entry from cache

        Args:
            file_path: Path to file
            algorithm: Hash algorithm
        """
        cache_key = f"{file_path}:{algorithm}"
        if cache_key in self.cache:
            del self.cache[cache_key]
            self._save_cache()


# Global cache instance
_hash_cache = HashCache()


def get_cached_file_hash(file_path: str, algorithm: str = "sha256") -> str:
    """Get file hash from cache or calculate it

    This is a convenience function that uses the global cache instance.

    Args:
        file_path: Path to the file
        algorithm: Hash algorithm to use (sha256, sha1, md5)

    Returns:
        Hexadecimal hash string
    """
    return _hash_cache.calculate_and_cache(file_path, algorithm)
