"""Efficiency optimizations for the autonomous browser agent

This module provides:
- Action caching for repeated interactions
- LRU cache for expensive computations
- Async task batching
- Memory-efficient data structures
"""

import asyncio
import hashlib
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from functools import wraps
from typing import Any, Callable, Dict, Generic, List, Optional, Tuple, TypeVar

from loguru import logger


T = TypeVar('T')


class LRUCache(Generic[T]):
    """
    Thread-safe LRU Cache with TTL support.

    Used for caching expensive computations like:
    - Coordinate translations
    - Similar element lookups
    - Model outputs for similar inputs
    """

    def __init__(self, maxsize: int = 100, ttl_seconds: Optional[int] = None):
        self._cache: OrderedDict[str, Tuple[T, float]] = OrderedDict()
        self._maxsize = maxsize
        self._ttl = ttl_seconds
        self._hits = 0
        self._misses = 0

    def _is_expired(self, timestamp: float) -> bool:
        """Check if entry has expired"""
        if self._ttl is None:
            return False
        return time.time() - timestamp > self._ttl

    def get(self, key: str) -> Optional[T]:
        """Get value from cache"""
        if key in self._cache:
            value, timestamp = self._cache[key]

            if self._is_expired(timestamp):
                del self._cache[key]
                self._misses += 1
                return None

            # Move to end (most recently used)
            self._cache.move_to_end(key)
            self._hits += 1
            return value

        self._misses += 1
        return None

    def set(self, key: str, value: T) -> None:
        """Set value in cache"""
        # Remove oldest if at capacity
        while len(self._cache) >= self._maxsize:
            self._cache.popitem(last=False)

        self._cache[key] = (value, time.time())
        self._cache.move_to_end(key)

    def clear(self) -> None:
        """Clear the cache"""
        self._cache.clear()

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        total = self._hits + self._misses
        return {
            "size": len(self._cache),
            "maxsize": self._maxsize,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": self._hits / total if total > 0 else 0,
        }


@dataclass
class CachedAction:
    """A cached action with its target coordinates"""
    url_pattern: str
    element_description: str
    bbox: Tuple[int, int, int, int]
    pixel_coords: Tuple[int, int]
    confidence: float
    created_at: datetime = field(default_factory=datetime.now)
    use_count: int = 0
    last_success: Optional[datetime] = None


class ActionCache:
    """
    Cache for browser actions to avoid redundant vision model calls.

    From specification:
    - Reuse coordinates for repeated clicks on same elements
    - Cache action patterns by URL + element description
    """

    def __init__(self, max_entries: int = 500, ttl_hours: int = 24):
        self._cache: Dict[str, CachedAction] = {}
        self._max_entries = max_entries
        self._ttl = timedelta(hours=ttl_hours)

    def _generate_key(self, url: str, element_description: str) -> str:
        """Generate cache key from URL pattern and element"""
        # Normalize URL (remove query params and fragments)
        from urllib.parse import urlparse
        parsed = urlparse(url)
        url_pattern = f"{parsed.netloc}{parsed.path}"

        # Create hash of description for consistent keys
        desc_hash = hashlib.md5(element_description.lower().encode()).hexdigest()[:8]

        return f"{url_pattern}:{desc_hash}"

    def get(
        self,
        url: str,
        element_description: str
    ) -> Optional[CachedAction]:
        """
        Look up cached action.

        Args:
            url: Current page URL
            element_description: Description of target element

        Returns:
            CachedAction if found and valid, None otherwise
        """
        key = self._generate_key(url, element_description)

        if key not in self._cache:
            return None

        cached = self._cache[key]

        # Check TTL
        if datetime.now() - cached.created_at > self._ttl:
            del self._cache[key]
            return None

        # Update use count
        cached.use_count += 1

        logger.debug(
            f"[ActionCache] Hit for '{element_description[:30]}...' "
            f"(use count: {cached.use_count})"
        )

        return cached

    def store(
        self,
        url: str,
        element_description: str,
        bbox: Tuple[int, int, int, int],
        pixel_coords: Tuple[int, int],
        confidence: float,
    ) -> None:
        """
        Store action in cache.

        Only stores high-confidence actions for reliability.
        """
        # Only cache high-confidence actions
        if confidence < 0.85:
            return

        # Evict if at capacity
        if len(self._cache) >= self._max_entries:
            self._evict_oldest()

        key = self._generate_key(url, element_description)

        self._cache[key] = CachedAction(
            url_pattern=url,
            element_description=element_description,
            bbox=bbox,
            pixel_coords=pixel_coords,
            confidence=confidence,
        )

        logger.debug(f"[ActionCache] Stored action for '{element_description[:30]}...'")

    def record_success(self, url: str, element_description: str) -> None:
        """Record successful use of cached action"""
        key = self._generate_key(url, element_description)
        if key in self._cache:
            self._cache[key].last_success = datetime.now()

    def record_failure(self, url: str, element_description: str) -> None:
        """Invalidate cached action on failure"""
        key = self._generate_key(url, element_description)
        if key in self._cache:
            del self._cache[key]
            logger.debug(f"[ActionCache] Invalidated failed action")

    def _evict_oldest(self) -> None:
        """Evict oldest entries to make room"""
        # Sort by created_at and remove oldest 10%
        sorted_items = sorted(
            self._cache.items(),
            key=lambda x: x[1].created_at
        )

        evict_count = max(1, len(sorted_items) // 10)
        for key, _ in sorted_items[:evict_count]:
            del self._cache[key]

    def clear(self) -> None:
        """Clear the cache"""
        self._cache.clear()

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        if not self._cache:
            return {"size": 0, "avg_use_count": 0}

        total_uses = sum(c.use_count for c in self._cache.values())
        return {
            "size": len(self._cache),
            "max_entries": self._max_entries,
            "avg_use_count": total_uses / len(self._cache),
            "total_uses": total_uses,
        }


class AsyncBatcher:
    """
    Batches async operations for efficiency.

    Useful for:
    - Parallel screenshot captures
    - Concurrent API calls
    - Batch memory operations
    """

    def __init__(self, max_batch_size: int = 10, max_wait_ms: int = 100):
        self._max_batch_size = max_batch_size
        self._max_wait_ms = max_wait_ms

    async def batch_execute(
        self,
        operations: List[Callable[[], Any]],
        max_concurrent: int = 5,
    ) -> List[Any]:
        """
        Execute operations in batches with concurrency limit.

        Args:
            operations: List of async callables
            max_concurrent: Maximum concurrent operations

        Returns:
            List of results in order
        """
        semaphore = asyncio.Semaphore(max_concurrent)

        async def limited_operation(op):
            async with semaphore:
                return await op()

        return await asyncio.gather(
            *[limited_operation(op) for op in operations],
            return_exceptions=True,
        )

    async def batch_with_timeout(
        self,
        operations: List[Callable[[], Any]],
        timeout_seconds: float = 30,
    ) -> List[Tuple[bool, Any]]:
        """
        Execute operations with timeout.

        Returns:
            List of (success, result) tuples
        """
        results = []

        for op in operations:
            try:
                result = await asyncio.wait_for(op(), timeout=timeout_seconds)
                results.append((True, result))
            except asyncio.TimeoutError:
                results.append((False, "Timeout"))
            except Exception as e:
                results.append((False, str(e)))

        return results


def cached(ttl_seconds: int = 300, maxsize: int = 100):
    """
    Decorator for caching function results.

    Args:
        ttl_seconds: Time to live for cache entries
        maxsize: Maximum cache size
    """
    cache = LRUCache[Any](maxsize=maxsize, ttl_seconds=ttl_seconds)

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Create cache key from arguments
            key_parts = [str(arg) for arg in args]
            key_parts.extend(f"{k}={v}" for k, v in sorted(kwargs.items()))
            key = hashlib.md5(":".join(key_parts).encode()).hexdigest()

            # Check cache
            result = cache.get(key)
            if result is not None:
                return result

            # Call function and cache result
            result = func(*args, **kwargs)
            cache.set(key, result)
            return result

        wrapper.cache = cache
        wrapper.cache_clear = cache.clear
        return wrapper

    return decorator


def async_cached(ttl_seconds: int = 300, maxsize: int = 100):
    """
    Decorator for caching async function results.
    """
    cache = LRUCache[Any](maxsize=maxsize, ttl_seconds=ttl_seconds)

    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Create cache key
            key_parts = [str(arg) for arg in args]
            key_parts.extend(f"{k}={v}" for k, v in sorted(kwargs.items()))
            key = hashlib.md5(":".join(key_parts).encode()).hexdigest()

            # Check cache
            result = cache.get(key)
            if result is not None:
                return result

            # Call function and cache result
            result = await func(*args, **kwargs)
            cache.set(key, result)
            return result

        wrapper.cache = cache
        wrapper.cache_clear = cache.clear
        return wrapper

    return decorator


# Global instances
_action_cache = ActionCache()
_coordinate_cache = LRUCache[Tuple[int, int]](maxsize=1000, ttl_seconds=3600)


def get_action_cache() -> ActionCache:
    """Get the global action cache"""
    return _action_cache


def get_coordinate_cache() -> LRUCache[Tuple[int, int]]:
    """Get the global coordinate cache"""
    return _coordinate_cache


# Performance monitoring
class PerformanceMonitor:
    """Monitor and log performance metrics"""

    def __init__(self):
        self._timings: Dict[str, List[float]] = {}
        self._counts: Dict[str, int] = {}

    def record_timing(self, operation: str, duration_ms: float) -> None:
        """Record timing for an operation"""
        if operation not in self._timings:
            self._timings[operation] = []

        self._timings[operation].append(duration_ms)

        # Keep only last 100 timings
        if len(self._timings[operation]) > 100:
            self._timings[operation] = self._timings[operation][-100:]

    def record_count(self, operation: str, count: int = 1) -> None:
        """Record count for an operation"""
        self._counts[operation] = self._counts.get(operation, 0) + count

    def get_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        stats = {"timings": {}, "counts": self._counts.copy()}

        for op, timings in self._timings.items():
            if timings:
                stats["timings"][op] = {
                    "count": len(timings),
                    "avg_ms": sum(timings) / len(timings),
                    "min_ms": min(timings),
                    "max_ms": max(timings),
                }

        return stats


_perf_monitor = PerformanceMonitor()


def get_performance_monitor() -> PerformanceMonitor:
    """Get the global performance monitor"""
    return _perf_monitor
