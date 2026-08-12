
#TIER 1 OF CACHING

"""
app/cache/local_cache.py

Tier 1 of the two-tier cache: an in-process, in-memory LRU used by
services/pipeline.py (in the async event-loop process) to skip
submitting a chunk to the worker pool entirely when the exact same
audio has already been seen.

Scope and boundaries, explicitly:
  - This module knows nothing about audio, models, or transcription.
    It stores and retrieves plain, JSON-serializable dicts against a
    string key. Composing that key (audio hash + a redaction/model
    version tag) is cache/hashing.py's + pipeline.py's job, not this
    file's — see the note below on why the version tag matters.
  - This cache lives in ONE process — whichever process is running
    the asyncio event loop that handles WebSocket connections. Run
    multiple uvicorn workers and each gets its own separate
    LocalCache instance with zero coordination between them. That's
    intentional: cross-process/cross-replica sharing is exactly what
    Tier 2 (Redis) exists for. Don't expect Redis-like guarantees here.
  - Designed for single-threaded asyncio access. A plain OrderedDict
    is safe under that assumption with no lock needed. If this cache
    is ever touched from a thread pool or from more than one
    event-loop thread, that assumption breaks and a lock must be
    added.

A note on cache correctness for a compliance product specifically:
if the redaction patterns (or the model) ever change between deploys,
a stale cache entry computed under the OLD rules could keep being
served indefinitely — which here means potentially under-redacted
text going out again. Sizing/TTL alone don't fix that; the fix is
that the KEY itself must change when redaction/model logic changes
(e.g. include a version tag in the key composed upstream), so a
deploy naturally invalidates old entries instead of an operator
having to remember to flush the cache. This file just stores
whatever key it's given — but that upstream requirement is why the
key is treated as an opaque string here rather than "just the audio
hash."
"""

from __future__ import annotations

import logging
import time
from collections import OrderedDict
from typing import Any, Optional

logger = logging.getLogger("cache.local")


class LocalCache:
    #we are using two cache invalidation and eviction methods for this problem to get the best outcome: Bounded, in-memory LRU and TTL (optional).
    def __init__(self, max_size: int = 2048, ttl_seconds: Optional[float] = None):
        if max_size < 1:
            raise ValueError("max_size must be >= 1")
        self._max_size = max_size #cache occupancy (2GB)
        self._ttl_seconds = ttl_seconds #cache expiry
        self._store: "OrderedDict[str, tuple[float, dict[str, Any]]]" = OrderedDict() #cache data type: ordered dictionary as a key,value tuple.

    def __getitem__(self, key: str) -> Optional[dict[str, Any]]:
        """Returns the value for the key. None if missing."""
        entry = self._store.get(key) #O(1) operation
        
        #"entry" is a tuple
        if entry is None:
            return None
        
        #extract dimensions:
        inserted_at, value = entry
        
        #removing (invalidating) cache based on TTL(optional)
        self.check_cache = self._is_expired(inserted_at)
        
        if self.check_cache:
            # Cleaned up lazily, on access, rather than via a background
            # sweep — simpler, and sufficient at MVP scale. A background
            # reaper is a reasonable post-MVP addition if profiling ever
            # shows dead entries accumulating between accesses.
            
            del self._store[key] #invalidating the tuple
            logger.debug("local cache expired: %s", key)
            return None

        #marking as recently-used. (last is )
        self._store.move_to_end(key) #reordering in O(1)
        return value

    def __setitem__(self, key: str, value: dict[str, Any]) -> None:
        """
        Inserts or overwrites an entry in cache. Evicts if the cache capacity hits maximum occupancy (_max_size) based on LRU approach.
        """
        if key in self._store:
            self._store.move_to_end(key) #moving it at the end of the stack.
        self._store[key] = (time.monotonic(), value)

        if len(self._store) > self._max_size:
            evicted_key, _ = self._store.popitem(last=False) # O(1) bidirectional eviction:(first is the least recently-used/ oldest)
            logger.debug("local cache evicted (over capacity): %s", evicted_key)

    #returns the length (size) of the current size of cache (out of the _max_size)
    def __len__(self) -> int:
        return len(self._store)

    #checks if the cache address(key) value is expired or not:
    def _is_expired(self, inserted_at: float) -> bool:
        if self._ttl_seconds is None:
            return False
        return (time.monotonic() - inserted_at) > self._ttl_seconds