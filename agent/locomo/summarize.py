import asyncio
import functools
import hashlib
import logging
import os
import sqlite3
from collections import OrderedDict
from pathlib import Path

import httpx
from tenacity import retry, stop_after_attempt, wait_random_exponential, retry_if_exception

from agent.llm import chat
from agent.utils import is_retryable_http_exc
from agent.locomo.config import SUMMARIZER_CONFIGS

logger = logging.getLogger(__name__)


_PROMPT = """Write a one-line retrieval index for the passage below, at most {}
words.
Use concrete names and distinctive facts rather than broad topic
labels. Keep who did, said, or experienced what clear.
Do not let lengthy generic advice crowd out brief factual statements
from the speakers.
If the passage covers several distinct events or topics, include cues
to each instead of describing only the dominant topic.
Use the passage’s wording where possible. Include dates or temporal
links when they distinguish events.
Do not invent facts or replace specific details with generic
commentary. Output only the index entry.
PASSAGE:
{}"""

_sem = asyncio.Semaphore(SUMMARIZER_CONFIGS["concurrency"])
_client = httpx.AsyncClient(**SUMMARIZER_CONFIGS["httpx_config"])


_DISK_CACHE_PATH = os.environ.get("SUMMARY_DISK_CACHE")
_disk_cache = None


def _disk_cache_conn():
    global _disk_cache
    if not _DISK_CACHE_PATH:
        return None
    if _disk_cache is None:
        path = Path(_DISK_CACHE_PATH)
        path.parent.mkdir(parents=True, exist_ok=True)
        _disk_cache = sqlite3.connect(path, timeout=60)
        _disk_cache.execute("PRAGMA journal_mode=WAL")
        _disk_cache.execute("PRAGMA synchronous=NORMAL")
        _disk_cache.execute("PRAGMA busy_timeout=60000")
        _disk_cache.execute(
            "CREATE TABLE IF NOT EXISTS summaries (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
    return _disk_cache


def _disk_get(key):
    conn = _disk_cache_conn()
    if conn is None:
        return None
    row = conn.execute("SELECT value FROM summaries WHERE key = ?", (key,)).fetchone()
    return None if row is None else row[0]


def _disk_put(key, value):
    conn = _disk_cache_conn()
    if conn is not None:
        conn.execute("INSERT OR REPLACE INTO summaries(key, value) VALUES (?, ?)", (key, value))
        conn.commit()


def _async_lru(maxsize, key):
    def deco(fn):
        cache = OrderedDict()

        @functools.wraps(fn)
        async def wrapper(*args, **kw):
            k = key(*args, **kw)
            if k in cache:
                cache.move_to_end(k)
                return cache[k]
            out = await fn(*args, **kw)
            cache[k] = out
            cache.move_to_end(k)
            if len(cache) > maxsize:
                cache.popitem(last=False)
            return out

        wrapper.cache = cache
        return wrapper
    return deco


@retry(
    stop=stop_after_attempt(3),
    wait=wait_random_exponential(multiplier=0.5, min=0.5, max=8),
    retry=retry_if_exception(is_retryable_http_exc),
    reraise=True)
async def _call(text):
    prompt = _PROMPT.format(SUMMARIZER_CONFIGS["max_words"], text)
    out = await chat(_client, _sem, SUMMARIZER_CONFIGS["summary_llm"], prompt)
    return " ".join(out.split())


def _key(text):
    llm = SUMMARIZER_CONFIGS["summary_llm"]["model_name"]
    cap = SUMMARIZER_CONFIGS["summary_llm"].get("extra_body", {}).get("max_tokens")
    return hashlib.sha1(
        f"{llm}\0cap{cap}\0{SUMMARIZER_CONFIGS['max_words']}\0{_PROMPT}\0{text}".encode()).hexdigest()


FAILED = "The summarization process failed."


@_async_lru(SUMMARIZER_CONFIGS["cache_size"], key=_key)
async def _summarize(text):
    key = _key(text)
    cached = _disk_get(key)
    if cached is not None:
        return cached
    out = await _call(text)
    _disk_put(key, out)
    return out


async def summarize(text):
    try:
        return await _summarize(text)
    except Exception as e:
        logger.warning("summarize failed: %s", e)
        return FAILED
