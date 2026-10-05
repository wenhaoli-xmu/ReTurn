import asyncio
import functools
import hashlib
import logging
import json
from collections import OrderedDict

import httpx
from tenacity import retry, stop_after_attempt, wait_random_exponential, retry_if_exception

from agent.llm import chat
from agent.utils import is_retryable_http_exc
from agent.code.config import SUMMARIZER_CONFIGS

logger = logging.getLogger(__name__)


_PROMPT_OBS = """One line. This is an INDEX ENTRY for a tool result, not a summary: the full
text stays retrievable on demand, so do NOT try to preserve it. The single question the
line must answer is "would an agent hunting for some fact need to pull this back?"

If the result is empty, an error, a login/paywall page, or off topic, say exactly that
and nothing else — that tells the agent never to pull it back.
Write in the language of the observation.

Be pathologically terse. No preamble, no meta-commentary. At most {} words.

OBSERVATION:
{}"""

_PROMPT_ASSISTANT = """One line. This is an INDEX ENTRY of what the agent did.

Format:  <conclusion>; <tool>(<args>)

If the turn calls a tool, write it as a call expression: the tool's own name, lowercase
and verbatim as it appears in the turn, then its argument values in parentheses. Never
describe it in prose ("searched for X", "retrieved Y via the tool") and never reproduce
<function=...> / <parameter=...> markup or JSON envelopes. Strip URLs down to
host/last-path-segment, truncate long values — but the call expression is mandatory and
the tool name is never dropped, no matter how tight the word budget gets.
If the turn calls no tool, write only the conclusion.

Shape of the line (substitute the turn's real tool name and arguments):
    nothing conclusive yet; TOOLNAME("first argument")
    need the missing field; TOOLNAME(short-arg, short-arg)
    answer settled: ANSWER

Be pathologically terse. No preamble, no meta-commentary. At most {} words.

AGENT TURN:
{}"""

_PROMPTS = {
    "obs": _PROMPT_OBS,
    "assistant": _PROMPT_ASSISTANT}

_sem = asyncio.Semaphore(SUMMARIZER_CONFIGS["concurrency"])
_client = httpx.AsyncClient(**SUMMARIZER_CONFIGS["httpx_config"])


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
async def _call(text, kind):
    prompt = _PROMPTS[kind].format(_max_words(kind), text)
    out = await chat(_client, _sem, SUMMARIZER_CONFIGS["summary_llm"], prompt)
    return out.strip()


def _max_words(kind):
    return SUMMARIZER_CONFIGS["max_words_assistant" if kind == "assistant" else "max_words"]


def _key(text, kind="obs"):

    cfg = SUMMARIZER_CONFIGS["summary_llm"]
    prompt = _PROMPTS[kind].format(_max_words(kind), text)
    settings = json.dumps(cfg.get("extra_body", {}), sort_keys=True)
    return hashlib.sha256(f"{cfg['model_name']}\0{kind}\0{settings}\0{prompt}".encode()).hexdigest()


FAILED = "The summarization process failed."


@_async_lru(SUMMARIZER_CONFIGS["cache_size"], key=_key)
async def _summarize(text, kind):
    return await _call(text, kind)


async def summarize(text, kind="obs"):
    try:
        s = await _summarize(text, kind)
    except Exception as e:
        logger.warning("summarize failed (%s): %s", kind, e)
        s = FAILED
    return s
