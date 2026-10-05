import os
import httpx
from agent.probe import PROMPT


JINA_API_KEY = os.environ.get("JINA_API_KEY", "")
JINA_BASE_URL = os.environ.get("JINA_BASE_URL", "")


SERPER_API_KEY = os.environ.get("SERPER_API_KEY", "")
SERPER_BASE_URL = os.environ.get("SERPER_BASE_URL", "")


ENABLE_LOCAL = True
LOCAL_LLM_BASE_URL = os.environ.get("LOCAL_LLM_BASE_URL", "")
LOCAL_LLM_MODEL_NAME = os.environ.get("LOCAL_LLM_MODEL_NAME", "Qwen3.5-4B")


SUMMARY_LLM_API_KEY = os.environ.get("SUMMARY_LLM_API_KEY", "")
SUMMARY_LLM_BASE_URL = os.environ.get("SUMMARY_LLM_BASE_URL", "")
SUMMARY_LLM_MODEL_NAME = os.environ.get("SUMMARY_LLM_MODEL_NAME", "deepseek-v4-flash")


GRADER_LLM_API_KEY = ""
GRADER_LLM_BASE_URL = ""
GRADER_LLM_MODEL_NAME = "deepseek-v4-flash"


AGENT_CONFIGS = {
    "max_length": 131072,
    "use_active_context_length": os.environ.get("USE_ACTIVE_CONTEXT_LENGTH", "0") == "1",
    "sampling": {"temperature": 0.6, "top_p": 0.95, "top_k": 20},
    "max_turns": 400,
    "single_turn_max_length": 32768,
    "sanity_check": True,
}


SUMMARIZER_CONFIGS = {
    "remote_summary_llm": {
        "api_key": SUMMARY_LLM_API_KEY,
        "base_url": SUMMARY_LLM_BASE_URL,
        "model_name": SUMMARY_LLM_MODEL_NAME,
        "extra_body": {"reasoning_effort": "low"},
    },
    "local_summary_llm": {
        "api_key": "",
        "base_url": LOCAL_LLM_BASE_URL,
        "model_name": LOCAL_LLM_MODEL_NAME,
        "extra_body": {"reasoning_effort": "low"},
    },
    "max_words": 12,
    "max_words_assistant": 12,
    "cache_size": 16384,
    "concurrency": 64,
    "httpx_config": {
        "limits": httpx.Limits(
            max_connections=64,
            max_keepalive_connections=64,
            keepalive_expiry=30.0),
        "timeout": httpx.Timeout(
            connect=10.0,
            read=300.0,
            write=10.0,
            pool=10.0),
    },
}


UNFOLD_CONFIGS = {
    "enable": True,
    "keep_last_k": 3,
    "probe_temperature": 0.0,
    "prompt": PROMPT,
    "probe_max_new_tokens": 256,


    "bm25_union_top_k": int(os.environ.get("BM25_UNION_TOP_K", 0)),
    "selection_mode": os.environ.get("UNFOLD_SELECTION_MODE", "model"),
}


SEARCH_CONFIGS = {
    "search_concurrency": 64,
    "search_httpx_config": {
        "limits": httpx.Limits(
            max_connections=64,
            max_keepalive_connections=64,
            keepalive_expiry=30.0,
        ),
        "timeout": httpx.Timeout(
            connect=10.0,
            read=30.0,
            write=10.0,
            pool=10.0)},
    "serper": {
        "api_key": SERPER_API_KEY,
        "base_url": SERPER_BASE_URL,
    }
}


FETCH_CONFIGS = {
    "jina": {
        "api_key": JINA_API_KEY,
        "base_url": JINA_BASE_URL,
    },
    "fetch_concurrency": 64,
    "fetch_httpx_config": {
        "limits": httpx.Limits(
            max_connections=128,
            max_keepalive_connections=100,
            keepalive_expiry=30.0,
        ),
        "timeout": httpx.Timeout(
            connect=10.0,
            read=300.0,
            write=10.0,
            pool=10.0)
    },
    "remote_summary_llm": {
        "api_key": SUMMARY_LLM_API_KEY,
        "base_url": SUMMARY_LLM_BASE_URL,
        "model_name": SUMMARY_LLM_MODEL_NAME,
        "extra_body": {"reasoning_effort": "low"},
    },
    "local_summary_llm": {
        "api_key": "",
        "base_url": LOCAL_LLM_BASE_URL,
        "model_name": LOCAL_LLM_MODEL_NAME,
        "extra_body": {"reasoning_effort": "low"},
    },
    "summary_concurrency": 16,
    "summary_httpx_config": {
        "limits": httpx.Limits(
            max_connections=16,
            max_keepalive_connections=16,
            keepalive_expiry=30.0,
        ),
        "timeout": httpx.Timeout(
            connect=10.0,
            read=300.0,
            write=10.0,
            pool=10.0)
    },
}


GRADER_CONFIGS = {
    "grader_llm": {
        "api_key": GRADER_LLM_API_KEY,
        "base_url": GRADER_LLM_BASE_URL,
        "model_name": GRADER_LLM_MODEL_NAME,
        "extra_body": {"reasoning_effort": "low"},
    },
    "grader_concurrency": 16,
    "grader_httpx_config": {
        "limits": httpx.Limits(
            max_connections=16,
            max_keepalive_connections=16,
            keepalive_expiry=30.0),
        "timeout": httpx.Timeout(
            connect=10.0,
            read=300.0,
            write=10.0,
            pool=10.0),
    },
}

for _c in (SUMMARIZER_CONFIGS, FETCH_CONFIGS):
    _c["summary_llm"] = _c["local_summary_llm" if ENABLE_LOCAL else "remote_summary_llm"]
