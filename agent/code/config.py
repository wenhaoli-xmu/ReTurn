import os
import httpx
from agent.probe import PROMPT


ENABLE_LOCAL = True
LOCAL_LLM_BASE_URL = os.environ.get("LOCAL_LLM_BASE_URL", "")
LOCAL_LLM_MODEL_NAME = os.environ.get("LOCAL_LLM_MODEL_NAME", "Qwen3.5-4B")


SUMMARY_LLM_API_KEY = os.environ.get("SUMMARY_LLM_API_KEY", "")
SUMMARY_LLM_BASE_URL = os.environ.get("SUMMARY_LLM_BASE_URL", "")
SUMMARY_LLM_MODEL_NAME = os.environ.get("SUMMARY_LLM_MODEL_NAME", "deepseek-v4-flash")


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


SUMMARIZER_CONFIGS["summary_llm"] = SUMMARIZER_CONFIGS[
    "local_summary_llm" if ENABLE_LOCAL else "remote_summary_llm"]
