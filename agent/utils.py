import torch
import httpx


RETRYABLE_HTTP_STATUS = frozenset({408, 429, 500, 502, 503, 504})
_RETRYABLE_NETWORK_EXC = (
    httpx.ConnectError,
    httpx.ReadError,
    httpx.TimeoutException,
    httpx.RemoteProtocolError,
)


_BANNED_URL_PATTERNS = (
    "unifuncs",    
    "huggingface.co/datasets",  
    "huggingface.co/spaces",
)


def is_retryable_http_exc(exc: BaseException) -> bool:

    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code in RETRYABLE_HTTP_STATUS
    return isinstance(exc, _RETRYABLE_NETWORK_EXC)


def is_banned_url(url: str) -> bool:

    if not url:
        return False
    lowered = url.lower()
    return any(pattern in lowered for pattern in _BANNED_URL_PATTERNS)


