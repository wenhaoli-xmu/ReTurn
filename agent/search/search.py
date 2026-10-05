import httpx, asyncio, orjson
from typing import Dict, Any
from tenacity import retry, stop_after_attempt, wait_random_exponential, retry_if_exception

from agent.utils import is_retryable_http_exc, is_banned_url
from agent.search.config import SEARCH_CONFIGS


_semaphore = asyncio.Semaphore(SEARCH_CONFIGS["search_concurrency"])

_serper_client = httpx.AsyncClient(
    http2=True,
    **SEARCH_CONFIGS["search_httpx_config"])


@retry(
    stop=stop_after_attempt(3),
    wait=wait_random_exponential(multiplier=0.5, min=0.5, max=4),
    retry=retry_if_exception(is_retryable_http_exc),
    reraise=True)
async def make_serper_request(
        payload: Dict[str, Any],
        headers: Dict[str, str],
    ) -> dict:

    serper_base_url = SEARCH_CONFIGS["serper"]["base_url"]
    body = orjson.dumps(payload)
    merged_headers = {"Content-Type": "application/json", **headers}
    async with _semaphore:
        response = await _serper_client.post(
            f"{serper_base_url}/search",
            content=body,
            headers=merged_headers,
        )
        response.raise_for_status()
        data = orjson.loads(response.content)

    return data


async def search(keyword: str) -> str:

    serper_api_key = SEARCH_CONFIGS["serper"]["api_key"]
    if not keyword or not keyword.strip():
        return "search keyword cannot be empty"
    try:
        payload = {"q": keyword.strip()}

        headers = {
            "X-API-KEY": serper_api_key,
            "Content-Type": "application/json",
        }

        data = await make_serper_request(payload, headers)


        organic_results = []
        if "organic" in data:
            for item in data["organic"]:
                if is_banned_url(item.get("link", "")):
                    continue
                organic_results.append(item)


        response_data = {
            "organic": organic_results,
            "searchParameters": data.get("searchParameters", {}),
        }

        return orjson.dumps(response_data).decode()

    except Exception as e:
        return f"serper_search failed for keyword {keyword}: {str(e)}"