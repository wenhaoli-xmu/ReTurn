import httpx, asyncio, orjson, json, logging

from typing import Dict, Any

from tenacity import retry, stop_after_attempt, wait_random_exponential, retry_if_exception
from agent.llm import chat
from agent.utils import is_retryable_http_exc, is_banned_url
from agent.search.config import FETCH_CONFIGS


logger = logging.getLogger(__name__)


_semapore = asyncio.Semaphore(FETCH_CONFIGS["fetch_concurrency"])

_jina_client = httpx.AsyncClient(
    http2=True,
    **FETCH_CONFIGS["fetch_httpx_config"]
)

_summary_semaphore = asyncio.Semaphore(FETCH_CONFIGS["summary_concurrency"])

_summary_client = httpx.AsyncClient(
    http2=True,
    **FETCH_CONFIGS["summary_httpx_config"]
)

@retry(
    stop=stop_after_attempt(4),
    wait=wait_random_exponential(multiplier=0.5, min=0.5, max=8),
    retry=retry_if_exception(is_retryable_http_exc),
    reraise=True)
async def fetch_jina(url: str):

    jina_url = f"{FETCH_CONFIGS['jina']['base_url']}/{url}"
    headers = {"Authorization": f"Bearer {FETCH_CONFIGS['jina']['api_key']}"}

    async with _semapore:
        response = await _jina_client.post(
            jina_url,
            headers=headers)
        response.raise_for_status()

    return response.text


EXTRACT_INFO_PROMPT = """You are given a piece of content and the requirement of information to extract. Your task is to extract the information specifically requested. Be precise and focus exclusively on the requested information.

INFORMATION TO EXTRACT:
{}

INSTRUCTIONS:
1. Extract the information relevant to the focus above.
2. If the exact information is not found, extract the most closely related details.
3. Be specific and include exact details when available.
4. Clearly organize the extracted information for easy understanding.
5. Do not include general summaries or unrelated content.

CONTENT TO ANALYZE:
{}

EXTRACTED INFORMATION:"""

def get_prompt_with_truncation(
        info_to_extract: str,
        content: str
    ) -> str:
    prompt = EXTRACT_INFO_PROMPT.format(info_to_extract, content)
    return prompt


@retry(
    stop=stop_after_attempt(4),
    wait=wait_random_exponential(multiplier=0.5, min=0.5, max=8),
    retry=retry_if_exception(is_retryable_http_exc),
    reraise=True)
async def extract(
        content: str,
        query: str,
    ) -> str:

    if not content or not content.strip():

        pass

    prompt = get_prompt_with_truncation(query, content)

    return await chat(
        _summary_client, _summary_semaphore, FETCH_CONFIGS["summary_llm"], prompt)


async def fetch(
        url: str,
        query: str,
) -> Dict[str, Any]:

    if is_banned_url(url):
        return "You are trying to scrape a banned answer source"

    content = await fetch_jina(url)
    result = await extract(
        content=content,
        query=query)

    return result
