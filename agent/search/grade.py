import re, string, httpx, asyncio

from tenacity import retry, stop_after_attempt, wait_random_exponential, retry_if_exception
from agent.llm import chat
from agent.utils import is_retryable_http_exc
from agent.search.config import GRADER_CONFIGS
from ..data import Judge


_grader_semaphore = asyncio.Semaphore(GRADER_CONFIGS["grader_concurrency"])
_grader_client = httpx.AsyncClient(http2=True, **GRADER_CONFIGS["grader_httpx_config"])


def normalize_answer(s: str) -> str:
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def em_check(prediction: str, golden_answers: list[str]) -> bool:
    pred = normalize_answer(prediction)
    return any(normalize_answer(g) == pred for g in golden_answers)


def extract_answer(response: str):
    idx = response.rfind("</think>")
    tail = response[idx + len("</think>"):] if idx != -1 else response
    tail = re.compile(r"<\|[^|]*?\|>").sub("", tail).strip()
    return tail or None


GRADE_PROMPT = """You are grading a question-answering system. Decide whether the PREDICTED answer is semantically equivalent to ANY of the golden answers (same factual content; ignore casing, punctuation, articles, and phrasing differences).

GOLDEN ANSWERS:
{}

PREDICTED ANSWER:
{}

Reply with exactly one word: "yes" if the predicted answer is correct, otherwise "no"."""


@retry(
    stop=stop_after_attempt(4),
    wait=wait_random_exponential(multiplier=0.5, min=0.5, max=8),
    retry=retry_if_exception(is_retryable_http_exc),
    reraise=True)
async def grade_with_llm(predicted: str, golden_answers: list[str]) -> dict:

    prompt = GRADE_PROMPT.format("\n".join(golden_answers), predicted)
    verdict = (await chat(
        _grader_client, _grader_semaphore, GRADER_CONFIGS["grader_llm"], prompt)).strip()
    return {"ok": verdict.lower().startswith("y"), "prompt": prompt, "verdict": verdict}


async def grade(response: str, ground_truth) -> Judge:

    if isinstance(ground_truth, dict):
        golden = ground_truth.get("target", ground_truth.get("ground_truth"))
    else:
        golden = ground_truth
    golden_answers = [golden] if isinstance(golden, str) else list(golden)

    answer = extract_answer(response)
    if answer is None:
        return Judge(score=0.0, golden=golden_answers, predicted=None)
    if em_check(answer, golden_answers):
        return Judge(score=1.0, golden=golden_answers, predicted=answer)
    try:
        r = await grade_with_llm(answer, golden_answers)
        return Judge(
            score=1.0 if r["ok"] else 0.0, 
            golden=golden_answers,
            predicted=answer,
            prompt=r["prompt"], 
            verdict=r["verdict"])
    except Exception as e:
        return Judge(
            score=0.0,
            golden=golden_answers,
            predicted=answer,
            error=str(e))
