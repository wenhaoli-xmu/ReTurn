import math
import re
from collections import Counter


_WORD_RE = re.compile(r"[a-z0-9]+(?:['’][a-z0-9]+)?")
_QUESTION_RE = re.compile(
    r"(?:^|\n)Question:\s*(.*?)\s*Short answer:\s*$", re.IGNORECASE | re.DOTALL)


def lexical_tokens(text: str) -> list[str]:

    return _WORD_RE.findall(text.lower().replace("’", "'"))


def question_text(qa: str) -> str:

    match = _QUESTION_RE.search(qa)
    return match.group(1).strip() if match else qa.strip()


def _chunks(tokens: list[str], size: int, stride: int) -> list[list[str]]:
    if not tokens:
        return [[]]
    if len(tokens) <= size:
        return [tokens]
    starts = list(range(0, len(tokens) - size + 1, stride))
    final = len(tokens) - size
    if starts[-1] != final:
        starts.append(final)
    return [tokens[start:start + size] for start in starts]


def session_scores(
        question: str,
        sessions: list[str],
        *,
        chunk_size: int = 192,
        chunk_stride: int = 128,
        k1: float = 1.5,
        b: float = 0.75,
) -> list[float]:

    if chunk_size <= 0 or not 0 < chunk_stride <= chunk_size:
        raise ValueError("require chunk_size > 0 and 0 < chunk_stride <= chunk_size")

    query = Counter(lexical_tokens(question))
    docs, owners = [], []
    for owner, text in enumerate(sessions):
        for chunk in _chunks(lexical_tokens(text), chunk_size, chunk_stride):
            docs.append(Counter(chunk))
            owners.append(owner)
    if not docs or not query:
        return [0.0] * len(sessions)

    lengths = [sum(doc.values()) for doc in docs]
    avgdl = sum(lengths) / len(lengths)
    df = Counter()
    for doc in docs:
        df.update(doc.keys() & query.keys())

    n_doc = len(docs)
    idf = {
        term: math.log1p((n_doc - freq + 0.5) / (freq + 0.5))
        for term, freq in df.items()
    }
    scores = [0.0] * len(sessions)
    for doc, owner, dl in zip(docs, owners, lengths):
        norm = k1 * (1.0 - b + b * dl / max(avgdl, 1e-12))
        score = 0.0
        for term, qtf in query.items():
            tf = doc.get(term, 0)
            if tf:
                score += qtf * idf[term] * tf * (k1 + 1.0) / (tf + norm)
        scores[owner] = max(scores[owner], score)
    return scores
