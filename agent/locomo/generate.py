import asyncio
import random
import re
import uuid

import httpx
from tenacity import retry, stop_after_attempt, wait_random_exponential, retry_if_exception

from agent.utils import is_retryable_http_exc
from agent.probe import assistant_guide
from agent.data import Request, Traj
from agent.locomo.bm25 import question_text, session_scores
from agent.locomo.config import LOCOMO_CONFIGS, UNFOLD_CONFIGS
from agent.locomo.fold import Ledger, pick_unfold, sanitize, sanity_check, window_fold
from agent.locomo.summarize import summarize


_gen_client = httpx.AsyncClient(
    limits=httpx.Limits(max_connections=None, max_keepalive_connections=None),
    timeout=httpx.Timeout(connect=10.0, read=None, write=10.0, pool=None))


@retry(
    stop=stop_after_attempt(3),
    wait=wait_random_exponential(multiplier=0.5, min=0.5, max=4),
    retry=retry_if_exception(is_retryable_http_exc),
    reraise=True)
async def send_request(
        session_id: str,
        input_ids: list[int],
        url: str,
        max_new_tokens: int,
        temperature: float = 0.0,
        top_p: float = 1.0,
        top_k: int = 0,
        stop_ids: list[list[int]] | None = None,
        split_ids: list[int] | None = None,
        subs: list | None = None,
        swap: list | None = None,
        ephem_at: int | None = None,
        swap_at: int | None = None,
) -> dict:
    payload = {
        "session_id": session_id,
        "prompt_ids": input_ids,
        "stop_ids": stop_ids or [],
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
        "split_ids": split_ids or [],
        "subs": subs or [],
        "swap": swap or [],
        "ephem_at": ephem_at,
        "swap_at": swap_at,
    }
    resp = await _gen_client.post(url, json=payload)
    resp.raise_for_status()
    return resp.json()


@retry(
    stop=stop_after_attempt(3),
    wait=wait_random_exponential(multiplier=0.5, min=0.5, max=4),
    retry=retry_if_exception(is_retryable_http_exc),
    reraise=True)
async def release(session_id: str, url: str) -> None:
    await _gen_client.post(url.rsplit("/generate", 1)[0] + "/release", json={"session_id": session_id})


_TOKEN_RE = re.compile(r"<\|[^|]*?\|>")


def extract_answer(response: str) -> str:
    idx = response.rfind("</think>")
    tail = response[idx + len("</think>"):] if idx != -1 else response
    return _TOKEN_RE.sub("", tail).strip()


async def generate(request: Request, sample) -> tuple[Traj, str, dict]:

    request = request.with_sampling(**LOCOMO_CONFIGS["sampling"])
    tok = request.tokenizer
    encode = lambda x: tok(x, add_special_tokens=False)["input_ids"]

    session_id = uuid.uuid4().hex

    fold_on = UNFOLD_CONFIGS["enable"]
    probe_on = fold_on


    _m = [{"role": "user", "content": ""}]
    _base = tok.apply_chat_template(_m, add_generation_prompt=False, tokenize=False, enable_thinking=False)
    _full = tok.apply_chat_template(_m, add_generation_prompt=True, tokenize=False, enable_thinking=False)
    assert _full.startswith(_base)


    guide_ids = encode(_full[len(_base):])
    probe_guide_ids = encode(assistant_guide(tok))
    pre_user_ids = encode("<|im_start|>user\n")
    post_user_ids = encode("<|im_end|>\n")
    nl_ids = encode("\n")
    im_end_id = tok.convert_tokens_to_ids("<|im_end|>")
    im_start_id = tok.convert_tokens_to_ids("<|im_start|>")
    split_ids = [im_start_id]

    st = Ledger(request.input_ids, im_start_id)
    swap, subs = [], []
    pids, named = [], []

    def user_para(kind, text):
        body = encode(sanitize(text))
        assert im_start_id not in body
        ids = pre_user_ids + body + post_user_ids
        return st.add_para(kind, ids, [0] * len(ids), [0.0] * len(ids))


    pid2text = {}
    for _, _, text in sample.sessions:
        para = user_para("session", text)
        pid2text[para.pid] = text
        if fold_on:
            para.task = asyncio.create_task(summarize(text))

    user_para("qa", sample.qa)


    if fold_on:
        await send_request(session_id, st.view, url=request.url, max_new_tokens=0, split_ids=split_ids)
        pids = window_fold(st, UNFOLD_CONFIGS["keep_last_k"])

        for p in pids:
            para = st.paras[p]
            para.summary, para.task = await para.task, None

        swap, subs = pids, [[p, []] for p in pids]
        st.apply(swap)


    if probe_on:
        table = "\n".join(f"{st.paras[p].uid}: {st.paras[p].summary}" for p in pids)
        body = encode(sanitize(UNFOLD_CONFIGS["prompt"].replace("{table}", table)))
        assert im_start_id not in body
        probe_ids = pre_user_ids + body + post_user_ids + probe_guide_ids
        result = await send_request(
            session_id,
            st.view + probe_ids,
            url=request.url,
            max_new_tokens=UNFOLD_CONFIGS["probe_max_new_tokens"],
            temperature=UNFOLD_CONFIGS["probe_temperature"],
            top_p=1.0,
            top_k=0,
            stop_ids=[[im_end_id]],
            split_ids=split_ids,
            subs=subs,
            swap=swap,
            ephem_at=len(st.view),
            swap_at=len(st.view))

        gen_ids = result["output_ids"]
        tail_ids = nl_ids if gen_ids and gen_ids[-1] == im_end_id else [im_end_id] + nl_ids
        p_para = st.add_para(
            kind="probe",
            ids=probe_ids + gen_ids + tail_ids,
            mask=[0] * (len(probe_ids) + len(gen_ids) + len(tail_ids)),
            logp=[0.0] * len(probe_ids) + result["output_logprobs"] + [0.0] * len(tail_ids),
            to_view=False,
            as_para=False)
        p_para.text = tok.decode(gen_ids)
        p_para.swap = swap

        probe_swap = pick_unfold(
            tok.decode(gen_ids, skip_special_tokens=True),
            {st.paras[p].uid: p for p in pids})


        selection_mode = UNFOLD_CONFIGS["selection_mode"]
        if selection_mode == "floor":
            probe_swap = []
        elif selection_mode == "random":
            k = len(probe_swap)
            probe_swap = random.sample(pids, k)
        elif selection_mode == "gold":
            explicit_gold = set(getattr(sample, "gold", []))
            if explicit_gold:
                gold_pids = {
                    p for p, (key, _, _) in zip(pid2text, sample.sessions)
                    if key in explicit_gold
                }
            else:
                answers = [str(answer).casefold() for answer in getattr(sample, "answers", [])]
                gold_pids = {
                    p for p, text in pid2text.items()
                    if any(answer and answer in text.casefold() for answer in answers)
                }
            probe_swap = [p for p in pids if p in gold_pids]
        elif selection_mode != "model":
            raise ValueError(f"unknown UNFOLD_SELECTION_MODE={selection_mode!r}")
        named = list(probe_swap)


        bm25_union_top_k = UNFOLD_CONFIGS["bm25_union_top_k"]
        if bm25_union_top_k:
            scores = session_scores(
                question_text(sample.qa), [pid2text[p] for p in pids])
            order = sorted(range(len(pids)), key=lambda i: (-scores[i], i))
            bm25_named = [pids[i] for i in order[:bm25_union_top_k]]
            named = list(dict.fromkeys([*probe_swap, *bm25_named]))

        p_para.selected_ids = [st.paras[p].uid for p in probe_swap]
        p_para.selected_pids = list(named)

        swap, subs = named, []
        st.apply(swap)


    answer_swap = list(swap)


    stats = {"context_tokens": st.context_tokens(), "full_tokens": st.full_tokens()}


    answer_ids = st.view + guide_ids

    result = await send_request(
        session_id,
        answer_ids,
        url=request.url,
        max_new_tokens=LOCOMO_CONFIGS["max_answer_tokens"],
        temperature=request.temperature,
        top_p=request.top_p,
        top_k=request.top_k,
        stop_ids=[[im_end_id]],
        split_ids=split_ids,
        subs=subs,
        swap=swap,
        swap_at=len(answer_ids) - len(guide_ids) if swap else None)

    gen_ids = result["output_ids"]
    a_para = st.add_para(
        kind="assistant",
        ids=guide_ids + gen_ids,
        mask=[0] * len(guide_ids) + [1] * len(gen_ids),
        logp=[0.0] * len(guide_ids) + result["output_logprobs"])
    a_para.text = tok.decode(gen_ids)
    a_para.swap = answer_swap


    for p in st.paras:
        if p.task:
            p.task.cancel()
            p.task = None
    await release(session_id, request.url)

    ordered = st.merge()
    if LOCOMO_CONFIGS["sanity_check"] is True:
        sanity_check(st.tokens, ordered, {
            "session": (pre_user_ids, post_user_ids),
            "qa": (pre_user_ids, post_user_ids),
            "probe": (pre_user_ids, nl_ids),
            "assistant": (guide_ids, [])})

    assert len(st.tokens) == len(st.loss_mask) == len(st.logprobs)

    traj = Traj(
        tokens=st.tokens,
        text=tok.decode(st.view),
        status="truncated" if result["finish_reason"] == "length" else "completed",
        prompt_length=len(request.input_ids),
        loss_mask=st.loss_mask,
        rollout_logprobs=st.logprobs,
        paras=ordered)
    return traj, extract_answer(tok.decode(gen_ids)), stats
