import asyncio
import json
import logging
import random
import re
import uuid

import httpx
from tenacity import retry, stop_after_attempt, wait_random_exponential, retry_if_exception

from agent.utils import is_retryable_http_exc
from agent.data import Para, Request, Traj
from agent.probe import alphabetic_id, parse_ids, assistant_guide
from agent.locomo.bm25 import session_scores
from agent.search.config import AGENT_CONFIGS, UNFOLD_CONFIGS
from agent.search.search import search
from agent.search.fetch import fetch
from agent.search.summarize import summarize

logger = logging.getLogger(__name__)


def pick_unfold(text, paras, uid2pid):
    return list(dict.fromkeys(
        p for u in parse_ids(text)
        if (p := uid2pid.get(u)) is not None and paras[p].folded))


def default_fold(paras):
    n, win = len(paras), UNFOLD_CONFIGS["keep_last_k"]
    keep = set(range(max(0, n - win), n))
    return [p for p in range(n) if p not in keep and paras[p].kind != "prompt"]


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
async def _release(session_id: str, url: str) -> None:
    await _gen_client.post(url.rsplit("/generate", 1)[0] + "/release", json={"session_id": session_id})


def _dispatch_tool(name, params):
    if not isinstance(params, dict):
        return None, None
    if name == "search" and set(params) == {"keyword"}:
        return search, {"keyword": params["keyword"]}
    if name == "fetch" and set(params) == {"url", "query"}:
        return fetch, {"url": params["url"], "query": params["query"]}
    return None, None


def tool_parse(prediction: str):
    tool = re.search(r"<tool_call>(.*?)</tool_call>", prediction, re.DOTALL)
    if tool is None:
        return None, None

    body = tool.group(1).strip()


    if fm := re.fullmatch(r"<function\s*=\s*([^>]+?)\s*>(.*)</function>", body, re.DOTALL):
        name = fm.group(1).strip()
        params = {
            pm.group(1).strip(): pm.group(2).strip()
            for pm in re.finditer(r"<parameter\s*=\s*([^>]+?)\s*>(.*?)</parameter>", fm.group(2), re.DOTALL)
        }
        return _dispatch_tool(name, params)


    try:
        obj = json.loads(body)
    except (json.JSONDecodeError, ValueError):
        return None, None
    if isinstance(obj, dict):
        return _dispatch_tool(obj.get("name"), obj.get("arguments"))

    return None, None


_SPECIAL_RE = re.compile(r"<\|(?:im_start|im_end|endoftext)\|>|</tool_response>")


def _sanitize(text):

    return _SPECIAL_RE.sub(
        lambda m: m.group(0
        ).replace("<|", "<"
        ).replace("|>", ">"
        ).replace("</tool_response>", "<-/tool_response>"), text)


def sanity_check(tokens, paras, table):

    ends = [p.start for p in paras[1:]] + [len(tokens)]
    for i, (p, b) in enumerate(zip(paras, ends)):
        if p.kind == "prompt":
            continue
        pre, suf = table[p.kind]
        seg = tokens[p.start:b]
        assert seg[:len(pre)] == pre, f" paragraph  {i}({p.kind})  prefix mismatch "
        if i == len(paras) - 1:
            continue
        assert seg[len(seg) - len(suf):] == suf, f" paragraph  {i}({p.kind})  suffix mismatch "


async def tool_execute(prediction: str):
    fcn, kwargs = tool_parse(prediction)

    try:
        if fcn is not None:
            results = await fcn(**kwargs)
            next_obs = f"\n{str(results).strip()}\n"
            done = False
        else:
            next_obs = ""
            done = True
    except Exception as e:
        logger.warning(f"Tool execution failed: {e}", exc_info=True)
        next_obs = f"\n{str(e)}\n"
        done = False

    return next_obs, done


async def generate(request: Request) -> Traj:
    request = request.with_sampling(**AGENT_CONFIGS["sampling"])
    tok = request.tokenizer
    encode = lambda x: tok(x, add_special_tokens=False)['input_ids']

    session_id = uuid.uuid4().hex
    single_turn_max_length = AGENT_CONFIGS["single_turn_max_length"]
    max_turns = AGENT_CONFIGS["max_turns"]
    max_length = AGENT_CONFIGS["max_length"]
    use_active_context_length = AGENT_CONFIGS["use_active_context_length"]


    _m = [{"role": "user", "content": ""}]
    _base = tok.apply_chat_template(_m, add_generation_prompt=False, tokenize=False)
    _full = tok.apply_chat_template(_m, add_generation_prompt=True, tokenize=False)
    assert _full.startswith(_base)
    guide_prompt = _full[len(_base):]
    probe_guide_prompt = assistant_guide(tok)


    guide_ids = encode(guide_prompt)
    probe_guide_ids = encode(probe_guide_prompt)
    pre_user_ids = encode("<|im_start|>user\n")
    post_user_ids = encode("<|im_end|>\n")
    post_toolcall_ids = encode("<|im_end|>\n")
    pre_obs_ids = encode("<|im_start|>user\n<tool_response>")
    post_obs_ids = encode("</tool_response><|im_end|>\n")
    im_end_id = tok.convert_tokens_to_ids("<|im_end|>")
    stop_ids = [encode("</tool_call>"), [im_end_id]]
    im_start_id = tok.convert_tokens_to_ids("<|im_start|>")
    split_ids = [im_start_id]


    unfold_on = UNFOLD_CONFIGS["enable"]
    fold_on = unfold_on

    nl_ids = encode("\n")

    assert request.input_ids[-len(guide_ids):] == guide_ids


    tokens = list(request.input_ids[:-len(guide_ids)])

    view = list(tokens)
    prompt_length = len(tokens)
    loss_mask = [0] * len(tokens)
    rollout_logprobs = [0.0] * len(tokens)

    assert tokens and tokens[0] == im_start_id


    paras = [Para(start=s, kind="prompt", pid=n)
             for n, s in enumerate(i for i, t in enumerate(tokens) if t == im_start_id)]
    probes = []
    uid2pid = {}
    para_sizes = {}

    def add_para(kind, ids, mask, logp, to_view=True):
        para = Para(start=len(tokens), kind=kind)


        tokens.extend(ids)
        loss_mask.extend(mask)
        rollout_logprobs.extend(logp)

        if not to_view:
            probes.append(para)
            return para
        view.extend(ids)


        uid = alphabetic_id(len(uid2pid))
        para.pid, para.uid = len(paras), uid
        uid2pid[uid] = para.pid

        paras.append(para)
        para_sizes[para.pid] = len(ids)
        return para

    async def build_fold():
        pids = default_fold(paras)
        swap, subs = [], []
        for p in pids:
            para = paras[p]
            if para.summary is None:

                para.summary = await para.task
                para.task = None
                subs.append([p, []])
            if not para.folded:
                swap.append(p)
        return pids, swap, subs

    def apply(swap):
        for p in swap:
            paras[p].folded = not paras[p].folded

    def context_length(swap=()):
        if not use_active_context_length:
            return len(view)
        toggled = set(swap)
        return prompt_length + sum(
            para_sizes[p.pid] for p in paras
            if p.kind != "prompt" and not (p.folded ^ (p.pid in toggled)))

    status = None
    last_turn_key = None
    consecutive_same_turns = 0


    for _ in range(max_turns):


        fold_pids, fold_swap, subs, probe_text = [], [], [], ""
        if fold_on:
            fold_pids, fold_swap, subs = await build_fold()


        probe_on = unfold_on and bool(fold_pids)
        probe_ids = []
        if probe_on:
            table = "\n".join(f"{paras[p].uid}: {paras[p].summary}" for p in fold_pids)
            body = encode(_sanitize(UNFOLD_CONFIGS["prompt"].replace("{table}", table)))
            assert im_start_id not in body
            probe_ids = pre_user_ids + body + post_user_ids + probe_guide_ids


        reserve = len(guide_ids) + 8
        if probe_on:
            probe_reserve = (
                len(probe_ids)
                + UNFOLD_CONFIGS["probe_max_new_tokens"]
                + len(nl_ids)
                + 1
            )
            reserve = max(reserve, probe_reserve)
        if max_length - context_length(fold_swap) - reserve <= 0:
            status = "truncated"
            break


        if probe_on:


            result = await send_request(
                session_id,
                view + probe_ids,
                url=request.url,
                max_new_tokens=UNFOLD_CONFIGS["probe_max_new_tokens"],
                temperature=UNFOLD_CONFIGS["probe_temperature"],
                top_p=1.0,
                top_k=0,
                stop_ids=[[im_end_id]],
                split_ids=split_ids,
                subs=subs,
                swap=fold_swap,


                ephem_at=len(view),
                swap_at=len(view))
            apply(fold_swap)

            gen_ids = result["output_ids"]
            tail_ids = nl_ids if gen_ids and gen_ids[-1] == im_end_id else [im_end_id] + nl_ids


            p_para = add_para(
                kind="probe",
                ids=probe_ids + gen_ids + tail_ids,
                mask=[0] * (len(probe_ids) + len(gen_ids) + len(tail_ids)),
                logp=[0.0] * len(probe_ids) + result["output_logprobs"] + [0.0] * len(tail_ids),
                to_view=False)
            p_para.text = tok.decode(gen_ids)
            p_para.swap = fold_swap
            fold_swap, subs = [], []

            probe_text = tok.decode(gen_ids, skip_special_tokens=True)


        probe_swap = pick_unfold(probe_text, paras, uid2pid)


        selection_mode = UNFOLD_CONFIGS["selection_mode"]
        if selection_mode == "floor":
            probe_swap = []
        elif selection_mode == "random":
            k = len(probe_swap)
            probe_swap = random.sample(fold_pids, k)
        elif selection_mode != "model":
            raise ValueError(f"unknown UNFOLD_SELECTION_MODE={selection_mode!r}")


        unfold_swap = list(probe_swap)


        if probe_on and UNFOLD_CONFIGS["bm25_union_top_k"]:
            docs = [paras[p].text or "" for p in fold_pids]
            scores = session_scores(_SPECIAL_RE.sub("", text), docs)
            order = sorted(range(len(fold_pids)), key=lambda i: (-scores[i], i))
            bm25_swap = [
                fold_pids[i]
                for i in order[:UNFOLD_CONFIGS["bm25_union_top_k"]]
            ]
            unfold_swap = list(dict.fromkeys([*probe_swap, *bm25_swap]))


        if probe_on:
            p_para.selected_ids = [paras[p].uid for p in probe_swap]
            p_para.selected_pids = list(unfold_swap)
        swap = fold_swap + unfold_swap

        budget = max_length - context_length(swap) - len(guide_ids) - 8
        if budget <= 0:
            status = "truncated"
            break
        result = await send_request(
            session_id,
            view + guide_ids,
            url=request.url,
            max_new_tokens=min(single_turn_max_length, budget),
            temperature=request.temperature,
            top_p=request.top_p,
            top_k=request.top_k,
            stop_ids=stop_ids,
            split_ids=split_ids,
            subs=subs,
            swap=swap,
            swap_at=len(view) if swap else None,
        )
        apply(unfold_swap)

        gen_ids = result["output_ids"]
        stop_reason = result["finish_reason"]
        text = tok.decode(gen_ids)

        a_para = add_para(
            kind="assistant",
            ids=guide_ids + gen_ids + post_toolcall_ids,
            mask=[0] * len(guide_ids) + [1] * len(gen_ids) + [0] * len(post_toolcall_ids),
            logp=[0.0] * len(guide_ids) + result["output_logprobs"] + [0.0] * len(post_toolcall_ids))
        a_para.text = text
        a_para.swap = swap


        if stop_reason == "length":
            status = "truncated"
            break


        key = tuple(gen_ids)
        if key == last_turn_key:
            consecutive_same_turns += 1
        else:
            last_turn_key = key
            consecutive_same_turns = 1
        if consecutive_same_turns >= 3:
            status = "repeat"
            break


        next_obs, done = await tool_execute(text)


        if done:

            n = len(post_toolcall_ids)
            del tokens[-n:], view[-n:], loss_mask[-n:], rollout_logprobs[-n:]
            status = "completed"
            break


        if fold_on:
            a_para.task = asyncio.create_task(summarize(text, kind="assistant"))


        body_ids = encode(_sanitize(next_obs))
        assert im_start_id not in body_ids

        o_para = add_para(
            kind="obs", 
            ids=pre_obs_ids + body_ids + post_obs_ids,
            mask=[0] * (len(pre_obs_ids) + len(body_ids) + len(post_obs_ids)),
            logp=[0.0] * (len(pre_obs_ids) + len(body_ids) + len(post_obs_ids)))
        o_para.text = next_obs


        if fold_on:
            o_para.task = asyncio.create_task(summarize(next_obs, kind="obs"))

    if status is None:
        status = "truncated"


    for p in paras:
        if p.task:
            p.task.cancel()
            p.task = None
    await _release(session_id, request.url)


    ordered = sorted(paras + probes, key=lambda p: p.start)

    if AGENT_CONFIGS['sanity_check'] is True:
        table = {
            "probe": (pre_user_ids, nl_ids),
            "assistant": (guide_ids, post_toolcall_ids),
            "obs": (pre_obs_ids, post_obs_ids)}
        sanity_check(tokens, ordered, table)

    assert len(tokens) == len(loss_mask) == len(rollout_logprobs)

    return Traj(
        tokens=tokens,
        text=tok.decode(view),
        status=status,
        prompt_length=prompt_length,
        loss_mask=loss_mask,
        rollout_logprobs=rollout_logprobs,
        paras=ordered,
    )
