import json
import random
from dataclasses import dataclass


CONV_START_PROMPT = "Below is a conversation between two people: {} and {}. The conversation takes place over multiple days and the date of each conversation is wriiten at the beginning of the conversation.\n\n"

QA_PROMPT = """
Based on the above context, write an answer in the form of a short phrase for the following question. Answer with exact words from the context whenever possible.

Question: {} Short answer:
"""

QA_PROMPT_CAT_5 = """
Based on the above context, answer the following question.

Question: {} Short answer:
"""

CAT_5_QUESTION = "{} Select the correct answer: (a) {} (b) {}. "

CAT_2_HINT = " Use DATE of CONVERSATION to answer with an approximate date."

NOT_MENTIONED = "Not mentioned in the conversation"


@dataclass
class Sample:
    conv_id: str
    qa_index: int
    start: str
    sessions: list
    question: str
    qa: str
    answer: object
    category: int
    evidence: list
    gold: list
    cat5_key: dict | None = None


def session_block(conv, i):
    out = "DATE: " + conv["session_%s_date_time" % i] + "\n" + "CONVERSATION:\n"
    for d in conv["session_%s" % i]:
        out += d["speaker"] + ' said, "' + d["text"] + '"' + "\n"
        if "blip_caption" in d:
            out += " and shared %s." % d["blip_caption"]
        out += "\n"
    return out


def sessions_of(conv):
    nums = sorted(int(k.split("_")[-1]) for k in conv if k.startswith("session_") and "date_time" not in k)
    return [(i, conv["session_%s_date_time" % i], session_block(conv, i)) for i in nums]


def start_prompt(conv):

    names = list(dict.fromkeys(d["speaker"] for d in conv["session_1"]))
    return CONV_START_PROMPT.format(names[0], names[1])


def qa_prompt(question, category):
    return (QA_PROMPT_CAT_5 if category == 5 else QA_PROMPT).format(question)


def prompt_ids(sample, tok):

    text = "<|im_start|>system\n" + sample.start + "<|im_end|>\n"
    return tok(text, add_special_tokens=False)["input_ids"]


def load(path, seed):

    out = []
    for data in json.load(open(path)):
        conv = data["conversation"]
        start, sessions = start_prompt(conv), sessions_of(conv)
        for k, qa in enumerate(data["qa"]):
            question, key = qa["question"], None


            answer = qa.get("answer", qa.get("adversarial_answer"))
            if qa["category"] == 2:
                question += CAT_2_HINT
            elif qa["category"] == 5:
                bait = qa["adversarial_answer"]
                rng = random.Random(f"{seed}\0{data['sample_id']}\0{k}")
                a, b = (NOT_MENTIONED, bait) if rng.random() < 0.5 else (bait, NOT_MENTIONED)
                question, key = CAT_5_QUESTION.format(question, a, b), {"a": a, "b": b}
            out.append(Sample(
                conv_id=data["sample_id"],
                qa_index=k,
                start=start,
                sessions=sessions,
                question=question,
                qa=qa_prompt(question, qa["category"]),
                answer=answer,
                category=qa["category"],
                evidence=qa.get("evidence", []),
                gold=sorted({
                    int(str(e).lstrip("D").split(":")[0])
                    for e in qa.get("evidence", [])
                    if str(e).lstrip("D").split(":")[0].isdigit()
                }),
                cat5_key=key))
    return out


def cat5_answer(prediction, key):

    p = prediction.strip().lower()
    if len(p) == 1:
        return key["a"] if "a" in p else key["b"]
    if len(p) == 3:
        return key["a"] if "(a)" in p else key["b"]
    return p
