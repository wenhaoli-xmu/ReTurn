import re


PROMPT = """The table lists passages currently hidden from your context. The
digests are brief and may omit useful details.
id: digest
{table}
Select every passage that could help answer the question, including
passages that may supply missing context. If you are unsure whether a
passage is needed, include it. There is no fixed number to select.
Reply only with selected uppercase ids from the table, each at most
once. Separate ids with spaces. Use a single line; no commas,
explanations, answers, or tool calls. If none are needed, output NONE."""

RESPONSE_PREFIX = "Selected ids (space-separated, no reasoning):"


def alphabetic_id(index):

    if index < 0:
        raise ValueError("ID index must be nonnegative")
    label = ""
    index += 1
    while index:
        index, remainder = divmod(index - 1, 26)
        label = chr(ord("A") + remainder) + label
    return label


def parse_ids(text):
    text = text.strip()
    if text in ("", "NONE"):
        return []
    if re.fullmatch(r"[A-Z]+(?: +[A-Z]+)*", text) is None:
        return []
    return list(dict.fromkeys(text.split()))


def assistant_guide(tokenizer):
    messages = [{"role": "user", "content": ""}]
    base = tokenizer.apply_chat_template(
        messages, add_generation_prompt=False, tokenize=False, enable_thinking=False)
    full = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False, enable_thinking=False)
    if not full.startswith(base):
        raise ValueError("Cannot extract the no-thinking assistant prefix")
    return full[len(base):] + RESPONSE_PREFIX
