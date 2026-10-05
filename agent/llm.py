def _strip_reasoning(text: str) -> str:
    idx = text.rfind("</think>")
    return text[idx + len("</think>"):].lstrip() if idx != -1 else text


async def chat(client, sem, cfg, prompt: str) -> str:
    payload = {
        "model": cfg["model_name"],
        "messages": [{"role": "user", "content": prompt}],
        **cfg.get("extra_body", {}),
    }
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {cfg['api_key']}",
    }
    async with sem:
        resp = await client.post(cfg["base_url"], headers=headers, json=payload)
    resp.raise_for_status()
    return _strip_reasoning(resp.json()["choices"][0]["message"]["content"])
