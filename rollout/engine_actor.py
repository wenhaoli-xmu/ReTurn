import os

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("FLA_DISABLE_TENSOR_CACHE", "1")

import ray
import torch
import pynvml

from rollout.engine import Engine, Request
from rollout.constant import USE_CUDA_GRAPH


def _load_hf(model_path):
    from transformers import AutoConfig, AutoModelForCausalLM
    config = AutoConfig.from_pretrained(model_path)
    kwargs = {}
    if "qwen3_5_moe" in config.model_type.lower():

        kwargs['experts_implementation'] = 'grouped_mm'
    return AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.bfloat16, **kwargs)


def _pick_model(hf_model):
    from rollout.monkey_patch.qwen35 import QwenModel as Qwen35Model
    from rollout.monkey_patch.qwen3 import QwenModel as Qwen3Model
    mt = hf_model.config.model_type.lower()
    if "qwen3_5" in mt:
        return Qwen35Model
    if "qwen3" in mt:
        return Qwen3Model
    raise ValueError(f"no model_type={mt!r}")


@ray.remote(num_gpus=1)
class EngineActor:
    def __init__(self, engine_id, model_path, max_token, max_reside,
                 offload=True, offload_cpu_gb=16., offload_transfer_mb=64.):
        self.engine_id = engine_id

        pynvml.nvmlInit()
        uuid = str(torch.cuda.get_device_properties(0).uuid)
        self._nvml = pynvml.nvmlDeviceGetHandleByUUID(f"GPU-{uuid}".encode())

        hf_model = _load_hf(model_path)
        hf_model.tie_weights()

        model = _pick_model(hf_model)(
            hf_model,
            max_token=max_token,
            device="cuda:0",
            max_reside=max_reside)

        if offload:
            model.get_kv_cache().configure_offload(
                int(offload_cpu_gb * 2**30), int(offload_transfer_mb * 2**20))

        if USE_CUDA_GRAPH:
            model.build_graph()

        self.engine = Engine(engine_id=engine_id, model=model)

    async def start(self):
        self.engine.start()

    async def submit(self, payload):
        payload = dict(payload)
        req = Request(id=payload.pop("session_id"), **payload)
        r = await self.engine.submit(req)
        out = r.output
        reason = "length" if len(out) >= req.max_new_tokens else "stop"

        return {
            "output_ids": out,
            "output_logprobs": r.output_logprobs,
            "num_tokens": len(out),
            "finish_reason": reason,
        }

    async def release(self, session_id):
        self.engine.release(session_id)

    async def stats(self):
        e = self.engine
        s = e.stats()
        pw = pynvml.nvmlDeviceGetPowerUsage(self._nvml) / 1000 if self._nvml else None
        return {
            "id": e.id,
            "prefilling": len(e.prefilling),
            "decoding": len(e.decoding),
            "reside": len(e.reside),
            "pending": len(e.pending),
            "decode_tps": e.decode_tps,
            "prefill_tps": e.prefill_tps,
            "decode_streak": e.decode_streak,
            "decode_duty": e.decode_duty,
            "prefill_duty": e.prefill_duty,
            "run_time": s["run_time"],
            "power": pw,
            "kv_used": s["page_used"], 
            "kv_total": s["page_total"],
            "kv_offload": s['kv_offload'],
        }
