import types

import torch
import torch.nn.functional as F
from transformers.modeling_outputs import BaseModelOutputWithPast

from rollout.cache import KVCache
from rollout.constant import PAGE_SIZE
from rollout.attention import flash_prefill, flash_decode
from rollout.fused import rms_norm, silu_and_mul
from rollout.model import Model
from rollout.prefill import DECODE_CONTEXT


def _fused_rmsnorm_forward(self, x):

    return rms_norm(x, self.weight, self.variance_epsilon, offset=0.0)


def _fused_mlp_forward(self, x):
    gu = F.linear(x, self._gate_up_weight)
    return self.down_proj(silu_and_mul(gu))


def attention_forward(self, hidden_states, **kwargs):
    input_shape = hidden_states.shape[:-1]


    query_states = self.q_norm(self.q_proj(hidden_states).view(*input_shape, -1, self.head_dim))
    key_states = self.k_norm(self.k_proj(hidden_states).view(*input_shape, -1, self.head_dim))
    value_states = self.v_proj(hidden_states).view(*input_shape, -1, self.head_dim)

    kv = kwargs["kv"]
    ctx = kwargs.get("forward_ctx", DECODE_CONTEXT)
    li = self.kv_layer_idx
    cos, sin = kwargs["rope"]["cos"], kwargs["rope"]["sin"]
    D = self.head_dim

    if ctx.is_prefill:
        k = key_states.detach()[0].contiguous()
        v = value_states.detach()[0].contiguous()
        kv.write(li, k, v, ctx.T)
        q = query_states[0]
        o = flash_prefill(q, kv.page_table, kv.mask_table, kv.cu_pages, kv.cu_q, kv.qpos, kv.page_pos,
                          kv.k_pool[li], kv.v_pool[li], cos, sin, kv.num_kv_heads, ctx.n, self.scaling)
    else:
        n = query_states.shape[0]
        k = key_states.contiguous().view(n, -1, D)
        v = value_states.contiguous().view(n, -1, D)
        kv.write(li, k, v, ctx.T)
        q = query_states.view(n, -1, D)
        o = flash_decode(q, kv.page_table, kv.mask_table, kv.cu_pages, kv.qpos, kv.page_pos,
                         kv.k_pool[li], kv.v_pool[li], cos, sin, kv.num_kv_heads, ctx.num_splits, self.scaling)

    attn_output = o.reshape(*input_shape, -1).contiguous()
    return self.o_proj(attn_output), None


def layer_forward(self, hidden_states, **kwargs):
    residual = hidden_states
    hidden_states = self.input_layernorm(hidden_states)
    hidden_states, _ = self.self_attn(hidden_states=hidden_states, **kwargs)
    hidden_states = residual + hidden_states

    residual = hidden_states
    hidden_states = self.post_attention_layernorm(hidden_states)
    hidden_states = residual + self.mlp(hidden_states)
    return hidden_states


def model_forward(self, input_ids=None, inputs_embeds=None, **kwargs):
    ctx = kwargs.get("forward_ctx", DECODE_CONTEXT)
    if inputs_embeds is None:
        inputs_embeds = self.embed_tokens(input_ids)
    hidden_states = inputs_embeds
    kwargs["rope"] = {"cos": self._rope_cos, "sin": self._rope_sin}
    kwargs["forward_ctx"] = ctx
    for layer in self.layers:
        hidden_states = layer(hidden_states, **kwargs)
    hidden_states = self.norm(hidden_states)
    return BaseModelOutputWithPast(last_hidden_state=hidden_states)


def causal_forward(self, input_ids=None, **kwargs):
    return self.model(input_ids=input_ids, **kwargs)


def monkey_patch(model):
    model.forward = types.MethodType(causal_forward, model)
    model.model.forward = types.MethodType(model_forward, model.model)
    for layer in model.model.layers:
        layer.forward = types.MethodType(layer_forward, layer)
        layer.self_attn.forward = types.MethodType(attention_forward, layer.self_attn)

    for m in model.modules():
        if type(m).__name__ == "Qwen3RMSNorm":
            m.forward = types.MethodType(_fused_rmsnorm_forward, m)

    for m in model.modules():
        if (hasattr(m, "gate_proj") and hasattr(m, "up_proj") and hasattr(m, "down_proj")
                and getattr(m.gate_proj, "bias", None) is None):
            m.register_buffer("_gate_up_weight",
                              torch.cat([m.gate_proj.weight, m.up_proj.weight], dim=0).detach(),
                              persistent=False)
            m.forward = types.MethodType(_fused_mlp_forward, m)
    return model


class QwenModel(Model):
    def __init__(self, hf_model, max_token, max_reside, device="cuda"):
        super().__init__(hf_model, max_token, max_reside, device)
        self.model = monkey_patch(hf_model).to(device).eval()

        cfg = hf_model.config
        self.vocab_size = cfg.vocab_size
        kv_heads = cfg.num_key_value_heads
        head_dim = getattr(cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads)
        self.num_kv_heads = kv_heads


        inner = self.model.model
        rot = inner.rotary_emb
        inv_freq = rot.inv_freq.to(device=device, dtype=torch.float32)
        pos = torch.arange(max_token + PAGE_SIZE, device=device, dtype=torch.float32)
        freqs = torch.outer(pos, inv_freq)
        inner._rope_cos = (freqs.cos() * rot.attention_scaling).to(torch.bfloat16).contiguous()
        inner._rope_sin = (freqs.sin() * rot.attention_scaling).to(torch.bfloat16).contiguous()


        layers = hf_model.model.layers
        assert all(getattr(l.self_attn, "sliding_window", None) is None for l in layers), \
            "Qwen3 sliding-window attention  not implemented （paged kernel  only full  causal）"
        self.kv = KVCache(len(layers), max_token, kv_heads, head_dim, max_reside, device)
        for i, l in enumerate(layers):
            l.self_attn.kv_layer_idx = i

        self.lin_caches = []

    def get_kv_cache(self):
        return self.kv

    def get_lin_caches(self):
        return self.lin_caches

    def forward_logits(self, input_ids, ctx=DECODE_CONTEXT, req_cu=None):
        h = self.model(
            input_ids=input_ids,
            kv=self.kv,
            lin_caches=self.lin_caches,
            forward_ctx=ctx).last_hidden_state
        if ctx.is_prefill:
            out_pos = torch.tensor([c - 1 for c in req_cu[1:]], device=self.device)
            h = h[0].index_select(0, out_pos)
        else:
            h = h[:, -1]
        return self.model.lm_head(h)
