import types

import torch
import torch.nn.functional as F
from transformers.modeling_outputs import BaseModelOutputWithPast

from rollout.cache import KVCache, LinearStateCache
from rollout.constant import PAGE_SIZE
from rollout.attention import flash_prefill, flash_decode
from rollout.fused import rms_norm, rms_norm_gated, silu_and_mul
from rollout.model import Model
from rollout.prefill import DECODE_CONTEXT


def _fused_rmsnorm_forward(self, x):
    return rms_norm(x, self.weight, self.eps, offset=1.0)


def _fused_mlp_forward(self, x):
    gu = F.linear(x, self._gate_up_weight)
    return self.down_proj(silu_and_mul(gu))


def _fused_rmsnorm_gated_forward(self, hidden_states, gate=None):
    return rms_norm_gated(hidden_states, self.weight, gate, self.variance_epsilon)


def attention_forward(self, hidden_states, **kwargs):
    input_shape = hidden_states.shape[:-1]

    query_states, gate = torch.chunk(
        self.q_proj(hidden_states).view(*input_shape, -1, self.head_dim * 2), 2, dim=-1)
    gate = gate.reshape(*input_shape, -1)


    query_states = self.q_norm(query_states)
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
    attn_output = attn_output * torch.sigmoid(gate)
    return self.o_proj(attn_output), None


def _split_qkv(self, mixed_qkv, B, L):
    mixed_qkv = mixed_qkv.transpose(1, 2)
    q, k, v = torch.split(mixed_qkv, [self.key_dim, self.key_dim, self.value_dim], dim=-1)
    q = q.reshape(B, L, -1, self.head_k_dim)
    k = k.reshape(B, L, -1, self.head_k_dim)
    v = v.reshape(B, L, -1, self.head_v_dim)
    return q, k, v


def linear_attn_forward(self, hidden_states, cache, ctx=None):

    B, L, _ = hidden_states.shape
    ctx = ctx or DECODE_CONTEXT
    mixed_qkv = self.in_proj_qkv(hidden_states).transpose(1, 2)
    z = self.in_proj_z(hidden_states).reshape(B, L, -1, self.head_v_dim)
    beta = self.in_proj_b(hidden_states).sigmoid()
    g = -self.A_log.float().exp() * F.softplus(self.in_proj_a(hidden_states).float() + self.dt_bias)
    K = self.conv_kernel_size

    if not ctx.is_prefill:
        conv_prev, rec_prev = cache.gather()
        mixed_qkv = self.causal_conv1d_update(
            mixed_qkv, conv_prev, self.conv1d.weight.squeeze(1), self.conv1d.bias, self.activation)
        q, k, v = _split_qkv(self, mixed_qkv, B, L)
        core, last = self.recurrent_gated_delta_rule(
            q, k, v, g=g, beta=beta, initial_state=rec_prev,
            output_final_state=True, use_qk_l2norm_in_kernel=True)
        cache.scatter(conv_prev, last)
    else:
        cv = ctx.conv
        C, Nreq = mixed_qkv.shape[1], cv["N"]
        hist, init = cache.prefill_state(ctx.sids, mixed_qkv)
        combined = mixed_qkv.new_zeros(C, cv["M"])
        combined.index_copy_(1, cv["hist_slot"], hist.permute(1, 0, 2).reshape(C, -1))
        combined.index_copy_(1, cv["x_slot"], mixed_qkv[0])
        full = F.conv1d(combined[None], self.conv1d.weight, self.conv1d.bias, groups=self.conv1d.groups)
        conv_out = F.silu(full[0].index_select(1, cv["out"]))[None]
        conv_new = combined.index_select(1, cv["cn_slot"]).reshape(C, Nreq, K).permute(1, 0, 2)

        q, k, v = _split_qkv(self, conv_out, B, L)
        q = q.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)
        k = k.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)
        core, last = self.chunk_gated_delta_rule(
            q, k, v, g=g, beta=beta, initial_state=init, output_final_state=True,
            use_qk_l2norm_in_kernel=True, cu_seqlens=cv["cu"])
        cache.scatter_prefill(ctx.sids, conv_new.contiguous(), last)

    core = self.norm(core.reshape(-1, self.head_v_dim), z.reshape(-1, self.head_v_dim))
    return self.out_proj(core.reshape(B, L, -1))


def layer_forward(self, hidden_states, **kwargs):
    residual = hidden_states
    hidden_states = self.input_layernorm(hidden_states)

    if self.layer_type == "linear_attention":
        hidden_states = self.linear_attn(
            hidden_states,
            kwargs["lin_caches"][self.linear_attn.gdn_idx],
            kwargs.get("forward_ctx", DECODE_CONTEXT))
    else:
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
        if layer.layer_type == "full_attention":
            layer.self_attn.forward = types.MethodType(attention_forward, layer.self_attn)
        else:
            layer.linear_attn.forward = types.MethodType(linear_attn_forward, layer.linear_attn)

    for m in model.modules():
        cls = type(m).__name__
        if cls in ("Qwen3_5RMSNorm", "Qwen3_5MoeRMSNorm"):
            m.forward = types.MethodType(_fused_rmsnorm_forward, m)
        elif cls in ("Qwen3_5RMSNormGated", "Qwen3_5MoeRMSNormGated"):
            m.forward = types.MethodType(_fused_rmsnorm_gated_forward, m)

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
        num_full = sum(1 for l in layers if l.layer_type == "full_attention")
        self.kv = KVCache(num_full, max_token, kv_heads, head_dim, max_reside, device)
        i = 0
        for l in layers:
            if l.layer_type == "full_attention":
                l.self_attn.kv_layer_idx = i
                i += 1

        la = next(l.linear_attn for l in layers if l.layer_type == "linear_attention")
        self.conv_K = la.conv_kernel_size
        gdn_dims = (la.num_v_heads, la.head_k_dim, la.head_v_dim, la.conv1d.weight.shape[0], la.conv_kernel_size)
        self.lin_caches = []
        j = 0
        for l in layers:
            if l.layer_type == "linear_attention":
                l.linear_attn.gdn_idx = j
                self.lin_caches.append(LinearStateCache(device, max_reside, gdn_dims))
                j += 1

    def get_kv_cache(self):
        return self.kv

    def get_lin_caches(self):
        return self.lin_caches

    def _conv_index(self, req_cu):


        K, dev = self.conv_K, self.device
        Km1, N, T = K - 1, len(req_cu) - 1, int(req_cu[-1])
        cu = torch.as_tensor(req_cu, dtype=torch.long, device=dev)
        seg, blk = cu[1:] - cu[:-1], torch.arange(N, device=dev)
        head = cu[:-1] + blk * Km1
        x_slot = torch.arange(T, device=dev) + (blk.repeat_interleave(seg) + 1) * Km1
        r = torch.arange(K, device=dev)
        return {
            "N": N, "M": T + N * Km1, "K": K,
            "cu": cu.to(torch.int32),
            "x_slot": x_slot,
            "out": x_slot - Km1,
            "hist_slot": (head[:, None] + r[:Km1]).reshape(-1),
            "cn_slot": (head[:, None] + seg[:, None] - 1 + r).reshape(-1),
        }

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
