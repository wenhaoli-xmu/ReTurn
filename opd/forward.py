import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from rollout.constant import PAGE_SIZE


def flash_prefill_func(*args):
    from rollout.attention import flash_prefill_func as kernel
    return kernel(*args)


@dataclass
class Runtime:

    body: object
    head: object
    cos: torch.Tensor
    sin: torch.Tensor
    nkv: int
    device: torch.device

    @classmethod
    def of(cls, model, device, max_pos):
        nkv = model.config.get_text_config().num_key_value_heads
        cos, sin = build_rope(model, max_pos, device)
        return cls(
            getattr(model, model.base_model_prefix),
            model.get_output_embeddings(),
            cos,
            sin,
            nkv,
            device)

    def forward(self, ids, ctx_kv=None, gdn_init=None):
        return body_forward(
            self.body,
            torch.tensor(ids, device=self.device)[None],
            ctx_kv,
            gdn_init,
            self.cos,
            self.sin,
            self.nkv)

    def forward_parallel(self, units):
        layout = ReplayLayout.of(units, self.device)
        tokens = torch.tensor([token for unit in units for token in unit.ids],
                              device=self.device, dtype=torch.long)[None]
        h = self.body.embed_tokens(tokens)
        for layer in self.body.layers:
            def step(h, layer=layer):
                residual = h
                hn = layer.input_layernorm(h)
                if layer.layer_type == "linear_attention":
                    out, _, _ = gdn_forward(layer.linear_attn, hn, None, None)
                else:
                    out = parallel_attention(layer.self_attn, hn, layout,
                                             self.cos, self.sin, self.nkv)
                h = residual + out
                return h + layer.mlp(layer.post_attention_layernorm(h))

            h = checkpoint(step, h, use_reentrant=False) if torch.is_grad_enabled() else step(h)
        return list(self.body.norm(h)[0].split(layout.lengths))


@dataclass
class ReplayLayout:
    lengths: list
    pool_slots: torch.Tensor
    pool_pages: int
    page_table: torch.Tensor
    mask_table: torch.Tensor
    cu_pages: torch.Tensor
    cu_q: torch.Tensor
    qpos: torch.Tensor
    page_pos: torch.Tensor

    @classmethod
    def of(cls, units, device):
        lengths = [len(unit.ids) for unit in units]
        if not lengths or any(length <= 0 for length in lengths):
            raise ValueError("Replay requires nonempty formal turns")
        pages, masks, slots, next_page = [], [], [], 0
        for length in lengths:
            count = (length + PAGE_SIZE - 1) // PAGE_SIZE
            pages.append(list(range(next_page, next_page + count)))
            masks.append([min(PAGE_SIZE, length - i * PAGE_SIZE) for i in range(count)])
            slots.extend(range(next_page * PAGE_SIZE, next_page * PAGE_SIZE + length))
            next_page += count
        table, valid, positions, cu_pages, cu_q, qpos = [], [], [], [0], [0], []
        for i, unit in enumerate(units):
            if unit.ctx != sorted(set(unit.ctx)) or any(j < 0 or j >= i for j in unit.ctx):
                raise ValueError("Replay history must be unique, chronological, and causal")
            offset = 0
            for j in [*unit.ctx, i]:
                for page, mask in zip(pages[j], masks[j]):
                    table.append(page)
                    valid.append(mask)
                    positions.append(offset)
                    offset += mask
            qpos.append(offset - lengths[i])
            cu_pages.append(len(table))
            cu_q.append(cu_q[-1] + lengths[i])
        ints = lambda values: torch.tensor(values, device=device, dtype=torch.int32)
        return cls(lengths, torch.tensor(slots, device=device, dtype=torch.long),
                   next_page, ints(table), torch.tensor(valid, device=device, dtype=torch.uint8),
                   ints(cu_pages), ints(cu_q), ints(qpos), ints(positions))


def parallel_attention(attn, hidden, layout, cos, sin, num_kv_heads):
    shape = hidden.shape[:-1]
    D = attn.head_dim
    q, gate = torch.chunk(attn.q_proj(hidden).view(*shape, -1, D * 2), 2, dim=-1)
    gate = gate.reshape(*shape, -1)
    q = attn.q_norm(q)[0].contiguous()
    k = attn.k_norm(attn.k_proj(hidden).view(*shape, -1, D))[0]
    v = attn.v_proj(hidden).view(*shape, -1, D)[0]
    pool_shape = (layout.pool_pages * PAGE_SIZE, num_kv_heads, D)
    kpool = k.new_zeros(pool_shape).index_copy(0, layout.pool_slots, k)
    vpool = v.new_zeros(pool_shape).index_copy(0, layout.pool_slots, v)
    scale = getattr(attn, "scaling", 1.0 / math.sqrt(D))
    o = flash_prefill_func(
        q, kpool.view(-1, PAGE_SIZE, num_kv_heads, D),
        vpool.view(-1, PAGE_SIZE, num_kv_heads, D),
        layout.page_table, layout.mask_table, layout.cu_pages, layout.cu_q,
        layout.qpos, layout.page_pos, cos, sin, num_kv_heads, len(layout.lengths), scale)
    return attn.o_proj(o.reshape(*shape, -1) * torch.sigmoid(gate))


def build_rope(model, max_pos, device, dtype=torch.bfloat16):
    rot = model.model.rotary_emb
    inv = rot.inv_freq.to(device=device, dtype=torch.float32)
    pos = torch.arange(max_pos, device=device, dtype=torch.float32)
    freqs = torch.outer(pos, inv)
    scale = getattr(rot, "attention_scaling", 1.0)
    cos = (freqs.cos() * scale).to(dtype).contiguous()
    sin = (freqs.sin() * scale).to(dtype).contiguous()
    return cos, sin


def _pool(kv, Lc, device):

    P = PAGE_SIZE
    total, Hkv, D = kv.shape
    npages = (total + P - 1) // P
    pad = npages * P - total
    if pad:
        kv = F.pad(kv, (0, 0, 0, 0, 0, pad))
    return kv.view(npages, P, Hkv, D), npages, total


def paged_attention(attn, hidden, ctx_k, ctx_v, cos, sin, num_kv_heads):

    P = PAGE_SIZE
    shape = hidden.shape[:-1]
    D = attn.head_dim
    q, gate = torch.chunk(
        attn.q_proj(hidden).view(*shape, -1, D * 2), 2, dim=-1)
    gate = gate.reshape(*shape, -1)
    q = attn.q_norm(q)[0]
    k = attn.k_norm(attn.k_proj(hidden).view(*shape, -1, D))[0]
    v = attn.v_proj(hidden).view(*shape, -1, D)[0]
    Lq = q.shape[0]
    Lc = ctx_k.shape[0] if ctx_k is not None else 0

    full_k = torch.cat([ctx_k, k], 0) if Lc else k
    full_v = torch.cat([ctx_v, v], 0) if Lc else v
    kpool, npages, total = _pool(full_k, Lc, hidden.device)
    vpool, _, _ = _pool(full_v, Lc, hidden.device)

    dev = hidden.device
    page_table = torch.arange(npages, dtype=torch.int32, device=dev)
    mask = [P] * (total // P) + ([total % P] if total % P else [])
    mask_table = torch.tensor(mask, dtype=torch.uint8, device=dev)
    cu_pages = torch.tensor([0, npages], dtype=torch.int32, device=dev)
    page_pos = torch.tensor(
        [i * P for i in range(npages)], dtype=torch.int32, device=dev)
    cu_q = torch.tensor([0, Lq], dtype=torch.int32, device=dev)
    qpos = torch.tensor([Lc], dtype=torch.int32, device=dev)
    scale = getattr(attn, "scaling", 1.0 / math.sqrt(D))

    o = flash_prefill_func(
        q.contiguous(), kpool, vpool, page_table, mask_table, cu_pages,
        cu_q, qpos, page_pos, cos, sin, num_kv_heads, 1, scale)
    o = o.reshape(*shape, -1) * torch.sigmoid(gate)
    return attn.o_proj(o), (k, v)


def gdn_forward(la, hidden, conv_init, rec_init):


    B, L, _ = hidden.shape
    K = la.conv_kernel_size
    mixed = la.in_proj_qkv(hidden).transpose(1, 2)
    C = mixed.shape[1]
    if conv_init is None:
        conv_init = mixed.new_zeros(C, K - 1)
    combined = torch.cat([conv_init[None], mixed], dim=-1)
    conv_out = F.conv1d(
        combined, la.conv1d.weight, la.conv1d.bias, groups=la.conv1d.groups)
    conv_out = F.silu(conv_out[:, :, -L:]).transpose(1, 2)
    conv_final = combined[0, :, -(K - 1):]

    query, key, value = torch.split(
        conv_out, [la.key_dim, la.key_dim, la.value_dim], dim=-1)
    query = query.reshape(B, L, -1, la.head_k_dim)
    key = key.reshape(B, L, -1, la.head_k_dim)
    value = value.reshape(B, L, -1, la.head_v_dim)
    repeat = la.num_v_heads // la.num_k_heads
    if repeat > 1:
        query = query.repeat_interleave(repeat, dim=2)
        key = key.repeat_interleave(repeat, dim=2)

    z = la.in_proj_z(hidden).reshape(B, L, -1, la.head_v_dim)
    beta = la.in_proj_b(hidden).sigmoid()
    g = -la.A_log.float().exp() * F.softplus(
        la.in_proj_a(hidden).float() + la.dt_bias)
    core, rec_final = la.chunk_gated_delta_rule(
        query, key, value, g=g, beta=beta,
        initial_state=rec_init, output_final_state=True,
        use_qk_l2norm_in_kernel=True)
    core = la.norm(core.reshape(-1, la.head_v_dim), z.reshape(-1, la.head_v_dim))
    return la.out_proj(core.reshape(B, L, -1)), conv_final, rec_final


def _layer(layer, h, c0, c1, cos, sin, nkv):
    residual = h
    hn = layer.input_layernorm(h)
    if layer.layer_type == "linear_attention":
        out, s0, s1 = gdn_forward(layer.linear_attn, hn, c0, c1)
    else:
        out, (s0, s1) = paged_attention(layer.self_attn, hn, c0, c1, cos, sin, nkv)
    h = residual + out
    h = h + layer.mlp(layer.post_attention_layernorm(h))
    return h, s0, s1


def body_forward(body, tokens, ctx_kv, gdn_init, cos, sin, nkv):
    h = body.embed_tokens(tokens)
    kv_out, gdn_out, a, g = [], [], 0, 0
    for layer in body.layers:
        linear = layer.layer_type == "linear_attention"
        init, idx = (gdn_init, g) if linear else (ctx_kv, a)
        c0, c1 = init[idx] if init and init[idx] else (None, None)
        step = lambda h, c0, c1, layer=layer: _layer(layer, h, c0, c1, cos, sin, nkv)
        if torch.is_grad_enabled():
            h, s0, s1 = checkpoint(step, h, c0, c1, use_reentrant=False)
        else:
            h, s0, s1 = step(h, c0, c1)
        if linear:
            gdn_out.append((s0, s1)); g += 1
        else:
            kv_out.append((s0, s1)); a += 1
    return body.norm(h)[0], kv_out, gdn_out
