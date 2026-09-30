"""
tinyformer.py — a transformer forward pass as a Quilt cell graph.

The claim: a transformer block's dataflow and a Quilt cell graph are the same shape.

That claim is only true if the cells can be wired WITHOUT special-casing. The first draft
of this file needed four cells rewired after construction to reach the constants -- which is
exactly the kind of thing that would mean the claim was false. So the signature is uniform
instead:

    every cell is  f(env, *cell_inputs) -> array
    every cell declares its inputs BY NAME
    the graph is a topological order of that declaration

If that is still awkward, the claim is wrong and this file says so.
"""
from __future__ import annotations
import numpy as np
from cellgraph import CellGraph

def rms_norm(x, g):
    return x * (1.0 / np.sqrt((x * x).mean(-1, keepdims=True) + 1e-6)) * g

def softmax(x, axis=-1):
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)

def silu(x):
    return x / (1.0 + np.exp(-x))

def rope(x, T, D, base=10000.0):
    """rotary position embedding, half-rotation convention (the llama3-from-scratch form:
    split the head dim in half and rotate the halves against each other)."""
    half = D // 2
    pos = np.arange(T, dtype=np.float32)[:, None]
    inv = 1.0 / (base ** (np.arange(half, dtype=np.float32) / half))[None, :]
    ang = pos * inv
    cos, sin = np.cos(ang).astype(np.float32), np.sin(ang).astype(np.float32)
    x1, x2 = x[..., :half], x[..., half:]
    return np.concatenate([x1 * cos - x2 * sin, x1 * sin + x2 * cos], axis=-1)

def attn(q, k, v, nh, causal=True, want_w=False):
    B, T, D = q.shape
    hd = D // nh
    qh = q.reshape(B, T, nh, hd).transpose(0, 2, 1, 3)
    kh = k.reshape(B, T, nh, hd).transpose(0, 2, 1, 3)
    vh = v.reshape(B, T, nh, hd).transpose(0, 2, 1, 3)
    s = qh @ kh.transpose(0, 1, 3, 2) / np.sqrt(hd)
    if causal:
        s = np.where(np.triu(np.ones((T, T), dtype=bool), 1), -1e9, s)
    w = softmax(s, -1)
    o = (w @ vh).transpose(0, 2, 1, 3).reshape(B, T, D)
    return w if want_w else o

D_MODEL, N_HEAD, VOCAB = 32, 4, 97

def build(seed=7):
    r = np.random.default_rng(seed)
    d, h, V = D_MODEL, N_HEAD, VOCAB
    env = {
        "E":    r.normal(0, .05, (V, d)).astype(np.float32),
        "gin":  np.ones(d, np.float32), "gmid": np.ones(d, np.float32), "gout": np.ones(d, np.float32),
        "Wq":   r.normal(0, .1, (d, d)).astype(np.float32),
        "Wk":   r.normal(0, .1, (d, d)).astype(np.float32),
        "Wv":   r.normal(0, .1, (d, d)).astype(np.float32),
        "Wo":   r.normal(0, .1, (d, d)).astype(np.float32),
        "W1":   r.normal(0, .1, (d, 4 * d)).astype(np.float32),
        "W2":   r.normal(0, .1, (4 * d, d)).astype(np.float32),
        "Wlog": r.normal(0, .1, (d, V)).astype(np.float32),
        "nh":   np.int64(h),
    }
    g = CellGraph()
    B = lambda *a, **k: g.bind(*a, **k)
    B("embed",       "EMBED",   lambda e, t: e["E"][t],                 ["tokens"])
    B("norm_in",     "NORM",    lambda e, x: rms_norm(x, e["gin"]),    ["embed"])
    B("q_proj",      "PROJ",    lambda e, x: x @ e["Wq"],              ["norm_in"])
    B("k_proj",      "PROJ",    lambda e, x: x @ e["Wk"],              ["norm_in"])
    B("v_proj",      "PROJ",    lambda e, x: x @ e["Wv"],              ["norm_in"])
    B("rope_q",      "POSENC",  lambda e, q: rope(q, q.shape[1], e["Wq"].shape[0]), ["q_proj"])
    B("rope_k",      "POSENC",  lambda e, k: rope(k, k.shape[1], e["Wq"].shape[0]), ["k_proj"])
    B("attend",      "ATTN",    lambda e, q, k, v: attn(q, k, v, int(e["nh"]), want_w=False),
                  ["rope_q", "rope_k", "v_proj"])
    B("attn_weights","ATTNW",   lambda e, q, k, v: attn(q, k, v, int(e["nh"]), want_w=True),
                  ["rope_q", "rope_k", "v_proj"])
    B("o_proj",      "PROJ",    lambda e, x: x @ e["Wo"],              ["attend"])
    B("res_attn",    "ADD",     lambda e, a, b: a + b,                ["embed", "o_proj"])
    B("norm_mid",    "NORM",    lambda e, x: rms_norm(x, e["gmid"]),  ["res_attn"])
    B("ff_in",       "PROJ",    lambda e, x: x @ e["W1"],              ["norm_mid"])
    B("act",         "ACT",     lambda e, x: silu(x),                 ["ff_in"])
    B("ff_out",      "PROJ",    lambda e, x: x @ e["W2"],              ["act"])
    B("res_ff",      "ADD",     lambda e, a, b: a + b,                ["res_attn", "ff_out"])
    B("norm_out",    "NORM",    lambda e, x: rms_norm(x, e["gout"]),  ["res_ff"])
    B("logits",      "HEAD",    lambda e, x: x @ e["Wlog"],            ["norm_out"])
    B("probs",       "DISTRIB", lambda e, x: softmax(x, -1),         ["logits"])
    return g, env

def run(tokens, seed=7):
    g, env = build(seed)
    # batch axis is explicit: tokens are (B, T), embed yields (B, T, d)
    x = {"tokens": np.asarray(tokens, dtype=np.int64).reshape(1, -1), "env": env}
    return g, g.effect(x)
