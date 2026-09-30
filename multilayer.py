"""
multilayer.py — N transformer blocks as one cell graph, and the tool the localization
property enables.

The one-block prototype proved the witness can point at the consumer of a perturbed weight.
That is only interesting if it survives two things the toy did not have:

  1. SCALE. Real models are 32-80 blocks. Does localization still hold, and does the graph
     stay uniform, or does the twentieth block need something the first did not?
  2. A QUERY. The one-block test knew which weight it perturbed. The useful version does
     not. It is handed a SYMPTOM -- an output that is wrong -- and has to find the cell.

layer_report()  -> the witness chain, per layer
find_fault()    -> given a reference and a suspect, name the first cell that disagrees
"""
from __future__ import annotations
import numpy as np
from cellgraph import CellGraph, tensor_digest, fnv1a64

D_MODEL, N_HEAD = 32, 4

def rms_norm(x, g):
    return x * (1.0 / np.sqrt((x*x).mean(-1, keepdims=True) + 1e-6)) * g

def softmax(x, axis=-1):
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x); return e / e.sum(axis=axis, keepdims=True)

def silu(x): return x / (1.0 + np.exp(-x))

def rope(x, T, D, base=10000.0):
    half = D // 2
    pos = np.arange(T, dtype=np.float32)[:, None]
    inv = 1.0 / (base ** (np.arange(half, dtype=np.float32) / half))[None, :]
    ang = pos * inv
    cos, sin = np.cos(ang).astype(np.float32), np.sin(ang).astype(np.float32)
    x1, x2 = x[..., :half], x[..., half:]
    return np.concatenate([x1*cos - x2*sin, x1*sin + x2*cos], axis=-1)

def attn(q, k, v, nh, causal=True):
    B, T, D = q.shape; hd = D // nh
    qh = q.reshape(B,T,nh,hd).transpose(0,2,1,3)
    kh = k.reshape(B,T,nh,hd).transpose(0,2,1,3)
    vh = v.reshape(B,T,nh,hd).transpose(0,2,1,3)
    s = qh @ kh.transpose(0,1,3,2) / np.sqrt(hd)
    if causal: s = np.where(np.triu(np.ones((T,T), dtype=bool), 1), -1e9, s)
    return (softmax(s,-1) @ vh).transpose(0,2,1,3).reshape(B,T,D)

def make_env(n_layer, vocab, seed):
    r = np.random.default_rng(seed)
    d = D_MODEL
    env = {"E": r.normal(0,.05,(vocab,d)).astype(np.float32), "nh": np.int64(N_HEAD)}
    for i in range(n_layer):
        # TWO separate gain vectors per block, as llama does: norm.weight and
        # post_attention_norm.weight. Sharing one made the two norm cells
        # indistinguishable, and the fault finder correctly blamed the earlier one.
        env[f"g{i}a"] = np.ones(d, np.float32)
        env[f"g{i}f"] = np.ones(d, np.float32)
        for nm, (a,b) in {"Wq":(d,d),"Wk":(d,d),"Wv":(d,d),"Wo":(d,d),
                          "W1":(d,4*d),"W2":(4*d,d)}.items():
            env[f"{nm}{i}"] = r.normal(0,.1,(a,b)).astype(np.float32)
    env["gout"] = np.ones(d, np.float32)
    env["Wlog"] = r.normal(0,.1,(d,vocab)).astype(np.float32)
    return env

def build(n_layer=4, vocab=97):
    """Uniform cell graph, N blocks, insertion order == topological order."""
    g = CellGraph(); d = D_MODEL
    B = lambda *a, **k: g.bind(*a, **k)
    B("embed", "EMBED", lambda e, t: e["E"][t], ["tokens"])
    src = "embed"
    for i in range(n_layer):
        p = f"L{i}."
        B(p+"norm_in",   "NORM",  lambda e,x,i=i: rms_norm(x, e[f"g{i}a"]),      [src])
        B(p+"q_proj",   "PROJ",  lambda e,x,i=i: x @ e[f"Wq{i}"],               [p+"norm_in"])
        B(p+"k_proj",   "PROJ",  lambda e,x,i=i: x @ e[f"Wk{i}"],               [p+"norm_in"])
        B(p+"v_proj",   "PROJ",  lambda e,x,i=i: x @ e[f"Wv{i}"],               [p+"norm_in"])
        B(p+"rope_q",   "POSENC",lambda e,q: rope(q, q.shape[1], D_MODEL),       [p+"q_proj"])
        B(p+"rope_k",   "POSENC",lambda e,k: rope(k, k.shape[1], D_MODEL),       [p+"k_proj"])
        B(p+"attend",   "ATTN",  lambda e,q,k,v: attn(q,k,v,int(e["nh"])),       [p+"rope_q",p+"rope_k",p+"v_proj"])
        B(p+"o_proj",   "PROJ",  lambda e,x,i=i: x @ e[f"Wo{i}"],               [p+"attend"])
        B(p+"res_attn", "ADD",   lambda e,a,b: a + b,                            [src, p+"o_proj"])
        B(p+"norm_mid", "NORM",  lambda e,x,i=i: rms_norm(x, e[f"g{i}f"]),      [p+"res_attn"])
        B(p+"ff_in",    "PROJ",  lambda e,x,i=i: x @ e[f"W1{i}"],               [p+"norm_mid"])
        B(p+"act",      "ACT",   lambda e,x: silu(x),                            [p+"ff_in"])
        B(p+"ff_out",   "PROJ",  lambda e,x,i=i: x @ e[f"W2{i}"],               [p+"act"])
        B(p+"res_ff",   "ADD",   lambda e,a,b: a + b,                            [p+"res_attn", p+"ff_out"])
        src = p + "res_ff"
    B("norm_out", "NORM",  lambda e,x: rms_norm(x, e["gout"]),  [src])
    B("logits",   "HEAD",  lambda e,x: x @ e["Wlog"],          ["norm_out"])
    B("probs",    "DISTRIB",lambda e,x: softmax(x, -1),       ["logits"])
    return g

def run(tokens, n_layer=4, vocab=97, seed=7, env=None):
    g = build(n_layer, vocab)
    e = env if env is not None else make_env(n_layer, vocab, seed)
    x = {"tokens": np.asarray(tokens, np.int64).reshape(1,-1), "env": e}
    return g, g.effect(x)

# ── the tool the property enables ───────────────────────────────────────────────
def find_fault(ref_digests, got_digests, order):
    """Given a REFERENCE run's digests and a SUSPECT run's, name the first cell that
    disagrees. No knowledge of what was changed is passed in. The answer falls out of
    the witness chain."""
    for c in order:
        if ref_digests.get(c) != got_digests.get(c):
            return c
    return None

def chain(g):
    return {w.cell: w.digest for w in g.witness}

def order_of(g):
    return [w.cell for w in g.witness]
