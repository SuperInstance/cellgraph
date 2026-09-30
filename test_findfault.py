"""
The one-block prototype knew which weight it perturbed. This does not.

find_fault() is handed a REFERENCE run's witness chain and a SUSPECT run's, and has to name
the first cell that disagrees. Nothing else is passed in. That is the difference between
"the digest moved" and a debugging tool.

Two classes of fault, because they fail differently:
  A. a WEIGHT is perturbed      -> should be localised to the cell that CONSUMES it
  B. a CELL IMPLEMENTATION is wrong -> should be localised to that CELL ITSELF

Class A is the one-block result generalised. Class B is new and is the harder test: a cell
that computes the wrong thing while consuming the right inputs.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from multilayer import run, make_env, build, chain, order_of, find_fault

TOK = [1, 5, 9, 13, 17]
N_LAYER = 4
R = []
def check(name, cond, detail=""):
    R.append((name, bool(cond), detail))
    print(f"  {'ok  ' if cond else 'FAIL'}  {name}" + (f"   {detail}" if detail else ""))
    return bool(cond)

# reference
gref, _ = run(TOK, n_layer=N_LAYER)
REF = chain(gref); ORDER = order_of(gref)
check("multi-layer graph runs and witnesses every cell",
      len(gref.witness) == len(gref.cells) == 60, f"{len(gref.cells)} cells")

# ── CLASS A: a weight is perturbed, nothing is said about it ─────────────────────
print("\n  class A — a weight is perturbed. Does the tool find the consumer?")
for layer, wname, expect in [(0,"Wq","L0.q_proj"), (2,"W1","L2.ff_in"),
                             (3,"W2","L3.ff_out"), (1,"Wo","L1.o_proj")]:
    env = make_env(N_LAYER, 97, 7)
    key = f"{wname}{layer}"
    env[key] = env[key].copy(); env[key].flat[0] += 0.37
    _, _ = run(TOK, n_layer=N_LAYER, env=env)
    got = chain(run(TOK, n_layer=N_LAYER, env=env)[0])
    found = find_fault(REF, got, ORDER)
    check(f"  {key:8} -> {expect:12}", found == expect, f"found {found}")

# ── CLASS B: a cell computes the wrong thing from the right inputs ───────────────
print("\n  class B — a CELL's implementation is wrong. Does the tool find the cell?")
FAULTS = [
    # (cell id, what we break, name of the cell we expect to be blamed)
    ("L1.act",        "silu -> relu",              "L1.act"),
    ("L2.norm_mid",   "gains set to 2.0",          "L2.norm_mid"),
    ("L0.o_proj",     "weight scaled by 1.5",      "L0.o_proj"),
    ("L3.res_attn",   "residual halved",           "L3.res_attn"),
    ("L1.attend",     "causal mask removed",       "L1.attend"),
]
for cid, what, expect in FAULTS:
    g = build(N_LAYER, 97)
    env = make_env(N_LAYER, 97, 7)
    c = g.cells[cid]
    orig = c.fn
    # layer number from the cell id: "L2.norm_mid" -> "2". The earlier cid[1:3] gave "2."
    # and wrote to a key nothing read, so the fault was invisible and the test reported
    # "no fault found" -- a broken instrument reporting a clean result.
    lay = cid.split(".")[0][1:]
    if what == "gains set to 2.0":
        env[f"g{lay}f"] = np.full(32, 2.0, np.float32)
    elif what == "weight scaled by 1.5":
        env[f"Wo{lay}"] = env[f"Wo{lay}"] * 1.5
    elif what == "residual halved":
        c.fn = lambda e, a, b: a + 0.5 * b
    elif what == "causal mask removed":
        import multilayer as M
        c.fn = lambda e, q, k, v: M.attn(q, k, v, int(e["nh"]), causal=False)
    elif what == "silu -> relu":
        c.fn = lambda e, x: np.maximum(x, 0.0)
    g.effect({"tokens": np.asarray(TOK, np.int64).reshape(1,-1), "env": env})
    found = find_fault(REF, chain(g), ORDER)
    check(f"  {cid:14} {what:24}", found == expect, f"found {found}")

# ── the negative control: two identical runs must report NO fault ────────────────
print("\n  control — an UNCHANGED model must report no fault at all")
_, _ = run(TOK, n_layer=N_LAYER)
same = chain(run(TOK, n_layer=N_LAYER)[0])
check("identical runs report no fault", find_fault(REF, same, ORDER) is None,
      f"reported {find_fault(REF, same, ORDER)}")

# ── the resolution limit, asserted rather than wished for ───────────────────────
print("\n  resolution — how small a change can the tool see? (it used to be blind below f32)")
env = make_env(N_LAYER, 97, 7)
env["W23"] = env["W23"].copy()
env["W23"][7, 11] = np.float32(np.nextafter(env["W23"][7, 11], np.float32(1e9)))
_, base_x = run(TOK, n_layer=N_LAYER)
_, tiny_x = run(TOK, n_layer=N_LAYER, env=env)
arrays_identical = np.array_equal(base_x["L3.ff_out"], tiny_x["L3.ff_out"])
no_fault = find_fault(REF, chain(run(TOK, n_layer=N_LAYER, env=env)[0]), ORDER) is None
check("a 1-ULP weight nudge IS caught, because the digest uses the native f64 precision",
      (not arrays_identical) and not no_fault,
      "this is the case that was invisible before the dtype fix -- it moved 5 activations by 9.8e-10")

# and a change well above the resolution is caught too
env2 = make_env(N_LAYER, 97, 7)
env2["W23"] = env2["W23"].copy(); env2["W23"][7, 11] = np.float32(env2["W23"][7, 11] + 1e-3)
found2 = find_fault(REF, chain(run(TOK, n_layer=N_LAYER, env=env2)[0]), ORDER)
check("a change above float32 resolution IS caught and localised",
      found2 == "L3.ff_out", f"found {found2}")

p = sum(1 for _, ok, _ in R if ok)
print(f"\n  {p}/{len(R)} checks pass")
sys.exit(0 if p == len(R) else 1)
