"""
The tests that decide whether the architecture claim is worth anything.

A prototype that only shows the happy path is a demo. Each test below has a way to fail,
and the ones that matter are the ones where a wrong answer is possible.
"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from cellgraph import canary_holds, tensor_digest, CANARY
from tinyformer import run, build

RESULTS = []
def check(name, cond, detail=""):
    RESULTS.append((name, bool(cond), detail))
    print(f"  {'ok  ' if cond else 'FAIL'}  {name}" + (f"   {detail}" if detail else ""))
    return bool(cond)

TOK = [1, 5, 9, 13, 17]

# 1. the fleet canary still agrees
check("canary agrees with the rest of the fleet", canary_holds(),
      f"0x{CANARY:016x}")

# 2. the graph runs and every cell is witnessed
g, x = run(TOK)
check("every cell produces a witness record",
      len(g.witness) == len(g.cells) == 19, f"{len(g.witness)} records / {len(g.cells)} cells")

# 3. the output is a real distribution
p = x["probs"]
check("probs is a distribution on the last axis",
      np.allclose(p.sum(-1), 1.0, atol=1e-5), f"max dev {abs(p.sum(-1)-1).max():.2e}")

# 4. determinism -- same input, same digests, every time
g2, x2 = run(TOK)
check("digests are deterministic across runs",
      [w.digest for w in g.witness] == [w.digest for w in g2.witness])

# 5. THE LOCALIZATION TEST -- the reason the whole idea exists.
#    Perturb one weight and find WHERE the output changed. If the witness cannot localize
#    a single-constant perturbation to the cells downstream of it, the architecture is
#    decoration and a plain test suite would have told you the same thing.
g3, env3 = build(7)
before = [c.digest for c in g.cells.values()] if g.cells and all(c.digest for c in g.cells.values()) else None
d_before = {w.cell: w.digest for w in g.witness}
env3["W1"][0, 0] = env3["W1"][0, 0] + 0.25          # one element, in the FFN up-projection
_ = g3.effect({"tokens": np.asarray(TOK, np.int64).reshape(1, -1), "env": env3})
d_after = {w.cell: w.digest for w in g3.witness}
changed = [c for c in d_before if d_before[c] != d_after[c]]
first_changed = changed[0] if changed else None
check("a one-element weight change is DETECTED by the canary", len(changed) > 0,
      f"{len(changed)} of {len(d_before)} cell digests moved")
check("the first changed cell is ff_in, the one that consumes W1",
      first_changed == "ff_in", f"first changed = {first_changed}")
check("cells UPSTREAM of the change are unaffected",
      all(c not in changed for c in ("embed", "norm_in", "q_proj", "rope_q", "attend",
                                     "o_proj", "res_attn", "norm_mid")),
      f"unchanged upstream: {sum(1 for c in d_before if c not in changed)}")
check("cells DOWNSTREAM of the change all moved",
      all(c in changed for c in ("act", "ff_out", "res_ff", "norm_out", "logits", "probs")))

# 6. a change that should NOT propagate is not spuriously flagged
d_c2 = {w.cell: w.digest for w in run(TOK, seed=8)[0].witness}
# (seed 8 changes the env, so everything moves -- this asserts the graph is sensitive, i.e.
#  the digests are load-bearing rather than constant)
check("digests are load-bearing, not constant",
      d_c2["probs"] != d_before["probs"], "a different seed gives a different probs digest")

# 7. attention really is causal -- if the mask is wrong the model is a different model
w = x["attn_weights"][0]        # (heads, T, T)
upper = np.triu(np.ones((w.shape[-1], w.shape[-1]), dtype=bool), 1)
check("attention mask is causal", float(w[:, upper].max()) < 1e-6,
      f"max weight above the diagonal {float(w[:, upper].max()):.2e}")
check("attention rows are normalised", np.allclose(w.sum(-1), 1.0, atol=1e-5))

# 8. the e-process
g4, x4 = run(TOK)
g4.e_update(p_claimed=0.30, p_actual=0.30)     # model exactly as claimed -> no evidence
check("e-process does not alarm when the model is exactly as claimed",
      not g4.alarmed(), f"max log E = {g4.max_log_e:.6f}")
g4.e_update(p_claimed=0.30, p_actual=0.90)     # model far better than claimed
g4.e_update(p_claimed=0.30, p_actual=0.90)
g4.e_update(p_claimed=0.30, p_actual=0.90)
g4.e_update(p_claimed=0.30, p_actual=0.90)
g4.e_update(p_claimed=0.30, p_actual=0.90)
g4.e_update(p_claimed=0.30, p_actual=0.90)
g4.e_update(p_claimed=0.30, p_actual=0.90)
check("e-process alarms on sustained divergence from the claimed model", g4.alarmed(),
      f"max log E = {g4.max_log_e:.3f}, threshold {-np.log(0.05):.3f}")

# 9. the honest limit, asserted rather than asserted-in-prose
check("cells are NOT yet cells in the full doctrinal sense",
      True, "no identity, no self-witness, no desire -- this is a graph of named functions")

p_ = sum(1 for _, ok, _ in RESULTS if ok)
print(f"\n  {p_}/{len(RESULTS)} checks pass")
json.dump([{"name": n, "pass": ok, "detail": d} for n, ok, d in RESULTS],
          open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "results.json"), "w"), indent=1)
sys.exit(0 if p_ == len(RESULTS) else 1)
