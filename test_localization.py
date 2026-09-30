"""
The negative control the first test needs.

Perturbing W1 and seeing ff_in change could be an ARTEFACT: ff_in is simply the first
cell in insertion order that sits after the change. If that is all that is happening, then
the witness does not localise anything and the architecture claim is decoration.

The real test: perturb a DIFFERENT weight and the first changed cell must be a DIFFERENT
cell, and it must be the one that consumes that weight. If the first changed cell is always
the same, or always "the next one after the perturbation point in file order", the
localisation is fake.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from cellgraph import CellGraph
from tinyformer import run, build

TOK = [1, 5, 9, 13, 17]

# which cell consumes which weight
CONSUMER = {
    "Wq": "q_proj", "Wk": "k_proj", "Wv": "v_proj",
    "W1": "ff_in",  "W2": "ff_out", "Wlog": "logits",
}

def digests(env):
    g, _ = build(7)
    g.effect({"tokens": np.asarray(TOK, np.int64).reshape(1, -1), "env": env})
    return {w.cell: w.digest for w in g.witness}

base_env = build(7)[1]
base = digests(base_env)
base_order = [w.cell for w in run(TOK)[0].witness]

print("  perturbing one weight at a time; asking WHICH cell notices FIRST\n")
print(f"  {'weight':6} {'expected consumer':16} {'first cell that moved':22} {'moved':>6}  verdict")
ok_all = True
for wname, expect in CONSUMER.items():
    env = build(7)[1]
    env[wname] = env[wname].copy()
    env[wname].flat[0] += 0.5                      # one element, different position each time
    after = digests(env)
    moved = [c for c in base_order if base[c] != after[c]]
    first = moved[0] if moved else None
    ok = (first == expect)
    ok_all &= ok
    print(f"  {wname:6} {expect:16} {str(first):22} {len(moved):>6}  {'ok' if ok else 'MISMATCH'}")

print()
# the second half of the control: the first-changed cell must depend on the PERTURBATION,
# not on position in the cell list. If it were positional, Wlog (the last weight, consumed
# near the end) would report an early cell.
idx_expect = {c: base_order.index(c) for c in CONSUMER.values()}
env = build(7)[1]; env["Wlog"] = env["Wlog"].copy(); env["Wlog"].flat[0] += 0.5
m = [c for c in base_order if base[c] != digests(env)[c]]
print(f"  Wlog is consumed at position {idx_expect['logits']} of {len(base_order)-1};")
print(f"  perturbing it reports first-changed = {m[0] if m else None} at position "
      f"{base_order.index(m[0]) if m else 'n/a'}")
print()
print("  VERDICT:", "localisation is real -- the witness points at the consumer of the "
      "weight that moved, not at a fixed position" if ok_all else
      "LOCALISATION IS AN ARTEFACT -- the first-changed cell does not track the perturbation")
sys.exit(0 if ok_all else 1)
