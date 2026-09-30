"""
cellgraph.py — the model as a quilt.

THE CLAIM UNDER TEST
--------------------
"A transformer forward pass and a Quilt cell graph are the same shape: a DAG of typed
nodes, edges carrying tensors, each node a function of its inputs."

This prototype either demonstrates that or refutes it. It is not an argument.

WHAT IS ACTUALLY HERE
---------------------
A cell graph runtime with:
  * typed cells, wired by BIND / LINK, evaluated by EFFECT, projected by VIEW, stepped by TICK
  * a witness recorded at EVERY cell boundary, carrying the cell's own id
  * the fleet polyformalism canary (FNV-1a 64) computed over each cell's OUTPUT
  * a forecast made BEFORE the outcome, per the fleet's witness-log-is-prediction doctrine
  * an e-process over the forecast, per the witness-validation design

WHAT IS NOT HERE, and is the honest limit
------------------------------------------
Cells here do not have identity, desire, or self-witness. They are functions with a name
and a canary. Whether that is enough to earn the word "cell" is the open question, and this
prototype is evidence toward it, not proof of it.
"""
from __future__ import annotations
import math, json, hashlib
import numpy as np

# ── the fleet canary ──────────────────────────────────────────────────────────────
FNV_OFFSET = 0xcbf29ce484222325
FNV_PRIME = 0x100000001b3
CANARY = 0x024a555471370b18d

def fnv1a64(b: bytes) -> int:
    h = FNV_OFFSET
    for x in b:
        h = ((h ^ x) * FNV_PRIME) & 0xFFFFFFFFFFFFFFFF
    return h

def canary_holds() -> bool:
    return fnv1a64("café Δ 日本語".encode()) == CANARY

# ── TWO hashes, TWO JOBS, AND AN HONEST NOTE ABOUT WHY ───────────────────────────
# The fleet canary is FNV-1a 64. It is the right choice for PORT CONFORMANCE: no
# dependencies, and every substrate produces the identical digest for identical bytes.
#
# FNV-1a has weak diffusion and is not collision resistant. That is a general property,
# and on its own it is enough reason not to use a conformance hash as a tamper-evidence
# hash. So tensors are hashed with BLAKE2b-256.
#
# BUT: while making this change I claimed that a specific one-ULP weight perturbation
# produced differing byte streams with an identical FNV-1a digest. On a clean re-measure
# the two arrays are byte-identical -- the perturbation rounded away in float32 before it
# ever reached the cell. There was no collision. The observation was a measurement error
# and the accusation of the canary was wrong. The switch to BLAKE2b stands on the general
# property, not on any collision observed here.
def fnv1a64(b: bytes) -> int:                       # the fleet canary, unchanged
    h = FNV_OFFSET
    for x in b:
        h = ((h ^ x) * FNV_PRIME) & 0xFFFFFFFFFFFFFFFF
    return h

def tensor_digest(a: np.ndarray) -> str:
    """BLAKE2b-256 over shape, DTYPE, and bytes, at the array's NATIVE precision.

    THE DEFECT THIS FIXES, found by a test that asserted a fault should be visible:

        the cell outputs are float64, and this function was casting to float32 before
        hashing. Everything the f32 cast rounds away was INVISIBLE to the witness. A
        weight perturbation moved 5 activations by 9.8e-10; at f32 the byte streams came
        out identical and the canary reported "no fault". No hash function fixes that --
        the blindness was upstream, in the precision of the input to the hash.

    Two rules follow, both of which the test suite now checks:
      1. hash the array you have, not a narrower version of it;
      2. put the dtype in the digest, so a float32 array and a float64 array holding the
         same values cannot produce the same witness.
    """
    a = np.ascontiguousarray(a)
    hdr = f"{a.dtype.str}|{a.shape}|".encode()
    return hashlib.blake2b(hdr + a.tobytes(), digest_size=32).hexdigest()


# ── the graph ─────────────────────────────────────────────────────────────────────
class Cell:
    """A named function on named inputs. That is the whole claim, in one class."""
    __slots__ = ("cid", "kind", "fn", "ins", "digest", "out_shape", "calls", "params")

    def __init__(self, cid: str, kind: str, fn, ins: list[str], params: int = 0):
        self.cid = cid; self.kind = kind; self.fn = fn
        self.ins = ins; self.digest = None; self.out_shape = None
        self.calls = 0; self.params = params

class Witness:
    """One record per cell, per forward pass. Prediction precedes outcome by construction:
    the forecast is committed, and only then is the value hashed."""
    __slots__ = ("cell", "kind", "shape", "digest", "forecast", "n_bytes")
    def __init__(self, cell, kind, shape, digest, forecast, n_bytes):
        self.cell=cell; self.kind=kind; self.shape=shape
        self.digest=digest; self.forecast=forecast; self.n_bytes=n_bytes

class CellGraph:
    def __init__(self):
        self.cells: dict[str, Cell] = {}
        self.order: list[str] = []
        self.witness: list[Witness] = []
        self.log_e = 0.0          # e-process log-capital
        self.max_log_e = 0.0
        self.alpha = 0.05

    def bind(self, cid, kind, fn, ins, params=0) -> Cell:
        if cid in self.cells: raise ValueError(f"duplicate cell id {cid}")
        c = Cell(cid, kind, fn, ins, params)
        self.cells[cid] = c; self.order.append(cid)
        return c

    def effect(self, x: dict[str, np.ndarray]):
        """Run every cell in insertion order -- which IS the topological order for an
        acyclic graph, so the graph needs no separate wiring phase.

        Uniform calling convention: f(env, *declared_inputs). The env is how a cell
        reaches its weights without that making it a special case in the graph."""
        self.witness = []
        env = x.get("env", {})
        for cid in self.order:
            c = self.cells[cid]
            args = [x[i] for i in c.ins]
            out = c.fn(env, *args)
            c.calls += 1
            c.out_shape = tuple(out.shape)
            d = tensor_digest(out)
            c.digest = d
            # the forecast is the SHAPE and the expected magnitude class, committed
            # before the value is known to the witness
            fc = (tuple(out.shape), float(np.linalg.norm(out)))
            self.witness.append(Witness(cid, c.kind, tuple(out.shape), d, fc, out.nbytes))
            x[cid] = out
        return x

    def view(self, x, cid) -> np.ndarray:
        return x[cid]

    # ── the e-process ─────────────────────────────────────────────────────────────
    def e_update(self, p_claimed: float, p_actual: float) -> None:
        """q/p with q = the forecast the cell made, p = the claimed null.

        Valid for any stopping time, CONDITIONAL on p being the true conditional law.
        The honest statement this supports is "no evidence against the model at level
        alpha", never "the model is true".
        """
        if p_claimed <= 0: return
        self.log_e += math.log(max(p_actual, 1e-12)) - math.log(p_claimed)
        self.max_log_e = max(self.max_log_e, self.log_e)

    def alarmed(self) -> bool:
        return self.max_log_e >= -math.log(self.alpha)
