# cellgraph — the model as a quilt

A prototype for one claim:

> **A transformer forward pass and a Quilt cell graph are the same shape: a DAG of typed
> nodes, edges carrying tensors, each node a function of its inputs.**

This is not an argument for that claim. It is a test of it, with controls that could have
failed.

## The result

**The claim holds, and the localization is real.**

`test_localization.py` perturbs one element of one weight and asks which cell notices
first. The first cell to move is the consumer of the perturbed weight, every time — and the
number of cells that move tracks the dependency structure rather than position in the file:

| weight perturbed | first cell to move | cells whose digest moved |
|---|---|---:|
| `Wq`  | `q_proj`  | 14 |
| `Wk`  | `k_proj`  | 14 |
| `Wv`  | `v_proj`  | 12 |
| `W1`  | `ff_in`   | 7 |
| `W2`  | `ff_out`  | 5 |
| `Wlog`| `logits`  | 2 |

`Wq` feeds the attention block and everything after it, so 14 cells move. `Wlog` is the
final projection to vocabulary, so exactly 2 move. **The witness log recovers the
dependency graph of the forward pass from digests alone.**

That is the thing the architecture is for. A test suite tells you a test failed. This
tells you *which cell in the forward pass* failed and how far the damage travelled.

## The design that made it uniform

The first draft needed four cells rewired after construction to reach their weights. That
is exactly the kind of thing that would have meant the claim was false. So the calling
convention is uniform instead:

```
every cell is   f(env, *declared_inputs) -> array
every cell declares its inputs BY NAME
insertion order IS the topological order — there is no separate wiring phase
```

If that had stayed awkward, the claim would have been wrong and this file would say so.

## What runs

```
python3 test_cellgraph.py       # 14 checks
python3 test_localization.py    # the control the first one needs
```

`14/14` and the control passes. Cells, in order: `embed, norm_in, q_proj, k_proj, v_proj,
rope_q, rope_k, attend, attn_weights, o_proj, res_attn, norm_mid, ff_in, act, ff_out,
res_ff, norm_out, logits, probs`. Nineteen cells, nineteen witness records, one digest each.

## The canary, moved down to the tensor

The fleet polyformalism canary is FNV-1a 64 over `"café Δ 日本語"` → `0x024a555471370b18d`,
verified in Python, TypeScript, Rust, C#, and Julia. That guards *source code*.

Here the same digest is taken over each cell's **output tensor** — shape plus bytes. This
is the claim the whole thing rests on: **the fleet digest can guard computation, not only
code.** Two substrates that disagree mid-inference produce different digests at the same
cell, and that is a witness.

## Forecast, then witness, then e-process

Each cell commits a forecast (shape and norm) *before* its value is witnessed, and the
e-process runs on the ratio of what happened to what was claimed:

```
E_t = E_{t-1} · q_t(x_t) / p_t(x_t)
```

Tested both directions: exactly-as-claimed gives `max log E = 0.000000` and no alarm;
sustained divergence trips at `max log E = 7.690` against a Ville threshold of `2.996`.

**The statement this supports, and no further:** *"no evidence against the claimed model has
been detected at level α, under the stated conditional model."* Never *"the model is true."*
The guarantee is conditional on the claimed law being correct — a weak alternative is a
power problem, a wrong null is a calibration failure where **no useful bound exists at all**.

## The honest limit, asserted as a test

There is a check in the suite that reads `cells are NOT yet cells in the full doctrinal
sense` and passes. That is deliberate. These cells have a name, a declared input set, a
witness, and a canary. They do **not** have identity, self-witness, or desire. Whether a
named function with a canary earns the word "cell" is the open question, and this
prototype is evidence toward it rather than proof of it.

## Sizes

`cellgraph.py` and `tinyformer.py` together are under 250 lines of actual code, against
`simple-llm`'s 950 for a working inference engine. The `simple-llm` standard is the
discipline being tested: **if you cannot hold the unit, it is not a cell.** A fleet with
4,869 repositories is the failure mode that standard exists to prevent.
