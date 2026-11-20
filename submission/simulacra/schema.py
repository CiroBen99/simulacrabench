"""Canonical supports and cells; matches pinned make_sandbox.options_for."""
import numpy as np
import pandas as pd

def items_for(schema, classes=("GIVEN", "PREDICT")):
    return [name for name, rec in schema["items"].items() if rec["class"] in classes]

def options_for(schema, name):
    rec = schema["items"][name]
    return list(rec["values"]) + ([schema["gated_value"]] if rec.get("gate") else [])

def cells_for(frame, schema):
    items = items_for(schema)
    grid = frame[items].to_numpy(dtype=object)
    rows, columns = np.nonzero(pd.isna(grid))  # C-order preserves the contract.
    respondents = frame["respondent_id"].to_numpy()
    return [(int(row), respondents[row], items[col]) for row, col in zip(rows, columns)]

def validate_vectors(vectors, schema, cells):
    if not isinstance(vectors, (list, tuple)) or len(vectors) != len(cells):
        raise ValueError("Expected a list/tuple with one vector per canonical cell")
    widths = {item: len(options_for(schema, item)) for item in {c[2] for c in cells}}
    for vector, (_, _, item) in zip(vectors, cells):
        q = np.asarray(vector)
        if q.shape != (widths[item],) or q.dtype.kind not in "fiu":
            raise ValueError("Incorrect vector width or non-numeric probabilities")
        if not np.isfinite(q).all() or (q < 0).any() or not 0 < q.sum() < np.inf:
            raise ValueError("Probabilities must be finite, nonnegative, with finite positive sum")

def uniform_reference(schema):
    return float(np.mean([np.log(len(options_for(schema, name)))
                          for name in items_for(schema, ("PREDICT",))]))
