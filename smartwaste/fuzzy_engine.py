"""Pure NumPy Mamdani min/max inference with a piecewise-linear centroid.

No classification, Streamlit or TensorFlow dependency is required here.
"""
from dataclasses import dataclass
import numpy as np
from config import MEMBERSHIPS, RULE_TABLE


def triangle(x, a, b, c):
    if not a < b < c:
        raise ValueError("Triangle parameters must satisfy a < b < c.")
    x = np.asarray(x, dtype=float)
    return np.maximum(0.0, np.minimum((x-a)/(b-a), (c-x)/(c-b)))


def trapezoid(x, a, b, c, d):
    if not a <= b <= c <= d or a == d:
        raise ValueError("Trapezoid requires a <= b <= c <= d and a < d.")
    x = np.asarray(x, dtype=float)
    left = np.ones_like(x) if a == b else (x-a)/(b-a)
    right = np.ones_like(x) if c == d else (d-x)/(d-c)
    return np.where((x >= a) & (x <= d), np.clip(np.minimum(left, right), 0, 1), 0)


def membership(x, definition):
    kind, params = definition
    if kind == "triangle":
        return triangle(x, *params)
    if kind == "trapezoid":
        return trapezoid(x, *params)
    raise ValueError(f"Unknown membership type: {kind}")


def fuzzify(value, definitions=None):
    if not np.isfinite(value) or not 0 <= value <= 100:
        raise ValueError("Fuzzy inputs must be finite numbers in [0, 100].")
    return {name: float(membership(value, definition))
            for name, definition in (definitions or MEMBERSHIPS).items()}


def centroid(x, y):
    """Exact first moment for the polygon defined by the sampled curve."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    if x.ndim != 1 or x.shape != y.shape or len(x) < 2:
        raise ValueError("Centroid requires equally sized one-dimensional arrays.")
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)) or np.any(y < 0):
        raise ValueError("Centroid coordinates must be finite, with nonnegative membership.")
    dx = np.diff(x)
    if np.any(dx <= 0):
        raise ValueError("Output universe must be strictly increasing.")
    area = np.sum(dx * (y[:-1]+y[1:])/2)
    if area <= np.finfo(float).eps:
        raise ValueError("No rule fired: aggregated output has zero area.")
    moment = np.sum(dx * (x[:-1]*(2*y[:-1]+y[1:]) + x[1:]*(y[:-1]+2*y[1:]))/6)
    return float(moment / area)


@dataclass
class FuzzyResult:
    potential: float
    contamination: float
    memberships: dict
    rules: list
    universe: np.ndarray
    outputs: dict
    aggregate: np.ndarray
    score: float
    status: str


def infer(potential, contamination, definitions=None, rule_table=None, resolution=1001):
    definitions = definitions or MEMBERSHIPS
    names = ("Low", "Medium", "High")
    if set(definitions) != set(names) or resolution < 101:
        raise ValueError("Define Low, Medium and High sets; use at least 101 output samples.")
    p, c = fuzzify(potential, definitions), fuzzify(contamination, definitions)
    universe = np.linspace(0, 100, resolution)
    outputs = {name: membership(universe, definitions[name]) for name in names}
    aggregate = np.zeros_like(universe)
    rules = []
    table = rule_table if rule_table is not None else RULE_TABLE
    if len(table) != 3 or any(len(row) != 3 for row in table):
        raise ValueError("The rule base must contain all nine input combinations.")
    for i, material in enumerate(names):
        for j, contamination_set in enumerate(names):
            consequent = table[i][j]
            strength = min(p[material], c[contamination_set])
            aggregate = np.maximum(aggregate, np.minimum(strength, outputs[consequent]))
            rules.append({"Rule": f"R{i*3+j+1}", "Material": material,
                          "Contamination": contamination_set, "Suitability": consequent,
                          "Firing strength": strength})
    score = centroid(universe, aggregate)
    status = max(fuzzify(score, definitions), key=fuzzify(score, definitions).get)
    return FuzzyResult(float(potential), float(contamination),
                       {"Material": p, "Contamination": c}, rules, universe,
                       outputs, aggregate, score, status)
