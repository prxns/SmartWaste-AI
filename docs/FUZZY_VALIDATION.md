# Fuzzy inference validation

The master specification requires both a precise Mamdani system and global monotonicity. The specified system does not satisfy global monotonicity. This is a mathematical compatibility issue, not rounding error or an omitted rule.

## Reproducible counterexamples

```python
from smartwaste.fuzzy_engine import infer
print(infer(15, 25).score)   # 19.444444...
print(infer(15, 26).score)   # 19.63157... (increases)
print(infer(38, 100).score)  # 21.9119...
print(infer(39, 100).score)  # 21.6899... (decreases)
```

The first pair activates only Low output consequents. At contamination 25 their maximum firing strength is 1. At contamination 26 it is 0.96. Clipping the asymmetric Low trapezoid at a lower level shifts its centroid to the right. There is no better consequent, yet the crisp score rises. The second pair has the same mechanism as material memberships change at fixed High contamination.

The min AND, clipping, max aggregation and centroid implementation follows the specified definitions. A full Low trapezoid has area 37.5 and first moment 729.1666667, so its centroid is 175/9 = 19.4444444. The reflected High trapezoid has centroid 725/9 = 80.5555556. A fully activated symmetric Medium triangle has centroid 50. These analytic checks pass independently of the counterexamples.

## Tests and interpretation

`test_fuzzy_engine.py` retains both requested global monotonicity checks as `unittest.expectedFailure`. Pytest reports them as `xfailed`; unittest reports them as expected failures. They remain visible in the suite and documentation. Other tests verify exact membership boundaries, multi-rule activation, clipped max aggregation, nonuniform polygon centroids, range coverage and monotonicity at linguistic peaks.

The prototype should be described as an explainable application of the specified rules. It must not claim that every increase in contamination lowers the exact score or that every increase in material potential raises it. A future redesign may test symmetric output sets, a different inference family, or a monotone calibrated surface. Any such change needs a new rule-system specification and full-grid validation; replacing the computed score with a hardcoded formula would not implement this Mamdani system.
