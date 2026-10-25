# Author: Yeosun Kyung
# Email: yeosun.kyung@yonsei.ac.kr

import numpy as np
from fractions import Fraction
from functools import lru_cache
from math import comb

from . import core

KEYS = ["units", "critical_units", "D_low", "D_high", "I_low", "I_high", "P", "Q"]


def choose(n, k):
    return comb(n, k) if 0 <= k <= n else 0


@lru_cache(maxsize=None)
def support_cutoff(U, m, epsilon=Fraction(1, 10), delta=Fraction(1, 20)):
    worst_a = [Fraction(0)] * (m + 2)
    worst_b = [Fraction(0)] * (m + 2)
    for K in range(1, U + 1):
        a, b = [0] * (m + 2), [0] * (m + 2)
        da = choose(U, m) - choose(U - K, m - K)
        db = K * choose(U - 1, m)
        for n in range(1, min(m, K - 1) + 1):
            h = K - n
            s = epsilon.numerator * h // epsilon.denominator + 1
            count = choose(K - s, n) * choose(U - K, m - n)
            a[n], b[n] = count, h * count
        for t in range(m, 0, -1):
            a[t] += a[t + 1]
            b[t] += b[t + 1]
            worst_a[t] = max(worst_a[t], Fraction(a[t], da))
            worst_b[t] = max(worst_b[t], Fraction(b[t], db))
    t = next(t for t in range(1, m + 2) if max(worst_a[t], worst_b[t]) <= delta)
    return t, worst_a[t], worst_b[t]


def guard_arrays(pool, master):
    U = len(pool.unique)
    cutoff, _, _ = support_cutoff(U, pool.m)
    upper = np.column_stack((pool.p[:, 1] + pool.p[:, 2], pool.p[:, 2], np.zeros(len(pool.y))))
    ordinal_max, critical_max = np.full(U, -np.inf), np.full(U, -np.inf)
    np.maximum.at(ordinal_max, pool.inverse, upper[np.arange(len(pool.y)), 2 - pool.y])
    critical = pool.y == 2
    np.maximum.at(critical_max, pool.inverse[critical], upper[critical, 0])
    counts_array = np.full((core.ALLOCATIONS, 2, U, 4), -1, dtype=np.int32)
    event_array = np.full((core.ALLOCATIONS, 2, U), -1, dtype=np.int16)
    for b, perm in enumerate(pool.permutations(master)):
        cal, ev = pool.split(perm)
        ids = pool.inverse[ev]
        eval_ids = np.unique(ids)
        eligible = (pool.inc[ev] <= core.threshold(pool.apsmax[cal])).any(axis=1)
        ordinal = upper[ev] <= core.threshold(ordinal_max[cal])
        positive = critical_max[cal]
        positive = positive[np.isfinite(positive)]
        q_low = core.threshold(positive)
        q_high = max(q_low, float(positive.max()) if len(positive) >= cutoff else np.inf)
        actions = []
        for g, q in enumerate((q_low, q_high)):
            kept = ordinal.copy()
            kept[:, 0] |= upper[ev, 0] <= q
            action = np.where(eligible, 2 - np.argmax(kept, axis=1), -1)
            if not kept.any(axis=1).all():
                raise ValueError("Ordinal sets must contain Normal")
            cells = np.bincount(4 * ids + action + 1, minlength=4 * U).reshape(U, 4)
            counts_array[b, g, eval_ids] = cells[eval_ids]
            active = action >= 0
            danger = active & (action < pool.y[ev])
            code = np.zeros(U, dtype=np.int16)
            for bit, event in ((1, danger), (2, danger & (pool.y[ev] == 2)),
                               (4, active & (action > pool.y[ev]))):
                code |= (np.bincount(ids, weights=event, minlength=U) > 0).astype(np.int16) * bit
            event_array[b, g, eval_ids] = code[eval_ids]
            actions.append(action)
        if np.any(actions[1] < actions[0]):
            raise ValueError("The stronger guard reduced severity")
    return dict(action_counts=counts_array, danger_event_bits=event_array,
                unit_contains_critical=np.isfinite(critical_max))


def counts(z):
    c = z["action_counts"].astype(np.int64)
    bits = z["danger_event_bits"]
    ev = bits[:, 0] >= 0
    Z = z["unit_contains_critical"].astype(bool)[None, :] & ev
    dl, dh = (((bits[:, g] & 2) > 0) & ev for g in (0, 1))
    il, ih = (ev & ((c[:, g, :, 0] > 0) | (c[:, g, :, 3] > 0)) for g in (0, 1))
    e, new = dl & ~dh, ih & ~il
    parts = [ev, Z, dl, dh, il, ih, e & il, e & new]
    return np.stack([v.sum(axis=1) for v in parts], axis=1)


def run(arrays):
    rows = [dict(zip(KEYS, map(int, n))) for z in arrays for n in counts(z)]
    t = {k: sum(r[k] for r in rows) for k in KEYS}
    out = {"allocation_predictor_pairs": len(rows), **{k + "_occurrences": v for k, v in t.items()}}
    for k in ["I_low", "I_high"]:
        out[k + "_mean_unit_fraction"] = float(np.mean([r[k] / r["units"] for r in rows]))
    out["P_share_of_error_reductions"] = t["P"] / (t["P"] + t["Q"])
    return [out]
