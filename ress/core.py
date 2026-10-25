# Author: Yeosun Kyung
# Email: yeosun.kyung@yonsei.ac.kr

import hashlib
import json
import math
from fractions import Fraction
from pathlib import Path

import numpy as np

DATA = Path(__file__).resolve().parents[1] / "data"
ALLOCATIONS = 256
RHO = 0.1


def inputs(data=DATA):
    meta = json.loads((data / "inputs.json").read_text())
    return meta["master_seed"], [(x["name"], x["dataset"]) for x in meta["inputs"]]


class Pool:
    def __init__(self, name, dataset, data=DATA):
        with np.load(data / "predictions.npz", allow_pickle=False) as z:
            self.p = z[name + "/probs"].astype(np.float64)
            self.y = z[name + "/label"].astype(np.int64)
            self.units = z[name + "/unit"].astype("<i8")
            self.cycles = z[name + "/cycle"].astype("<i8")
        if not np.isfinite(self.p).all() or np.any(self.p < 0):
            raise ValueError("Probabilities must be finite and nonnegative")
        mass = self.p.sum(axis=1, keepdims=True)
        if np.any(mass <= 0):
            raise ValueError("Each probability row must have positive mass")
        self.p = self.p / mass
        self.name = name
        self.dataset = dataset
        self.unique, self.inverse = np.unique(self.units, return_inverse=True)
        self.inc = aps_scores(self.p)
        own = self.inc[np.arange(len(self.y)), 2 - self.y]
        self.apsmax = np.full(len(self.unique), -np.inf)
        np.maximum.at(self.apsmax, self.inverse, own)
        self.identity = json.dumps({"dataset": dataset, "unit_ids": self.unique.tolist()},
                                   sort_keys=True, separators=(",", ":"))
        self.m = math.ceil(Fraction(3, 10) * len(self.unique))

    def permutations(self, master, block=0):
        key = hashlib.sha256(f"{master}|{self.identity}|block={block}".encode()).digest()
        rng = np.random.Generator(np.random.PCG64(int.from_bytes(key, "big")))
        for _ in range(ALLOCATIONS):
            yield rng.permutation(len(self.unique))

    def split(self, perm):
        rank = np.empty(len(perm), dtype=int)
        rank[perm] = np.arange(len(perm))
        ev = np.flatnonzero(rank[self.inverse] >= self.m)
        return perm[:self.m], ev


def aps_scores(p):
    order = np.argsort(-p, axis=1, kind="stable")
    sums = np.cumsum(np.take_along_axis(p, order, axis=1), axis=1, dtype=np.float64)
    sums[:, -1] = 1.0
    inc = np.empty_like(p)
    np.put_along_axis(inc, order, sums, axis=1)
    return inc


def threshold(cal_scores, alpha=Fraction(1, 10)):
    m = len(cal_scores)
    k = ((m + 1) * (alpha.denominator - alpha.numerator) + alpha.denominator - 1) // alpha.denominator
    return float(np.sort(cal_scores)[k - 1]) if k <= m else math.inf


def floor_actions(inc_rows, th):
    kept = inc_rows <= th
    return np.where(kept.any(axis=1), 2 - np.argmax(kept, axis=1), -1).astype(np.int8)


def blocks(cycles, unit, width=10):
    pairs = np.column_stack((unit.astype(np.int64), (cycles.astype(np.int64) - 1) // width))
    _, group, counts = np.unique(pairs, axis=0, return_inverse=True, return_counts=True)
    return group, len(counts)


def requested(a, g, B):
    return np.bincount(g, weights=(a < 0) | (a == 2), minlength=B) > 0
