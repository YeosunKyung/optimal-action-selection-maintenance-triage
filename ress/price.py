# Author: Yeosun Kyung
# Email: yeosun.kyung@yonsei.ac.kr

import numpy as np

from .core import blocks, floor_actions, requested, threshold
from .nested import loss_matrix, optimize

GROUPS = ["row", "block10", "unit"]
PRICES = [0., .1, .2]
METHODS = ["RP", "PO", "FF", "UO"]
METRICS = ["predicted_cost", "deferred_rows", "dangerous_rows", "over_rows", "recorded_loss", "request_fraction"]
DISPLAY = np.linspace(0, .2, 21)


def run(pool, master):
    cost_all = pool.p[:, ::-1] @ loss_matrix()
    values = np.zeros((256, 3, 3, 4, len(METRICS)))
    curves = np.zeros((256, 3, 21, 2))
    residuals = np.zeros((256, 3))
    for rep, perm in enumerate(pool.permutations(master)):
        cal, ev = pool.split(perm)
        _, u = np.unique(pool.units[ev], return_inverse=True)
        c, y, cost = pool.cycles[ev], pool.y[ev], cost_all[ev]
        n = len(ev)
        ix = np.arange(n)
        floor = floor_actions(pool.inc[ev], threshold(pool.apsmax[cal])).astype(np.int64)
        eligible = floor >= 0
        ref = floor
        g10, B10 = blocks(c, u)
        partitions = [(np.arange(n), n), (g10, B10), (u, int(u.max()) + 1)]
        permitted = np.where(eligible[:, None] & (np.arange(3)[None, :] >= floor[:, None]), cost, np.inf)
        best_a = 2 - np.argmin(permitted[:, ::-1], axis=1)
        b = permitted[ix, best_a]
        low_a = 1 - np.argmin(permitted[:, :2][:, ::-1], axis=1)
        t = permitted[ix, low_a]
        ref_req = (ref < 0) | (ref == 2)
        for gi, (g, B) in enumerate(partitions):
            mask = np.bincount(g, weights=ref_req, minlength=B) > 0
            q = int(mask.sum())
            M = eligible & mask[g]
            O = ~mask[g]
            bM = b[M]
            residuals[rep, gi] = (np.sum(cost[ix[M], floor[M]] - bM) + np.sum(cost[ix[O], floor[O]] - t[O])) / n
            for pi, rho in enumerate(PRICES):
                policies = [ref] + [optimize(cost, g, B, q, rho, floor, mode)[0]
                                    for mode in ("PO", "FF", "UO")]
                for mi, a in enumerate(policies):
                    auto = a >= 0
                    defer = float((~auto).mean())
                    danger = float((auto & (a < y)).mean())
                    over = float((auto & (a > y)).mean())
                    objective = float(cost[ix[auto], a[auto]].sum() / n + rho * defer)
                    values[rep, gi, pi, mi] = [objective, defer, danger, over, 10 * danger + .2 * over + rho * defer,
                                               float(requested(a, g, B).mean())]
            for ri, rho in enumerate(DISPLAY):
                curves[rep, gi, ri] = [residuals[rep, gi] + np.maximum(bM - rho, 0).sum() / n,
                                       np.count_nonzero(bM > rho) / n]
    return dict(values=values, curves=curves, residuals=residuals)


def tables(results, datasets):
    grouped = {}
    for r, ds in zip(results, datasets):
        grouped.setdefault(ds, []).append(r)
    means = {ds: {k: np.mean([x[k] for x in items], axis=0) for k in ("values", "curves", "residuals")}
             for ds, items in grouped.items()}
    means["four_cmapss_equal_dataset"] = {k: np.mean([v[k] for v in means.values()], axis=0)
                                          for k in ("values", "curves", "residuals")}
    summary, parts, curve = [], [], []
    for scope, data in means.items():
        v = data["values"]
        for gi, group in enumerate(GROUPS):
            for pi, price in enumerate(PRICES):
                for mi, method in enumerate(METHODS):
                    for ki, metric in enumerate(METRICS):
                        a = v[:, gi, pi, mi, ki]
                        summary.append(dict(scope=scope, group=group, price=price, method=method, metric=metric,
                                            mean=float(a.mean()), mcse=float(a.std(ddof=1) / 16)))
                gap = v[:, gi, pi, 0, 0] - v[:, gi, pi, 1, 0]
                total = v[:, gi, pi, 0, 0] - v[:, gi, pi, 3, 0]
                res = data["residuals"][:, gi]
                parts.append(dict(
                    scope=scope, group=group, price=price, conversion_gap=float(gap.mean()),
                    conversion_mcse=float(gap.std(ddof=1) / 16), automatic_residual=float(res.mean()),
                    flexible_deferral_component=float((gap - res).mean()), total_RP_UO_gap=float(total.mean()),
                    conversion_share=float(gap.mean() / total.mean()) if total.mean() > 1e-15 else None,
                    automatic_share_of_conversion=float(res.mean() / gap.mean()) if gap.mean() > 1e-15 else None))
            for ri, price in enumerate(DISPLAY):
                a = data["curves"][:, gi, ri]
                curve.append(dict(scope=scope, group=group, price=float(price), conversion_gap=float(a[:, 0].mean()),
                                  additional_deferral=float(a[:, 1].mean()), gap_mcse=float(a[:, 0].std(ddof=1) / 16)))
    return summary, parts, curve
