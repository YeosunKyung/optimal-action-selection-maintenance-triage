# Author: Yeosun Kyung
# Email: yeosun.kyung@yonsei.ac.kr

import numpy as np

from .core import RHO, blocks, floor_actions, requested, threshold

METHODS = ["RP", "UO", "FF", "PO"]
METRICS = ["row_danger", "row_over", "row_deferred", "requests", "request_fraction_blocks",
           "posterior_objective", "critical_units", "no_post_request_units", "critical_any_danger_units",
           "no_post_fraction_critical", "danger_fraction_critical"]


def loss_matrix():
    gap = np.arange(3)[:, None] - np.arange(3)[None, :]
    return 10.0 * (gap > 0) + 0.2 * (gap < 0)


def optimize(cost, g, B, q, rho, minimum, mode):
    n = len(g)
    ix = np.arange(n)
    allowed = np.ones((n, 3), dtype=bool)
    if mode != "UO":
        allowed &= (minimum >= 0)[:, None]
    if mode == "PO":
        allowed &= np.arange(3)[None, :] >= minimum[:, None]
    c = np.where(allowed, cost, np.inf)
    inactive = 1 - np.argmin(c[:, :2][:, ::-1], axis=1)
    lv = c[ix, inactive]
    mandatory = np.bincount(g, weights=~np.isfinite(lv), minlength=B) > 0
    if q < int(mandatory.sum()):
        return None
    active = 2 - np.argmin(c[:, ::-1], axis=1)
    av = c[ix, active].copy()
    defer = rho < av
    active[defer] = -1
    av[defer] = rho
    force = ~(np.bincount(g, weights=(active < 0) | (active == 2), minlength=B) > 0)
    if force.any():
        high = np.where(rho < c[:, 2], -1, 2)
        hv = np.minimum(rho, c[:, 2])
        penalty = hv - av
        best = np.full(B, np.inf)
        np.minimum.at(best, g, penalty)
        first = np.full(B, n, dtype=int)
        np.minimum.at(first, g, np.where(penalty == best[g], ix, n))
        chosen = first[force]
        active[chosen] = high[chosen]
        av[chosen] = hv[chosen]
    high_cost = np.bincount(g, weights=av, minlength=B)
    low_cost = np.bincount(g, weights=np.where(np.isfinite(lv), lv, 0.), minlength=B)
    optional = np.flatnonzero(~mandatory)
    order = optional[np.argsort(high_cost[optional] - low_cost[optional], kind="stable")]
    selected = mandatory.copy()
    selected[order[(high_cost[order] - low_cost[order]) < 0][:q - int(mandatory.sum())]] = True
    a = np.where(selected[g], active, inactive)
    committed = a >= 0
    b = float(np.sum(cost[ix[committed], a[committed]]) / n)
    d = float((~committed).sum() / n)
    return a, b, d, int(selected.sum())


def unit_metrics(a, y, u, c):
    req = (a < 0) | (a == 2)
    crit = y == 2
    P = int(u.max()) + 1
    pos = np.bincount(u, weights=crit, minlength=P) > 0
    when = np.full(P, np.inf)
    np.minimum.at(when, u[crit & req], c[crit & req])
    danger = (a >= 0) & (a < y)
    return (float(danger.mean()), float(((a >= 0) & (a > y)).mean()), float((a < 0).mean()),
            int(pos.sum()), int(np.count_nonzero(pos & ~np.isfinite(when))),
            int(np.count_nonzero(np.bincount(u, weights=crit & danger, minlength=P) > 0)))


def row(a, b, d, k, B, y, u, c):
    danger, over, deferred, positive, nopost, cdanger = unit_metrics(a, y, u, c)
    frac = [x / positive if positive else np.nan for x in (nopost, cdanger)]
    return [danger, over, deferred, k, k / B, b + RHO * d, positive, nopost, cdanger, *frac]


def run(pool, master):
    cost_all = pool.p[:, ::-1] @ loss_matrix()
    out = np.empty((256, 4, len(METRICS)))
    for rep, perm in enumerate(pool.permutations(master)):
        cal, ev = pool.split(perm)
        th = threshold(pool.apsmax[cal])
        _, u = np.unique(pool.units[ev], return_inverse=True)
        c, y, cost = pool.cycles[ev], pool.y[ev], cost_all[ev]
        g, B = blocks(c, u)
        minimum = floor_actions(pool.inc[ev], th).astype(np.int64)
        ref = minimum
        q = int(requested(ref, g, B).sum())
        committed = ref >= 0
        n = len(ev)
        b = float(np.sum(cost[np.flatnonzero(committed), ref[committed]]) / n)
        out[rep, 0] = row(ref, b, float((~committed).sum() / n), q, B, y, u, c)
        for mi, mode in [(1, "UO"), (2, "FF"), (3, "PO")]:
            a, b, d, k = optimize(cost, g, B, q, RHO, minimum, mode)
            out[rep, mi] = row(a, b, d, k, B, y, u, c)
    return out


def summarize(values_by_input, datasets):
    groups = {}
    for v, ds in zip(values_by_input, datasets):
        loss = 10 * v[:, :, 0] + .2 * v[:, :, 1] + .1 * v[:, :, 2]
        groups.setdefault(ds, []).append(np.concatenate([v, loss[:, :, None]], axis=2))
    scopes = {ds: np.mean(v, axis=0) for ds, v in groups.items()}
    scopes["four_cmapss_equal_dataset"] = np.mean(list(scopes.values()), axis=0)
    return scopes, METRICS + ["realized_assigned_scenario_loss"]


def stats(x):
    return dict(mean=float(x.mean()), paired_assignment_mcse=float(x.std(ddof=1) / np.sqrt(len(x))),
                primary0=float(x[0]))


def tables(scopes, names):
    summary, contrasts = [], []
    for scope, a in scopes.items():
        for mi, method in enumerate(METHODS):
            for j, name in enumerate(names):
                summary.append(dict(scope=scope, method=method, metric=name, **stats(a[:, mi, j])))
        for hi, lo, label in [(3, 0, "PO_minus_RP"), (3, 1, "PO_minus_UO"), (2, 1, "FF_minus_UO"), (3, 2, "PO_minus_FF")]:
            for j, name in enumerate(names):
                contrasts.append(dict(scope=scope, contrast=label, metric=name, **stats(a[:, hi, j] - a[:, lo, j])))
    return summary, contrasts
