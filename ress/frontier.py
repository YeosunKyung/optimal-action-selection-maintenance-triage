# Author: Yeosun Kyung
# Email: yeosun.kyung@yonsei.ac.kr

from fractions import Fraction

import numpy as np

from .core import blocks, floor_actions, threshold

DISPLAY = np.linspace(0, 1, 21)
METRICS = ["predicted_cost", "recorded_loss", "additional_defer_fraction",
           "row_danger", "row_over", "unit_danger", "automatic_critical_fraction"]


def cost_matrix(p, ratio):
    s = np.arange(3)
    loss = .2 * (ratio * (s[:, None] > s[None, :]) + (s[:, None] < s[None, :]))
    return p[:, ::-1] @ loss


def curves(cost, truth, prob, floor, group, unit, rho, ratio):
    n = len(floor)
    ix = np.arange(n)
    G = int(group.max()) + 1
    eligible = floor >= 0
    mandatory = np.bincount(group, weights=(~eligible) | (floor == 2), minlength=G) > 0
    M = eligible & mandatory[group]
    allowed = eligible[:, None] & (np.arange(3)[None, :] >= floor[:, None])
    allowed[~mandatory[group], 2] = False
    masked = np.where(allowed, cost, np.inf)
    a0 = (2 - np.argmin(masked[:, ::-1], axis=1)).astype(np.int8)
    a0[~eligible] = -1
    b = np.zeros(n)
    b[eligible] = cost[ix[eligible], a0[eligible]]
    ref_cost = (cost[ix[eligible], floor[eligible]].sum() + rho * np.count_nonzero(~eligible)) / n
    auto_cost = (b.sum() + rho * np.count_nonzero(~eligible)) / n
    under = eligible & (a0 < truth)
    over = eligible & (a0 > truth)
    critical = eligible & (a0 == 2)
    positive = np.flatnonzero(M & (b > rho))
    optimal = positive[np.lexsort((positive, -b[positive]))]
    confidence = positive[np.lexsort((positive, prob[positive].max(axis=1)))]
    capacity = np.floor(DISPLAY * np.count_nonzero(M) + 1e-10).astype(int)
    capacity[-1] = np.count_nonzero(M)
    values = np.empty((2, len(DISPLAY), len(METRICS)))
    row_loss = .2 * over + (.2 * ratio) * under
    old_bad = np.bincount(unit, weights=under, minlength=int(unit.max()) + 1) > 0
    for mi, order in enumerate([optimal, confidence]):
        k = np.minimum(capacity, len(order))

        def csum(x):
            return np.r_[0., np.cumsum(x, dtype=np.float64)][k]

        gain = csum(b[order] - rho)
        danger = under.sum() - csum(under[order])
        excessive = over.sum() - csum(over[order])
        auto_critical = critical.sum() - csum(critical[order])
        recorded = (row_loss.sum() + rho * np.count_nonzero(~eligible) + csum(rho - row_loss[order])) / n
        positions = np.full(n, len(order) + 1, dtype=np.int64)
        positions[order] = np.arange(1, len(order) + 1)
        last = np.zeros(len(old_bad), dtype=np.int64)
        np.maximum.at(last, unit[under], positions[under])
        removed = np.searchsorted(np.sort(last[old_bad]), k, side="right")
        values[mi] = np.column_stack([auto_cost - gain / n, recorded, k / n, danger / n, excessive / n,
                                      (old_bad.sum() - removed) / len(old_bad), auto_critical / n])
    gain = np.r_[0., np.cumsum(b[optimal] - rho, dtype=np.float64)] / n
    full = ref_cost - auto_cost + gain[-1]
    inverse = []
    for share in [.25, .5, .75, 1.]:
        need = share * full - (ref_cost - auto_cost)
        inverse.append(0 if need <= 0 else min(int(np.searchsorted(gain, need - 2e-14)), len(optimal)))
    baseline = np.array([
        ref_cost,
        (.2 * ratio * np.count_nonzero(eligible & (floor < truth)) + .2 * np.count_nonzero(eligible & (floor > truth))
         + rho * np.count_nonzero(~eligible)) / n,
        0., np.mean(eligible & (floor < truth)), np.mean(eligible & (floor > truth)),
        np.mean(np.bincount(unit, weights=eligible & (floor < truth), minlength=len(old_bad)) > 0),
        np.mean(floor == 2)])
    geometry = dict(rows=n, groups=G, requested=int(mandatory.sum()), empty=int((~eligible).sum()),
                    eligible_inside=int(M.sum()), beneficial=len(optimal), automatic_saving=ref_cost - auto_cost,
                    protected_saving=full, inverse_counts=inverse)
    return values, baseline, geometry, optimal, b[optimal], np.sort(b[M])[::-1], auto_cost


RHO_GRID = np.arange(41, dtype=float) * .005
X_GRID = np.linspace(0, 1, 101)


def charge_grid(all_b, j0, empty, n, m):
    cb = np.r_[0., np.cumsum(all_b)]
    cap = np.floor(X_GRID * m + 1e-10).astype(int)
    cap[-1] = m
    cost = np.empty((len(RHO_GRID), len(X_GRID)))
    chosen = np.empty_like(cost)
    for ri, rho in enumerate(RHO_GRID):
        k = np.minimum(cap, int(np.searchsorted(-all_b, -rho, side="left")))
        cost[ri] = j0 + (rho - .1) * empty / n - (cb[k] - rho * k) / n
        chosen[ri] = k / n
    return cost, chosen


def run(pool, master):
    cmapss = pool.dataset.startswith("cmapss")
    values, baselines, geometry, robust, gains = [], [], [], [], []
    grid = np.empty((256, len(RHO_GRID), len(X_GRID)))
    chosen = np.empty_like(grid)
    for rep, perm in enumerate(pool.permutations(master)):
        cal, ev = pool.split(perm)
        _, unit = np.unique(pool.units[ev], return_inverse=True)
        group = blocks(pool.cycles[ev], unit)[0] if cmapss else unit
        p, y = pool.p[ev], pool.y[ev]
        floor = floor_actions(pool.inc[ev], threshold(pool.apsmax[cal]))
        v, base, g, _, b, all_b, j0 = curves(cost_matrix(p, 50.), y, p, floor, group, unit, .1, 50.)
        values.append(v)
        baselines.append(base)
        geometry.append(g)
        gains.append(b)
        grid[rep], chosen[rep] = charge_grid(all_b, j0, g["empty"], g["rows"], g["eligible_inside"])
        if rep == 0:
            for alpha in [Fraction(1, 20), Fraction(1, 10), Fraction(1, 5)]:
                f = floor_actions(pool.inc[ev], threshold(pool.apsmax[cal], alpha))
                for ratio in [1., 50.]:
                    c = cost_matrix(p, ratio)
                    for rho in [0., .1, .2]:
                        vv, bb, gg, *_ = curves(c, y, p, f, group, unit, rho, ratio)
                        robust.append(dict(alpha=float(alpha), ratio=ratio, rho=rho, values=vv.tolist(),
                                           baseline=bb.tolist(), geometry=gg))
    return dict(values=np.asarray(values), baseline=np.asarray(baselines), geometry=geometry,
                robustness=robust, gains=gains, grid=grid, chosen=chosen)


def summary_table(results, datasets):
    by = {}
    for r, ds in zip(results, datasets):
        by.setdefault(ds, []).append((r["values"], r["baseline"]))
    means = {d: (np.mean([z[0] for z in items], axis=0), np.mean([z[1] for z in items], axis=0))
             for d, items in by.items()}
    cm = [v for d, v in means.items() if d.startswith("cmapss")]
    means["four_cmapss_equal_dataset"] = (np.mean([x[0] for x in cm], axis=0), np.mean([x[1] for x in cm], axis=0))
    rows = []
    for scope, (v, base) in means.items():
        for mi, method in enumerate(["cost_priority", "confidence_priority"]):
            for ci, fraction in enumerate(DISPLAY):
                for ki, metric in enumerate(METRICS):
                    a = v[:, mi, ci, ki]
                    delta = a - base[:, ki]
                    rows.append(dict(scope=scope, method=method, capacity_fraction=float(fraction), metric=metric,
                                     mean=float(a.mean()), mcse=float(a.std(ddof=1) / np.sqrt(len(a))),
                                     paired_delta=float(delta.mean()),
                                     paired_mcse=float(delta.std(ddof=1) / np.sqrt(len(a)))))
    return rows


def full_range(results, datasets):
    out = {}
    for ds in sorted(set(datasets)):
        items = [r for r, d in zip(results, datasets) if d == ds]
        ymax = np.ceil(max(float(r["values"][:, 0, 0, 0].max()) for r in items) * 100) / 100
        density = np.zeros((220, 240), dtype=np.int64)
        points, req = 0, []
        for r in items:
            for rep, g in enumerate(r["geometry"]):
                b, n, M = r["gains"][rep], g["rows"], g["eligible_inside"]
                gain = np.r_[0., np.cumsum(b - .1)]
                B = np.arange(M + 1, dtype=np.int64)
                cost = r["values"][rep, 0, 0, 0] - gain[np.minimum(B, len(b))] / n
                xx = np.zeros_like(B) if M == 0 else np.minimum(239, (B * 240 // M))
                yy = np.clip((cost / ymax * 220).astype(int), 0, 219)
                density += np.bincount(yy * 240 + xx, minlength=220 * 240).reshape(220, 240)
                points += len(B)
                req.append(g["requested"])
        grid = np.concatenate([r["grid"] for r in items], axis=0)
        chosen = np.concatenate([r["chosen"] for r in items], axis=0)
        out[ds] = dict(n_predictors=len(items), n_allocations=len(grid), all_integer_points=points,
                       mean_cost=grid.mean(axis=0), p10_cost=np.quantile(grid, .1, axis=0),
                       p90_cost=np.quantile(grid, .9, axis=0), mean_actual_extra_review_fraction=chosen.mean(axis=0),
                       density=density, density_ymax=ymax, request_count_range_across_allocations=[min(req), max(req)])
    return dict(capacity_fraction=X_GRID, rho=RHO_GRID, focal_charge_index=20, datasets=out,
                four_cmapss_mean_cost=np.mean([v["mean_cost"] for k, v in out.items() if k.startswith("cmapss")], axis=0))


NAMES = ["cmapss_FD001", "cmapss_FD002", "cmapss_FD003", "cmapss_FD004", "uci_hydraulic", "cwru"]


def capacity_summary(results, datasets):
    by = {}
    for r, ds in zip(results, datasets):
        inv = np.array([[k / g["rows"] for k in g["inverse_counts"]] for g in r["geometry"]])
        by.setdefault(ds, []).append((r["values"], r["baseline"], inv))
    means = {ds: tuple(np.mean([a[i] for a in rows], axis=0) for i in range(3)) for ds, rows in by.items()}
    means["four_cmapss_equal_dataset"] = tuple(np.mean([means[x][i] for x in NAMES[:4]], axis=0) for i in range(3))
    summary = {}
    for ds, (v, base, inv) in means.items():
        a = v.mean(axis=0)
        b = base.mean(axis=0)
        total = b[0] - a[0, -1, 0]
        records = []
        for j in [0, 5, 10, 15, 20]:
            dd = a[0, j, 2]
            dc = a[0, j, 6] - b[6]
            saving = b[1] - a[0, j, 1]
            records.append(dict(
                capacity_fraction=j / 20, predicted=a[0, j, 0], assigned_loss=a[0, j, 1], extra_review=dd,
                critical_change=dc,
                fraction_protected_saving=None if total <= 0 else (b[0] - a[0, j, 0]) / total,
                workload_ratio_break_even=None if dc >= 0 else dd / (-dc),
                additional_review_residual_loss_break_even=None if dd <= 0 else saving / dd,
                confidence_cost_excess=a[1, j, 0] - a[0, j, 0]))
        summary[ds] = dict(
            reference_cost=b[0], reference_loss=b[1], protected_gap=total,
            automatic_fraction=(b[0] - a[0, 0, 0]) / total if total > 0 else None,
            inverse_review_fraction_mean=inv.mean(axis=0).tolist(),
            inverse_review_fraction_mcse=(inv.std(axis=0, ddof=1) / 16).tolist(),
            capacity_records=records, max_heuristic_cost_excess=float(np.max(a[1, :, 0] - a[0, :, 0])))
    return summary


def sensitivity(results, datasets):
    rows = []
    by = {}
    for r, ds in zip(results, datasets):
        by.setdefault(ds, []).append(r["robustness"])
    for ds, items in by.items():
        for j in range(18):
            vals = np.array([it[j]["values"] for it in items])
            bs = np.array([it[j]["baseline"] for it in items])
            s = items[0][j]
            full = float(np.mean(bs[:, 0] - vals[:, 0, -1, 0]))
            auto = float(np.mean(bs[:, 0] - vals[:, 0, 0, 0]))
            rows.append(dict(dataset=ds, alpha=s["alpha"], ratio=s["ratio"], rho=s["rho"], protected_saving=full,
                             automatic_saving=auto, automatic_share=auto / full if full > 1e-15 else None,
                             extra_review=float(vals[:, 0, -1, 2].mean())))
    return rows
