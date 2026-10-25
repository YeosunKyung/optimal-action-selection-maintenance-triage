# Author: Yeosun Kyung
# Email: yeosun.kyung@yonsei.ac.kr

import argparse
import csv
import json
import os
import time
from pathlib import Path

for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(k, "1")

from concurrent.futures import ProcessPoolExecutor

import numpy as np

from ress import core, figures, frontier, hydraulic, nested, price

FOUR = "four_cmapss_equal_dataset"
SHORT = {"cmapss_FD001": "FD001", "cmapss_FD002": "FD002", "cmapss_FD003": "FD003", "cmapss_FD004": "FD004",
         "uci_hydraulic": "Hydraulic", "cwru": "CWRU"}
POLICIES = ["RP", "PO", "FF", "UO"]


def study(item):
    name, dataset, master = item
    pool = core.Pool(name, dataset)
    r = dict(name=name, dataset=dataset, frontier=frontier.run(pool, master))
    if dataset.startswith("cmapss"):
        r["nested"] = nested.run(pool, master)
        r["price"] = price.run(pool, master)
    return r


def save(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results")
    ap.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    args = ap.parse_args()
    out = Path(args.out)
    tab, fig = out / "tables", out / "figures"
    tab.mkdir(parents=True, exist_ok=True)
    fig.mkdir(exist_ok=True)
    start = time.time()

    master, items = core.inputs()
    with ProcessPoolExecutor(args.workers) as ex:
        res = list(ex.map(study, [(n, d, master) for n, d in items]))
    names = [r["name"] for r in res]
    datasets = [r["dataset"] for r in res]
    cm = [r for r in res if "nested" in r]
    cm_ds = [r["dataset"] for r in cm]
    print(f"{len(res)} inputs, {time.time() - start:.0f} s")
    v = {}

    scopes, metrics = nested.summarize([r["nested"] for r in cm], cm_ds)
    summary, contrasts = nested.tables(scopes, metrics)
    save(tab / "nested-summary.csv", summary)
    save(tab / "nested-contrasts.csv", contrasts)
    mean = {(r["method"], r["metric"]): r["mean"] for r in summary if r["scope"] == FOUR}
    for m in POLICIES:
        v[f"cost {m}"] = mean[m, "posterior_objective"]
        v[f"recorded loss {m}"] = mean[m, "realized_assigned_scenario_loss"]
        v[f"blocks requested % {m}"] = 100 * mean[m, "request_fraction_blocks"]
        v[f"referred % {m}"] = 100 * mean[m, "row_deferred"]
        v[f"under-triage % {m}"] = 100 * mean[m, "row_danger"]
        v[f"Critical units with no subsequent request % {m}"] = 100 * mean[m, "no_post_fraction_critical"]
    rp, po, ff, uo = (v[f"cost {m}"] for m in POLICIES)
    v["RP to PO reduction %"] = 100 * (rp - po) / rp
    v["RP-PO share of RP-UO %"] = 100 * (rp - po) / (rp - uo)
    v["PO-FF share of RP-UO %"] = 100 * (po - ff) / (rp - uo)
    v["FF-UO share of RP-UO %"] = 100 * (ff - uo) / (rp - uo)
    con = {r["contrast"]: r for r in contrasts if r["scope"] == FOUR and r["metric"] == "posterior_objective"}
    for label, key in [("J_PO-J_RP", "PO_minus_RP"), ("J_PO-J_FF", "PO_minus_FF"), ("J_FF-J_UO", "FF_minus_UO")]:
        v[f"{label} mean"] = con[key]["mean"]
        v[f"{label} MCSE"] = con[key]["paired_assignment_mcse"]
        v[f"{label} first allocation"] = con[key]["primary0"]

    psum, parts, curve = price.tables([r["price"] for r in cm], cm_ds)
    save(tab / "price-summary.csv", psum)
    save(tab / "price-decomposition.csv", parts)
    save(tab / "price-curve.csv", curve)
    for p in parts:
        if p["scope"] == FOUR:
            key = f"{p['group']}, rho {p['price']}"
            v[f"S {key}"] = p["conversion_gap"]
            v[f"S_auto {key}"] = p["automatic_residual"]
            v[f"S share of RP-UO % {key}"] = 100 * p["conversion_share"]
        if p["group"] == "block10" and p["price"] == .1:
            ds = SHORT.get(p["scope"], "four C-MAPSS")
            v[f"S share of RP-UO % {ds}"] = 100 * p["conversion_share"]
            v[f"S_auto share of S % {ds}"] = 100 * p["automatic_residual"] / p["conversion_gap"]
            if p["scope"] == FOUR:
                v["S MCSE"] = p["conversion_mcse"]
                v["S - S_auto"] = p["flexible_deferral_component"]
                v["S - S_auto share of S %"] = 100 * p["flexible_deferral_component"] / p["conversion_gap"]

    runs = [r["frontier"] for r in res]
    save(tab / "frontier-summary.csv", frontier.summary_table(runs, datasets))
    cap = frontier.capacity_summary(runs, datasets)
    four = cap[FOUR]
    zero, quarter, full = (four["capacity_records"][i] for i in (0, 1, -1))
    v["capacity: reference cost"] = four["reference_cost"]
    v["capacity: zero-review cost"] = zero["predicted"]
    v["capacity: full cost"] = full["predicted"]
    v["capacity: S_auto share of S %"] = 100 * four["automatic_fraction"]
    v["capacity: max confidence-priority excess"] = four["max_heuristic_cost_excess"]
    v["full capacity: added referrals, points"] = 100 * full["extra_review"]
    v["full capacity: automatic Critical removed, points"] = -100 * full["critical_change"]
    v["break-even t_C/t_D, full capacity"] = full["workload_ratio_break_even"]
    v["break-even t_C/t_D, quarter capacity"] = quarter["workload_ratio_break_even"]
    v["capacity: recorded loss RP"] = four["reference_loss"]
    v["capacity: recorded loss, full capacity"] = full["assigned_loss"]
    v["residual-loss threshold per review"] = full["additional_review_residual_loss_break_even"]
    for ds, short in SHORT.items():
        c = cap[ds]
        for share, x in zip([25, 50, 75, 100], c["inverse_review_fraction_mean"]):
            v[f"referrals for {share}% of S % {short}"] = 100 * x
        z, f = c["capacity_records"][0]["predicted"], c["capacity_records"][-1]["predicted"]
        v[f"endpoint zero-review cost {short}"] = z
        v[f"endpoint full cost {short}"] = f
        v[f"endpoint reduction % {short}"] = 100 * (z - f) / z
        v[f"endpoint referrals % {short}"] = 100 * c["capacity_records"][-1]["extra_review"]
    half = [v[f"referrals for 50% of S % {s}"] for s in ["FD001", "FD002", "FD003", "FD004"]]
    v["half-saving referrals, max/min over C-MAPSS"] = max(half) / min(half)

    sens = frontier.sensitivity(runs, datasets)
    save(tab / "sensitivity.csv", sens)
    for alpha in [.05, .1, .2]:
        for ratio in [1., 50.]:
            s = [x for x in sens if x["dataset"].startswith("cmapss")
                 and x["alpha"] == alpha and x["ratio"] == ratio and x["rho"] == .1]
            total = np.mean([x["protected_saving"] for x in s])
            v[f"sensitivity S, alpha {alpha}, ratio {ratio:g}"] = total
            v[f"sensitivity S_auto share %, alpha {alpha}, ratio {ratio:g}"] = (
                100 * np.mean([x["automatic_saving"] for x in s]) / total)
            v[f"sensitivity referrals %, alpha {alpha}, ratio {ratio:g}"] = 100 * np.mean([x["extra_review"] for x in s])

    rec = frontier.full_range(runs, datasets)
    v["grid: combinations"] = sum(d["n_allocations"] for d in rec["datasets"].values())
    v["grid: combinations per dataset"] = rec["datasets"]["cmapss_FD001"]["n_allocations"]
    v["grid: integer capacity points"] = sum(d["all_integer_points"] for d in rec["datasets"].values())
    for ds, rho in [("cmapss_FD002", .15), ("cmapss_FD004", .15), ("uci_hydraulic", 0.), ("uci_hydraulic", .05),
                    ("uci_hydraulic", .1), ("cwru", .1), ("cwru", .15), ("cwru", .19), ("cwru", .195)]:
        ri = int(round(rho / .005))
        v[f"selected % at rho {rho} {SHORT[ds]}"] = 100 * rec["datasets"][ds]["mean_actual_extra_review_fraction"][ri, -1]

    arrays = [hydraulic.guard_arrays(core.Pool(n, d), master) for n, d in items if d == "uci_hydraulic"]
    hyd = hydraulic.run(arrays)
    save(tab / "hydraulic.csv", hyd)
    h = hyd[0]
    v["Hydraulic unit occurrences"] = h["units_occurrences"]
    v["Hydraulic Critical-positive occurrences"] = h["critical_units_occurrences"]
    v["Hydraulic danger units, nominal guard"] = h["D_low_occurrences"]
    v["Hydraulic danger units, stronger guard"] = h["D_high_occurrences"]
    v["Hydraulic removed danger events"] = h["P_occurrences"] + h["Q_occurrences"]
    v["Hydraulic removed in requested units"] = h["P_occurrences"]
    v["Hydraulic removed in requested units %"] = 100 * h["P_share_of_error_reductions"]
    v["Hydraulic requested-unit fraction % nominal"] = 100 * h["I_low_mean_unit_fraction"]
    v["Hydraulic requested-unit fraction % stronger"] = 100 * h["I_high_mean_unit_fraction"]

    lookup = {(r["scope"], r["method"], r["capacity_fraction"], r["metric"]): r["mean"]
              for r in frontier.summary_table(runs, datasets)}
    geometry = {}
    for name, r, ds in sorted(zip(names, runs, datasets), key=lambda x: x[0]):
        geometry.setdefault(ds, []).append(r["geometry"])
    first = runs[names.index("cmapss_FD002_seed1_lgbm")]
    example = dict(values=first["values"][0], baseline=first["baseline"][0], gains=first["gains"][0],
                   geometry=first["geometry"][0])
    figures.decision_blocks(fig)
    figures.cost_referral_curves(fig, lookup, cap)
    worked = figures.target_referral_count(fig, example)
    figures.referral_sensitivity(fig, sens)
    figures.price_response(fig, [c for c in curve if c["scope"] == FOUR])
    info = figures.selected_info(rec, geometry)
    figures.cost_surfaces(fig, rec, info)
    figures.supplement_response(fig, rec, info)
    figures.capacity_distribution(fig, rec)
    for ds in ["cmapss_FD001", "cmapss_FD003"]:
        v[f"selected % at rho 0.1 {SHORT[ds]}"] = info[ds]["mean_selected_percent_of_all_observations"]
        v[f"minimum allowance % of eligible {SHORT[ds]}"] = info[ds]["mean_minimum_allowance_percent_of_eligible"]

    g = example["geometry"]
    v["example: observations"] = g["rows"]
    v["example: request blocks"] = g["groups"]
    v["example: requested blocks"] = g["requested"]
    v["example: empty sets"] = g["empty"]
    v["example: reference cost"] = float(example["baseline"][0])
    v["example: zero-review cost"] = float(example["values"][0, 0, 0])
    v["example: full saving"] = g["protected_saving"]
    v["example: half-saving target"] = worked["target"]
    v["example: minimum count"] = worked["first_count"]
    v["example: cost at minimum count"] = worked["cost_at"]
    v["example: one fewer"] = worked["first_count"] - 1
    v["example: cost with one fewer"] = worked["cost_before"]
    v["example: integer allowances"] = g["eligible_inside"] + 1
    v["example: largest allowance"] = g["eligible_inside"]

    with open(out / "numbers.json", "w") as f:
        json.dump({k: float(x) if isinstance(x, (float, np.floating)) else int(x) for k, x in v.items()}, f, indent=1)
    print(f"done in {time.time() - start:.0f} s, outputs in {out}")


if __name__ == "__main__":
    main()
