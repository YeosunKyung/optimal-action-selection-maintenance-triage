# Author: Yeosun Kyung
# Email: yeosun.kyung@yonsei.ac.kr

import logging

import matplotlib
matplotlib.use("Agg")
logging.getLogger("fontTools").setLevel(logging.ERROR)
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm, Normalize
from matplotlib.font_manager import FontProperties, findfont
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle
from matplotlib.text import Text
from matplotlib.ticker import FormatStrFormatter, FuncFormatter, MaxNLocator

GREEN = "#0D8060"
BLUE = "#4169D1"
GOLD = "#B08100"
GRAY = "#777F87"
KEYS = ["cmapss_FD001", "cmapss_FD002", "cmapss_FD003", "cmapss_FD004", "uci_hydraulic", "cwru"]

CM_STYLE = {"font.family": "serif", "font.serif": ["cmr10"], "mathtext.fontset": "cm",
            "axes.formatter.use_mathtext": True, "axes.unicode_minus": False, "font.size": 8.8,
            "axes.labelsize": 8.8, "axes.titlesize": 9.4, "xtick.labelsize": 8.2, "ytick.labelsize": 8.2,
            "axes.linewidth": .6, "axes.axisbelow": True, "xtick.direction": "out", "ytick.direction": "out",
            "pdf.fonttype": 42, "svg.fonttype": "none", "legend.frameon": True, "legend.edgecolor": "#888888",
            "legend.fancybox": False, "savefig.facecolor": "white", "figure.facecolor": "white",
            "legend.fontsize": 8.2}
SURFACE_STYLE = {"font.family": "serif", "font.serif": ["DejaVu Serif"], "mathtext.fontset": "dejavuserif",
                 "font.size": 8.2, "axes.titlesize": 9, "axes.labelsize": 8, "xtick.labelsize": 7,
                 "ytick.labelsize": 7, "axes.linewidth": .65, "pdf.fonttype": 42, "svg.fonttype": "none",
                 "savefig.facecolor": "white"}


def style(params):
    matplotlib.rc_file_defaults()
    plt.rcParams.update(params)


def save_cm(fig, out, name):
    for t in fig.findobj(Text):
        if t.get_fontweight() == "bold":
            t.set_fontproperties(FontProperties(fname=findfont("cmb10"), size=t.get_fontsize()))
    fig.canvas.draw()
    save_plain(fig, out, name, 240)


def save_plain(fig, out, name, dpi):
    fig.savefig(out / f"{name}.pdf", dpi=dpi)
    plt.close(fig)


def decision_blocks(out):
    style({"font.family": "DejaVu Sans", "font.size": 8.2, "mathtext.fontset": "dejavusans",
           "pdf.fonttype": 42, "svg.fonttype": "none", "savefig.facecolor": "white"})
    BLUE_, YELLOW, YEDGE, TEAL = "#4169D1", "#F2C438", "#8A6500", "#0D8060"
    INK, GRAY_, LIGHT = "#203247", "#7B8998", "#DCE4EE"
    fig = plt.figure(figsize=(7.4, 3.86))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set(xlim=(0, 1020), ylim=(0, 520))
    ax.axis("off")

    def text(x, y, s, size=8.2, color=INK, weight="normal", ha="center", va="center"):
        return ax.text(x, y, s, fontsize=size, color=color, fontweight=weight, ha=ha, va=va, linespacing=1.35)

    def box(x, y, w, h, fill="white", edge=LIGHT, lw=.8, r=6):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}",
                                    facecolor=fill, edgecolor=edge, linewidth=lw))

    def line(points, color=GRAY_, lw=.8):
        ax.plot([p[0] for p in points], [p[1] for p in points], color=color, lw=lw, solid_capstyle="round")

    def arrow(a, b, color=GRAY_, lw=.9):
        ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=9, color=color, lw=lw, shrinkA=0, shrinkB=0))

    for n, x, title in [(1, 24, "Set action constraints"), (2, 365, "Fix requested blocks"), (3, 708, "Select additional reviews")]:
        ax.add_patch(Circle((x, 492), 11, fc=BLUE_ if n < 3 else YELLOW, ec="none"))
        text(x, 492, str(n), 8.7, "white" if n < 3 else INK, "bold")
        text(x + 22, 492, title, 9.0, INK, "bold", ha="left")
    line([(20, 470), (322, 470)], BLUE_, .8)
    line([(354, 470), (657, 470)], BLUE_, .8)
    line([(694, 470), (1004, 470)], YELLOW, 1.1)

    box(20, 354, 85, 66, "#F5F8FE", "#A6B9E9")
    text(62.5, 395, "Prediction", 8.0)
    text(62.5, 374, r"set $C_i$", 9.5)
    line([(105, 387), (120, 387), (120, 257)], BLUE_, .9)
    arrow((120, 387), (137, 387), BLUE_)
    arrow((120, 257), (137, 257), YEDGE)
    text(227, 441, r"$C_i\ne\varnothing$", 10.5, BLUE_)
    box(138, 346, 179, 81, "#EEF3FE", "#96AFE6")
    text(227, 405, "Choose", 8.3)
    text(227, 384, r"$a_i\geq\max C_i$", 11)
    text(227, 363, "or refer for review.", 8.3)
    text(227, 313, r"$C_i=\varnothing$", 10.5, YEDGE)
    box(138, 220, 179, 74, "#FFF5BF", YELLOW)
    text(227, 257, "Review referral is\ncompulsory.", 8.8)
    text(227, 199, r"Compulsory count: $d_0$", 8.0, YEDGE)
    text(169, 156, "Normal < Warning < Critical", 8.0, INK)

    line([(317, 387), (333, 387)], BLUE_, .9)
    line([(317, 257), (333, 257)], YEDGE, .9)
    line([(333, 257), (333, 440)], BLUE_, .9)
    arrow((333, 440), (355, 440), BLUE_)
    box(355, 421, 277, 38, "white", "#B4C4E6")
    text(493.5, 440, r"Request blocks $B_g$", 8.8, BLUE_)
    arrow((500, 421), (500, 406), BLUE_)
    line([(365, 421), (365, 151)], GRAY_, .75)
    arrow((365, 151), (374, 151), GRAY_, .75)
    box(375, 250, 257, 156, "#F0F5FF", BLUE_, 1.0)
    text(503.5, 384, "Mandatory block", 9.0, BLUE_, "bold")
    for dx, dy in [(10, 10), (5, 5), (0, 0)]:
        box(393 + dx, 315 + dy, 57, 40, "#E5EDFF", "#91A7D7", .8, 3)
    text(421.5, 335, r"$\cdots$", 13, BLUE_)
    text(473, 336, "}", 25, BLUE_)
    text(549, 337, "One shared\nrequest", 8.4)
    text(503.5, 283, "Empty set, or a set whose\nmaximum is Critical.", 8.0)
    text(503.5, 228, r"$K=K_{\mathrm{RP}}$", 12.0, BLUE_)
    text(503.5, 207, "At the reference request cap", 8.0, GRAY_)
    box(375, 121, 257, 60, "#F5F7FA", "#C6D0DC", .7)
    text(503.5, 151, "Other blocks: permitted\nnontriggering automatic actions.", 8.0)

    arrow((632, 380), (691, 380), BLUE_)
    text(849, 444, "Nonempty sets in mandatory blocks", 8.1, BLUE_)
    box(695, 366, 139, 56, "#EAF8F1", "#80CDAA")
    box(860, 366, 143, 56, "#EAF8F1", "#80CDAA")
    text(764.5, 403, "Automatic cost", 7.9)
    text(764.5, 381, r"$b_i$", 11.5, TEAL)
    text(847, 394, "vs", 8.0, GRAY_)
    text(931.5, 403, "Review charge", 7.9)
    text(931.5, 381, r"$\rho$", 11.5, TEAL)
    text(849, 344, r"Positive gain: $b_i-\rho>0$", 9.0, TEAL)
    tile_x, tile_w, tile_y, tile_h = 700, 73, 286, 45
    for j, label in enumerate([r"$v_{(1)}$", r"$v_{(2)}$", r"$\cdots$", r"$v_{(L)}$"]):
        ax.add_patch(Rectangle((tile_x + j * tile_w, tile_y), tile_w, tile_h, fc="#DDF4E9", ec=TEAL, lw=.85))
        text(tile_x + (j + .5) * tile_w, tile_y + tile_h / 2, label, 11.3, TEAL)
    text(846, 266, r"$v_{(1)}\geq v_{(2)}\geq\cdots\geq v_{(L)}>0$", 9.0, TEAL)
    line([(700, 253), (700, 244), (992, 244), (992, 253)], YEDGE, .8)
    text(846, 228, r"Select up to $B$ positive gains.", 8.3, YEDGE)
    line([(846, 216), (846, 208), (758, 208), (758, 191)], YEDGE, .8)
    line([(846, 208), (939, 208), (939, 191)], BLUE_, .8)
    arrow((758, 191), (758, 180), YEDGE)
    arrow((939, 191), (939, 180), BLUE_)
    text(747, 191, "Selected", 7.9, YEDGE, ha="right")
    text(950, 191, "Others", 7.9, BLUE_, ha="left")
    box(695, 137, 139, 42, "#FFF3AD", YELLOW)
    box(860, 137, 143, 42, "#EDF3FF", "#A5B9E7")
    text(764.5, 158, "Refer for review", 8.1)
    text(931.5, 158, "Automatic", 8.6)
    text(931.5, 119, "Least permitted cost", 7.9, TEAL)

    for x, w, c, fill, title, eq in [(20, 302, BLUE_, "#EEF4FF", "REQUESTED BLOCKS", r"$K=K_{\mathrm{RP}}$"),
                                     (354, 303, YEDGE, "#FFF7CE", "ADDITIONAL REVIEWS", r"$N_D-d_0\leq B$"),
                                     (694, 310, TEAL, "#E7F7F2", "MINIMUM COST", r"$J_{\mathrm{PO}_B}$")]:
        box(x, 26, w, 65, fill, "none", 0, 5)
        line([(x, 91), (x + w, 91)], YELLOW if c == YEDGE else c, 1.3)
        text(x + w / 2, 70, title, 7.9, c, "bold")
        text(x + w / 2, 48, eq, 11.8, c)
    for x, c, label in [(171, BLUE_, "Protection and requests"), (505.5, YEDGE, "Reviews and allowance"),
                        (849, TEAL, "Costs and gains")]:
        text(x, 9, label, 7.6, c)
    save_plain(fig, out, "ress-decision-resources", 260)


def cost_referral_curves(out, lookup, capacity):
    style(CM_STYLE)
    fractions = sorted({k[2] for k in lookup})
    fig, axes = plt.subplots(2, 2, figsize=(7.4, 4.9))
    fig.subplots_adjust(left=.09, right=.98, bottom=.15, top=.94, hspace=.43, wspace=.27)
    for i, ax in enumerate(axes.flat):
        key = f"cmapss_FD00{i + 1}"
        ref = capacity[key]["reference_cost"]
        xs = np.array([100 * lookup[key, "cost_priority", f, "additional_defer_fraction"] for f in fractions])
        costs = np.array([lookup[key, "cost_priority", f, "predicted_cost"] for f in fractions])
        ys = 100 * costs / ref
        ax.axhline(100, color=GRAY, ls="--", lw=.8)
        ax.plot(xs, ys, color=GREEN, lw=1.4, marker=".", ms=3)
        ax.plot(xs[0], ys[0], marker="o", ms=5, mfc="white", mec=GREEN, zorder=5)
        ax.plot(xs[-1], ys[-1], marker="*", ms=9, mfc=GREEN, mec=GREEN, zorder=6)
        ax.set(xlim=(-.04 * xs[-1], 1.12 * xs[-1]), ylim=(49, 104),
               xlabel="Additional review referrals (% of observations)", ylabel="Minimum cost (% of reference)")
        ax.set_yticks([50, 75, 100])
        ax.xaxis.set_major_locator(MaxNLocator(4))
        ax.grid(color="#e6e6e6", lw=.45)
        ax.set_title(f"({chr(97 + i)}) C-MAPSS FD00{i + 1}", loc="left", fontweight="bold", pad=6)
        ax.text(.05, .29, f"{100 - ys[-1]:.2f}% cost reduction", transform=ax.transAxes, color=GREEN,
                fontweight="bold", fontsize=9.3)
        ax.text(.05, .17, f"{xs[-1]:.2f}% additional referrals", transform=ax.transAxes, color=GOLD, fontsize=8.8)
    fig.legend(handles=[Line2D([], [], color=GRAY, ls="--", label="Reference cost"),
                        Line2D([], [], color=GREEN, marker="o", mfc="white", lw=0, label="No additional referrals"),
                        Line2D([], [], color=GREEN, marker="*", lw=0, ms=8, label=r"Minimum cost at $\rho=0.1$")],
               loc="lower center", bbox_to_anchor=(.5, .015), ncol=3)
    save_cm(fig, out, "ress-cost-referral-curves")


def target_referral_count(out, ex):
    style(CM_STYLE)
    g, b = ex["geometry"], ex["gains"]
    j0, jr, n, m = float(ex["values"][0, 0, 0]), float(ex["baseline"][0]), g["rows"], g["eligible_inside"]
    B = np.arange(m + 1)
    j = j0 - np.r_[0., np.cumsum(b - .1)][np.minimum(B, len(b))] / n
    target = jr - .5 * (jr - j[-1])
    bstar = int(np.flatnonzero(j <= target)[0])
    fig, axes = plt.subplots(2, 1, figsize=(3.5, 4.65))
    fig.subplots_adjust(left=.235, right=.96, bottom=.16, top=.94, hspace=.48)
    ax = axes[0]
    ax.step(B, j, where="post", color=GREEN, lw=1.35)
    ax.axhline(jr, color=GRAY, ls="--", lw=.75)
    ax.axhline(target, color=BLUE, ls=":", lw=1.05)
    ax.axvline(bstar, color=GOLD, ls="--", lw=.85)
    ax.plot(bstar, j[bstar], "*", ms=9, color=GREEN)
    span = jr - float(j.min())
    ax.set(xlim=(-.02 * m, 1.015 * m), ylim=(float(j.min()) - .06 * span, jr + .12 * span), ylabel="Minimum predicted cost",
           xlabel=r"Additional-review allowance $B$ (count)")
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, p: f"{x:,.0f}"))
    ax.xaxis.set_major_locator(MaxNLocator(4))
    ax.set_title("(a) A saved FD002 evaluation batch", loc="left", fontweight="bold", pad=8)
    ax.text(.98, .92, f"Reference: {jr:.6f}", transform=ax.transAxes, ha="right", fontsize=8, color=GRAY)
    ax.text(.98, .53, f"Target: {target:.6f}", transform=ax.transAxes, ha="right", fontsize=8, color=BLUE)
    ax.text(.98, .77, f"First feasible count: {bstar:,}", transform=ax.transAxes, ha="right", fontsize=8,
            color=GREEN, fontweight="bold")
    ax.grid(color="#e6e6e6", lw=.45)
    ax = axes[1]
    zoom = np.arange(bstar - 2, bstar + 3)
    ax.step(zoom, j[zoom], where="post", color=GREEN, lw=1.1)
    ax.plot(zoom, j[zoom], "o", ms=4, mfc="white", mec=GREEN)
    ax.axhline(target, color=BLUE, ls=":", lw=1.0)
    ax.axvline(bstar, color=GOLD, ls="--", lw=.8)
    ax.plot(bstar, j[bstar], "*", ms=10, color=GREEN)
    ax.set(xlim=(bstar - 2.3, bstar + 2.3), ylabel="Minimum predicted cost",
           xlabel=r"Additional-review allowance $B$ (count)")
    ax.set_xticks(zoom[::2])
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, p: f"{x:,.0f}"))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, p: f"{x:.6f}"))
    ax.yaxis.set_major_locator(MaxNLocator(4))
    ax.set_title("(b) The first count that meets the target", loc="left", fontweight="bold", pad=8)
    ax.annotate(f"{bstar - 1:,}: above target", xy=(bstar - 1, j[bstar - 1]), xytext=(.97, .92),
                textcoords="axes fraction", fontsize=8, ha="right", va="top",
                arrowprops={"arrowstyle": "-", "color": GRAY, "lw": .6})
    ax.annotate(f"{bstar:,}: target met", xy=(bstar, j[bstar]), xytext=(.03, .1), textcoords="axes fraction",
                fontsize=8, color=GREEN, arrowprops={"arrowstyle": "-", "color": GREEN, "lw": .6})
    fig.legend(handles=[Line2D([], [], color=GREEN, marker="*", lw=0, ms=8, label="Minimum count meeting the cost target")],
               loc="lower center", bbox_to_anchor=(.5, .008), fontsize=6.9)
    ax.grid(color="#e6e6e6", lw=.45)
    save_cm(fig, out, "ress-target-referral-count")
    return dict(target=float(target), first_count=bstar, cost_before=float(j[bstar - 1]), cost_at=float(j[bstar]))


def referral_sensitivity(out, sens):
    style(CM_STYLE)
    fig, axes = plt.subplots(3, 2, figsize=(6.3, 7.4))
    fig.subplots_adjust(left=.105, right=.98, bottom=.07, top=.845, hspace=.59, wspace=.28)
    labels = ["C-MAPSS FD001", "C-MAPSS FD002", "C-MAPSS FD003", "C-MAPSS FD004", "Hydraulic", "CWRU"]
    for ax, key, label in zip(axes.flat, KEYS, labels):
        for ratio, col in [(1, BLUE), (50, GOLD)]:
            for alpha, ls in [(.05, ":"), (.1, "-"), (.2, "--")]:
                rr = sorted([d for d in sens if d["dataset"] == key and d["ratio"] == ratio and d["alpha"] == alpha],
                            key=lambda d: d["rho"])
                ax.plot([d["rho"] for d in rr], [100 * d["extra_review"] for d in rr], color=col, ls=ls, marker="o",
                        ms=3, lw=1.1, label=rf"$c_u/c_o={ratio},\ \alpha={alpha:g}$")
        ax.set(xlim=(-.01, .21), ylim=(-3, 104), xlabel=r"Cost per review referral $\rho$",
               ylabel="Additional referrals\n(% of observations)")
        ax.set_xticks([0, .1, .2])
        ax.set_yticks([0, 50, 100])
        ax.grid(color="#e6e6e6", lw=.45)
        ax.set_title(label, loc="left", fontweight="bold", pad=6)
    handles, labels_ = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels_, loc="upper center", bbox_to_anchor=(.54, .992), ncol=2)
    save_cm(fig, out, "ress-referral-sensitivity")


def price_response(out, curve):
    style(CM_STYLE)
    fig, axes = plt.subplots(2, 1, figsize=(6.3, 4.6), layout="constrained")
    for group, ls, label in [("row", ":", "One observation per request"), ("block10", "-", "Ten cycles per request"),
                             ("unit", "--", "Complete recorded unit per request")]:
        rr = sorted([v for v in curve if v["group"] == group], key=lambda v: float(v["price"]))
        for ax, key, scale, col in [(axes[0], "conversion_gap", 1, GREEN), (axes[1], "additional_deferral", 100, GOLD)]:
            ax.plot([float(v["price"]) for v in rr], [scale * float(v[key]) for v in rr], ls=ls, color=col, lw=1.2, label=label)
    for ax in axes:
        ax.set(xlim=(0, .2), xlabel=r"Cost per review referral $\rho$")
        ax.set_ylim(bottom=0)
        ax.grid(color="#e6e6e6", lw=.45)
    axes[0].set_ylabel(r"Cost saving $J_{\mathrm{RP}}-J_{\mathrm{PO}}$")
    axes[1].set_ylabel("Additional review referrals\n(% of observations)")
    axes[0].set_title("(a) Cost saving under the same action constraints", loc="left", fontweight="bold")
    axes[1].set_title("(b) Observations sent for additional review", loc="left", fontweight="bold")
    axes[0].legend(loc="upper right")
    save_cm(fig, out, "ress-price-response-lines")


INK = "#203247"
SURF_GOLD = "#F2C438"
SURF_GRAY = "#7B8998"
LABELS = ["C-MAPSS FD001", "C-MAPSS FD002", "C-MAPSS FD003", "C-MAPSS FD004", "Hydraulic", "CWRU"]


def selected_info(rec, geometry):
    info = {}
    ri = rec["focal_charge_index"]
    for key in KEYS:
        cells = [g for item in geometry[key] for g in item]
        A = 100 * np.array(rec["datasets"][key]["mean_actual_extra_review_fraction"])
        needed = np.array([g["beneficial"] for g in cells])
        m = np.array([g["eligible_inside"] for g in cells])
        frac = np.divide(needed, m, out=np.zeros(len(m), float), where=m > 0)
        info[key] = {"mean_minimum_allowance_percent_of_eligible": float(100 * frac.mean()),
                     "mean_selected_percent_of_all_observations": float(A[ri, -1])}
    return info


def surface(ax, rec, key, label):
    f = np.array(rec["capacity_fraction"])
    rho = np.array(rec["rho"])
    ri = rec["focal_charge_index"]
    F, P = np.meshgrid(f * 100, rho)
    v = rec["datasets"][key]
    Z = np.array(v["mean_cost"])
    A = 100 * np.array(v["mean_actual_extra_review_fraction"])
    ax.computed_zorder = False
    im = ax.plot_surface(A, P, Z, cmap="cividis", norm=Normalize(Z.min(), Z.max()), rcount=41, ccount=101,
                         shade=False, linewidth=.12, edgecolor=(.15, .25, .35, .13), zorder=1)
    ax.plot(A[ri], np.full(len(f), .1), Z[ri], ls="--", color=BLUE, lw=1.3, zorder=30,
            path_effects=[pe.Stroke(linewidth=2.7, foreground="white"), pe.Normal()])
    ax.scatter([A[ri, -1]], [.1], [Z[ri, -1]], marker="*", s=85, facecolor=SURF_GOLD, edgecolor=INK, depthshade=False, zorder=41)
    ax.set(xlim=(0, 1.06 * A.max()), ylim=(0, .2))
    ax.set_yticks([0, .1, .2])
    ax.xaxis.set_major_locator(MaxNLocator(3))
    ax.zaxis.set_major_locator(MaxNLocator(3))
    ax.zaxis.set_major_formatter(FormatStrFormatter("%.4f" if np.ptp(Z) < .01 else "%.3f"))
    ax.tick_params(pad=0, labelsize=7)
    ax.tick_params(axis="z", pad=2)
    ax.set_xlabel("Additional review referrals\n(% of all observations)", labelpad=3, fontsize=7.3)
    ax.set_ylabel(r"Review charge $\rho$", labelpad=2, fontsize=8)
    ax.view_init(elev=25, azim=-58)
    ax.set_box_aspect((1.6, 1, 1))
    ax.set_proj_type("ortho")
    ax.set_title(label, pad=8, fontsize=10)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.fill = False
        axis._axinfo["grid"].update(color="#e2e5ea", linewidth=.5)
    return im, Z


def surface_legend():
    return [Line2D([], [], color=BLUE, ls="--", lw=1.3, label=r"Focal review charge: $\rho=0.1$"),
            Line2D([], [], color=INK, marker="*", mfc=SURF_GOLD, ms=9, lw=0,
                   label=r"Minimum cost at $\rho=0.1$" + "\nAll positive-gain referrals selected")]


def cost_surfaces(out, rec, info):
    style(SURFACE_STYLE)
    fig = plt.figure(figsize=(6.45, 4.15))
    for j, i in enumerate([0, 2]):
        key = KEYS[i]
        ax = fig.add_axes([.005 + j * .5, .37, .46, .57], projection="3d")
        im, Z = surface(ax, rec, key, LABELS[i])
        fig.text(.488 + j * .5, .68, "Minimum predicted cost", ha="center", va="center", rotation=90, fontsize=8)
        cb = fig.colorbar(im, cax=fig.add_axes([.11 + j * .5, .255, .26, .018]), orientation="horizontal")
        cb.set_ticks([Z.min(), Z.max()])
        cb.ax.xaxis.set_major_formatter(FormatStrFormatter("%.4f"))
        cb.ax.tick_params(labelsize=7)
        fig.text(.25 + j * .5, .177, f'Selected: {info[key]["mean_selected_percent_of_all_observations"]:.2f}% of all observations',
                 ha="center", fontsize=8.1, color=GREEN, fontweight="bold")
        fig.text(.25 + j * .5, .125, f'Minimum allowance: {info[key]["mean_minimum_allowance_percent_of_eligible"]:.2f}% of eligible (mean)',
                 ha="center", fontsize=7.2, color=INK)
    fig.legend(handles=surface_legend(), loc="lower center", bbox_to_anchor=(.5, .012), ncol=2, frameon=True, fontsize=6.7)
    save_plain(fig, out, "ress-cost-surfaces-fd001-fd003", 230)


def supplement_response(out, rec, info):
    ri = rec["focal_charge_index"]
    for pair, tag in [([0, 2], "fd001-fd003"), ([1, 3], "fd002-fd004"), ([4, 5], "hydraulic-cwru")]:
        style(SURFACE_STYLE)
        fig = plt.figure(figsize=(6.45, 6.6))
        for j, i in enumerate(pair):
            key = KEYS[i]
            ax = fig.add_axes([.012 + j * .5, .545, .43, .40], projection="3d")
            im, Z = surface(ax, rec, key, LABELS[i])
            fig.text(.25 + j * .5, .515, "Minimum predicted cost", ha="center", fontsize=7.5)
            cb = fig.colorbar(im, cax=fig.add_axes([.105 + j * .5, .48, .27, .012]), orientation="horizontal")
            cb.set_ticks([Z.min(), Z.max()])
            cb.ax.xaxis.set_major_formatter(FormatStrFormatter("%.4f"))
            cb.ax.tick_params(labelsize=7)
            fig.text(.25 + j * .5, .435, f'Selected: {info[key]["mean_selected_percent_of_all_observations"]:.2f}% of all observations',
                     ha="center", fontsize=8.1, color=GREEN, fontweight="bold")
            fig.text(.25 + j * .5, .405, f'Minimum allowance: {info[key]["mean_minimum_allowance_percent_of_eligible"]:.2f}% of eligible (mean)',
                     ha="center", fontsize=7.1, color=INK)
            ax2 = fig.add_axes([.115 + j * .5, .145, .33, .21])
            X = 100 * np.array(rec["datasets"][key]["mean_actual_extra_review_fraction"])[ri]
            Y = Z[ri]
            ax2.axhline(Y[0], color=SURF_GRAY, ls=":", lw=.9)
            ax2.plot(X, Y, color=GREEN, lw=1.5)
            ax2.plot(X[-1], Y[-1], marker="*", ms=8, mfc=SURF_GOLD, mec=INK, zorder=5)
            ax2.set_title(r"Cost along the $\rho=0.1$ curve", fontsize=8.5, pad=10)
            ax2.set(xlim=(-.025 * X[-1], 1.07 * X[-1]), xlabel="Additional referrals\n(% of all observations)",
                    ylabel="Minimum predicted cost")
            ax2.yaxis.label.set_size(7.4)
            ax2.yaxis.labelpad = 3
            ax2.tick_params(labelsize=7)
            ax2.xaxis.set_major_locator(MaxNLocator(3))
            ax2.yaxis.set_major_locator(MaxNLocator(3))
            ax2.yaxis.set_major_formatter(FormatStrFormatter("%.4f" if np.ptp(Y) < .01 else "%.3f"))
            ax2.grid(color="#e5e9ed", lw=.5)
            reduction = float(100 * (Y[0] - Y[-1]) / Y[0])
            ax2.text(.96, .85, f"{reduction:.2f}% cost reduction", transform=ax2.transAxes, ha="right", fontsize=7.5,
                     color=GREEN, fontweight="bold", bbox={"fc": "white", "ec": "none", "alpha": .85, "pad": 1.5})
        handles = [Line2D([], [], color=BLUE, ls="--", lw=1.3, label=r"Focal charge: $\rho=0.1$"),
                   Line2D([], [], color=SURF_GRAY, ls=":", lw=.9, label="No additional referrals"),
                   Line2D([], [], color=INK, marker="*", mfc=SURF_GOLD, ms=8, lw=0,
                          label=r"Minimum cost at $\rho=0.1$" + "\nAll positive-gain referrals selected")]
        fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(.5, .022), ncol=3, frameon=True, fontsize=6.5,
                   columnspacing=1.0, handlelength=2)
        save_plain(fig, out, f"ress-supplement-response-{tag}", 200)


def capacity_distribution(out, rec):
    style({"font.family": "DejaVu Serif", "font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8,
           "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.linewidth": .7, "pdf.fonttype": 42,
           "svg.fonttype": "none", "savefig.facecolor": "white"})
    x = np.array(rec["capacity_fraction"]) * 100
    ri = rec["focal_charge_index"]
    blue = "#245788"
    labels = ["C-MAPSS FD001", "C-MAPSS FD002", "C-MAPSS FD003", "C-MAPSS FD004", "Hydraulic",
              "CWRU\nAll-Critical reference case"]
    fig, axs = plt.subplots(3, 2, figsize=(6.45, 7.5))
    fig.subplots_adjust(left=.10, right=.92, bottom=.09, top=.94, hspace=.56, wspace=.55)
    for ax, key, label in zip(axs.flat, KEYS, labels):
        ds = rec["datasets"][key]
        a = np.array(ds["density"])
        im = ax.imshow(np.ma.masked_equal(a, 0), origin="lower", extent=(0, 100, 0, ds["density_ymax"]), aspect="auto",
                       cmap="Greys", norm=LogNorm(1, max(2, a.max())), rasterized=True)
        ax.fill_between(x, ds["p10_cost"][ri], ds["p90_cost"][ri], color="#5584b0", alpha=.24, linewidth=0)
        ax.plot(x, ds["mean_cost"][ri], color=blue, lw=1.4)
        ax.set(xlabel="Review allowance (%)", ylabel="Minimum predicted cost", title=label)
        ax.set_xticks([0, 50, 100])
        ax.yaxis.set_major_locator(MaxNLocator(4))
        cb = fig.colorbar(im, ax=ax, fraction=.045, pad=.045)
        cb.ax.tick_params(labelsize=6)
    fig.legend(handles=[Line2D([], [], color=blue, lw=1.5, label="Mean cost"),
                        Line2D([], [], color="#5584b0", alpha=.3, lw=6, label="10th–90th percentiles")],
               loc="lower center", bbox_to_anchor=(.5, .005), ncol=2, frameon=True, fontsize=7)
    save_plain(fig, out, "ress-integer-capacity-distribution", 220)
