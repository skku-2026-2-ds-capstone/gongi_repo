import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import EXTERNAL, OUTPUTS

FIG = OUTPUTS / "figures"
BLACK, DARK, MID, LIGHT = "#000000", "#404040", "#8C8C8C", "#C8C8C8"
plt.rcParams.update({"font.family": "Malgun Gothic", "font.size": 14, "axes.unicode_minus": False,
                     "axes.spines.top": False, "axes.spines.right": False})


def fig1_forest():
    gu = pd.read_csv(OUTPUTS / "model_paired_diff.csv")
    gu = gu[gu.comparison.str.startswith("사다리")].set_index("a")
    dn = pd.read_csv(OUTPUTS / "dong_paired_diff.csv").set_index("a")
    rows = [("구 25개 · M1", gu, "M1/linear"), ("구 25개 · M2", gu, "M2/linear"),
            ("동 144개 · M1", dn, "M1/linear"), ("동 144개 · M2", dn, "M2/linear")]
    ys = [3, 2, 0.6, -0.4]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True)
    for ax, (m, lab) in zip(axes, [("auc", "AUC 차이 (고위험 선별)"), ("spearman", "ρ 차이 (순위)")]):
        for y, (name, df, key) in zip(ys, rows):
            r = df.loc[key]
            col = BLACK if "M1" in name else MID
            ax.plot([r[f"ci_{m}_lo"], r[f"ci_{m}_hi"]], [y, y], color=col, lw=2.2)
            ax.plot(r[f"diff_{m}"], y, "o", color=col, ms=9)
            ax.text(r[f"diff_{m}"], y + 0.22, f"{r[f'diff_{m}']:+.2f}".replace("-0.00", "0.00"), ha="center",
                    va="bottom", fontsize=12, color=col, bbox=dict(fc="white", ec="none", pad=1))
        ax.axvline(0, color=BLACK, lw=1)
        for v in (-0.10, 0.10):
            ax.axvline(v, color="#B0B0B0", lw=1, ls="--")
        ax.set_xlim(-0.65, 0.75)
        ax.set_xlabel(lab)
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="y", length=0)
    axes[0].set_yticks(ys, [r[0] for r in rows])
    fig.tight_layout()
    fig.savefig(FIG / "fig1_forest.png", dpi=200)


def fig2_lag_ladder():
    cv = pd.read_csv(OUTPUTS / "dong_lag_ladder.csv")
    cv = cv[cv.condition.str.startswith("기본")].set_index("model")
    tm = pd.read_csv(OUTPUTS / "dong_temporal.csv")
    tm = tm[tm["sample"].str.startswith("b")]
    tm = tm.assign(key=tm.model.str.split(" ").str[0]).set_index("key")
    models = [("M0", "M0 전세가율", LIGHT), ("M1", "M1 + 시장 변화", MID),
              ("P0", "P0 작년 값", DARK), ("P0+M1", "P0 + 시장 변화", BLACK)]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    for ax, (cvm, tmm, lab) in zip(axes, [("spearman", "spearman_2026", "ρ (순위)"),
                                          ("auc", "auc_2026_top_quartile", "AUC (상위 사분위)")]):
        x = np.arange(2)
        for i, (k, name, col) in enumerate(models):
            vals = np.array([cv.loc[k, cvm], tm.loc[k, tmm]])
            m = "spearman" if cvm == "spearman" else "auc"
            lo = np.array([cv.loc[k, f"ci_{m}_lo"], tm.loc[k, f"ci_{m}_lo"]])
            hi = np.array([cv.loc[k, f"ci_{m}_hi"], tm.loc[k, f"ci_{m}_hi"]])
            b = ax.bar(x + (i - 1.5) * 0.2, vals, 0.19, color=col, label=name,
                       yerr=[vals - lo, hi - vals], error_kw=dict(ecolor="#E06000" if col == BLACK else BLACK, lw=1.2, capsize=3))
            for xi, v, h in zip(x + (i - 1.5) * 0.2, vals, hi):
                ax.text(xi, h + 0.015, f"{v:.2f}", ha="center", va="bottom", fontsize=11.5)
        ax.set_xticks(x, [f"교차검증 2025 (n={int(cv.loc['P0', 'n'])})", f"시점 외 2026 (n={int(tm.loc['P0', 'n'])})"])
        ax.set_ylim(0, 1.12)
        ax.set_ylabel(lab)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=4, fontsize=12.5, frameon=False)
    fig.text(0.99, 0.01, "막대 위 선: 구를 다시 뽑은 bootstrap 95% CI (2,000회)", ha="right", fontsize=11.5, color=DARK)
    fig.tight_layout(rect=(0, 0.02, 1, 0.92))
    fig.savefig(FIG / "fig2_lag_ladder.png", dpi=200)


def fig3_rq1():
    gu = pd.read_csv(OUTPUTS / "rq1_gu_halfyear.csv")
    halves = sorted(gu.half.unique())
    panels = [("new_share", "신축 비중 (0~1)"), ("jr_cell", "전세가율 (전세÷매매)")]
    style = {"강서구": (BLACK, "-"), "중랑구": (BLACK, "--"), "은평구": (MID, "-"), "동작구": (MID, "--")}
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    for ax, (v, lab) in zip(axes, panels):
        w = gu.pivot(index="half", columns="gu", values=v).reindex(halves)
        sep = gu.drop_duplicates("half").set_index("half")[f"sep_{v}"].reindex(halves)
        x = np.arange(len(halves))
        for xi in x[(sep == "high>low").to_numpy()]:
            ax.axvspan(xi - 0.5, xi + 0.5, color="#EDEDED", lw=0, zorder=0)
        for g, (col, ls) in style.items():
            ax.plot(x, w[g], ls, color=col, lw=2, marker="o", ms=3.5, label=g.replace("구", ""))
        hi, lo = w[["강서구", "중랑구"]].iloc[-1].mean(), w[["은평구", "동작구"]].iloc[-1].mean()
        ax.text(0.02, 0.03, f"2023H2 구 평균  고위험 {hi:.2f} · 저위험 {lo:.2f}", transform=ax.transAxes, fontsize=12,
                va="bottom", bbox=dict(fc="white", ec="none", pad=2))
        n = int((sep == "high>low").sum())
        ax.set_title(f"구 단위 완전 분리 {n}/{sep.notna().sum()} 반기 (음영)", fontsize=13.5, loc="left")
        ax.set_xticks(x[::2], [h for h in halves[::2]], rotation=45, fontsize=12)
        ax.set_xlim(-0.6, len(halves) - 0.4)
        ax.set_ylabel(lab)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, [f"{x} (고위험)" if x in ("강서", "중랑") else f"{x} (저위험)" for x in l], loc="upper center",
               ncol=4, fontsize=12.5, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(FIG / "fig3_rq1.png", dpi=200)


def fig4_context():
    g = pd.read_csv(EXTERNAL / "lease_order_dong.csv")
    m = g.groupby("ym").n_aggregate.sum()
    m.index = pd.to_datetime(m.index)
    m = m[m.index >= "2020-01-01"]
    ev = pd.read_csv(EXTERNAL / "events.csv", parse_dates=["date"])
    fig, ax = plt.subplots(figsize=(11, 4.6))
    ax.bar(m.index, m.values, width=25, color=LIGHT)
    ax.set_ylabel("서울 임차권등기명령 (집합건물, 건/월)")
    top = m.max()
    lines = [f"{i}  {d:%Y.%m}  {lab}" for i, (d, lab) in enumerate(zip(ev.date, ev.label), 1)]
    groups = ev.assign(n=range(1, len(ev) + 1)).groupby("date").n.apply(lambda x: "·".join(map(str, x)))
    last = {}
    for d, lab in groups.items():
        level = next(k for k in range(4) if d - last.get(k, pd.Timestamp("1900-01-01")) > pd.Timedelta(days=55))
        last[level] = d
        ax.vlines(d, 0, top * (1.04 + 0.085 * level), color="#9A9A9A", lw=0.9, zorder=0.5)
        ax.text(d, top * (1.04 + 0.085 * level), lab, fontsize=12, ha="center", va="bottom", fontweight="bold",
                bbox=dict(boxstyle="circle,pad=0.25" if "·" not in lab else "round,pad=0.25", fc="white", ec=DARK, lw=0.8))
    ax.text(0.695, 0.97, "\n".join(lines), transform=ax.transAxes, fontsize=12.5, va="top", ha="left", linespacing=1.5,
            bbox=dict(fc="white", ec="none", pad=2), zorder=5)
    ax.set_ylim(0, top * 1.38)
    fig.tight_layout()
    fig.savefig(FIG / "fig4_context.png", dpi=200)


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    for f in (fig1_forest, fig2_lag_ladder, fig3_rq1, fig4_context):
        f()
        print("saved", f.__name__)


if __name__ == "__main__":
    main()
