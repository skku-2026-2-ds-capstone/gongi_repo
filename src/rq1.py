from itertools import combinations
import warnings

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import mannwhitneyu

from config import OUTPUTS, INPUT_CUTOFF, DEEP_DIVE_GU
from dong import as_dong, unit_features
from features import load

HIGH, LOW = DEEP_DIVE_GU[:2], DEEP_DIVE_GU[2:]
VARS = ["jr_cell", "jeonse_ppsqm_yoy", "new_share"]
MIN_DONG_N = 15
FIRST_YEAR = 2016
MAIN_HALVES = ["2021H2", "2022H1", "2022H2", "2023H1", "2023H2"]


def gu_perm(vals):
    obs = vals[HIGH].mean() - vals[LOW].mean()
    d = [vals[list(h)].mean() - vals[[g for g in vals.index if g not in h]].mean() for h in combinations(vals.index, 2)]
    return obs, np.mean(np.abs(d) >= abs(obs) - 1e-12)


def gu_level(p, gu):
    rows = []
    for v in VARS:
        w = gu.pivot(index="half", columns="gu", values=v)
        blocks = [(h, w.loc[[h]]) for h in w.index] + [("주 구간 2021H2~2023H2", w.loc[w.index.isin(MAIN_HALVES)])]
        for name, b in blocks:
            vals = b.mean()
            if vals[HIGH + LOW].isna().any():
                continue
            diff, pv = gu_perm(vals[HIGH + LOW])
            sep = "high>low" if vals[HIGH].min() > vals[LOW].max() else "high<low" if vals[HIGH].max() < vals[LOW].min() else "overlap"
            rows.append({"variable": v, "block": name, **{g: vals[g] for g in HIGH + LOW},
                         "diff_gu": diff, "p_perm_gu": pv, "separation": sep})
        d = p[p.half.isin(MAIN_HALVES)].groupby(["unit", "gu", "group"])[v].mean().dropna().reset_index()
        d["high"] = (d.group == "high").astype(int)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fit = smf.mixedlm(f"{v} ~ high", d, groups=d["gu"]).fit(reml=True)
        lo, hi = fit.conf_int().loc["high"]
        rows.append({"variable": v, "block": "혼합효과 주 구간 (동 평균 ~ 고위험 + 구 무작위)", "diff_gu": fit.params["high"],
                     "mixed_ci_lo": lo, "mixed_ci_hi": hi, "mixed_p": fit.pvalues["high"],
                     "mixed_gu_var": float(fit.cov_re.iloc[0, 0]), "n_dong": len(d)})
    res = pd.DataFrame(rows)
    res.to_csv(OUTPUTS / "rq1_gu_test.csv", index=False, encoding="utf-8-sig")
    half = res[res.block.str.match(r"^\d{4}H\d$")]
    print("\n구 단위 완전 분리 반기 수 (고>저 / 전체)")
    for v, g in half.groupby("variable"):
        print(f"  {v}: {(g.separation == 'high>low').sum()}/{len(g)}  (고<저 {(g.separation == 'high<low').sum()})")
    print(res[~res.block.isin(half.block)].round(3).to_string(index=False))
    return res


def halves(last):
    out = []
    for y in range(FIRST_YEAR, last.year + 1):
        for k, (a, b) in enumerate([("01-01", "06-30"), ("07-01", "12-31")], 1):
            s, e = pd.Timestamp(f"{y}-{a}"), pd.Timestamp(f"{y}-{b}")
            if e <= last:
                out.append((f"{y}H{k}", s, e))
    return out


def holm(p):
    p = np.asarray(p, float)
    ok = ~np.isnan(p)
    q = np.full_like(p, np.nan)
    idx = np.flatnonzero(ok)[np.argsort(p[ok])]
    m, run = len(idx), 0.0
    for rank, i in enumerate(idx):
        run = max(run, min(1.0, (m - rank) * p[i]))
        q[i] = run
    return q


def main():
    t, r = load()
    t, r = (as_dong(x[x.gu.isin(HIGH + LOW)]) for x in (t, r))
    panel = []
    for name, s, e in halves(INPUT_CUTOFF):
        f = unit_features(t, r, s, e)
        prev = unit_features(t, r, s - pd.DateOffset(years=1), e - pd.DateOffset(years=1)) if s.year > FIRST_YEAR else None
        f["jeonse_ppsqm_yoy"] = f.jeonse_ppsqm / prev.jeonse_ppsqm.reindex(f.index) - 1 if prev is not None else np.nan
        f["half"] = name
        panel.append(f[["half", "n_jeonse"] + VARS])
    p = pd.concat(panel).rename_axis("unit").reset_index()
    p["gu"] = p.unit.str.split(" ").str[0]
    p["group"] = np.where(p.gu.isin(HIGH), "high", "low")
    p = p[p.n_jeonse >= MIN_DONG_N]

    rows = []
    for v in VARS:
        for h, g in p.groupby("half"):
            a, b = g.loc[g.group == "high", v].dropna(), g.loc[g.group == "low", v].dropna()
            row = {"variable": v, "half": h, "n_high": len(a), "n_low": len(b),
                   "median_high": a.median(), "median_low": b.median(), "diff": a.median() - b.median()}
            if len(a) >= 5 and len(b) >= 5:
                u, pv = mannwhitneyu(a, b, alternative="two-sided")
                row.update(r_rb=2 * u / (len(a) * len(b)) - 1, p=pv)
            rows.append(row)
    res = pd.DataFrame(rows)
    res["p_holm"] = res.groupby("variable").p.transform(holm)
    res.to_csv(OUTPUTS / "rq1_dong_halfyear.csv", index=False, encoding="utf-8-sig")

    gu = p.groupby(["half", "gu"])[VARS].median().reset_index()
    sep = []
    for (h, v), g in gu.melt(id_vars=["half", "gu"], value_vars=VARS, var_name="variable").groupby(["half", "variable"]):
        hi, lo = g[g.gu.isin(HIGH)].value, g[g.gu.isin(LOW)].value
        if hi.notna().sum() == 2 and lo.notna().sum() == 2:
            sep.append({"half": h, "variable": v,
                        "separation": "high>low" if hi.min() > lo.max() else "high<low" if hi.max() < lo.min() else "overlap"})
    gu = gu.merge(pd.DataFrame(sep).pivot(index="half", columns="variable", values="separation").add_prefix("sep_"),
                  on="half", how="left")
    gu.to_csv(OUTPUTS / "rq1_gu_halfyear.csv", index=False, encoding="utf-8-sig")

    div = []
    for v, g in res.dropna(subset=["p_holm"]).groupby("variable"):
        g = g.sort_values("half").reset_index(drop=True)
        sign = np.sign(g["diff"].iloc[-1])
        ok = (np.sign(g["diff"]) == sign) & (g.p_holm < 0.05)
        start = None
        for i in range(len(g)):
            if ok.iloc[i:].all():
                start = g.half.iloc[i]
                break
        div.append({"variable": v, "direction_2023H2": "high>low" if sign > 0 else "high<low", "diverged_from": start})
    div = pd.DataFrame(div)
    div.to_csv(OUTPUTS / "rq1_divergence.csv", index=False, encoding="utf-8-sig")

    print(res.round(3).to_string(index=False))
    print("\n" + div.to_string(index=False))
    gu_level(p, gu)


if __name__ == "__main__":
    main()
