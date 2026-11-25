import argparse

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import spearmanr
from sklearn.base import clone
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import LeaveOneGroupOut

from config import EXTERNAL, INTERIM, OUTPUTS, PROCESSED, WINDOWS, MIN_GU_N, JR_TYPES, dataset_name
from features import load, window_slice, basic, jr_cell
from models import MODELS, SEED, N_BOOT, N_PAIR_PERM, MIN_DIFF, _to_rank

TARGET_YEAR, TEST_YEAR, LAG = 2025, 2026, 2
MIN_DENOM = MIN_GU_N
SETS = {"M0": ["jr_cell"], "M1": ["jr_cell", "jeonse_ppsqm_yoy", "jr_cell_chg"], "M2": ["jr_cell", "new_share"]}
LAG_SETS = {"P0": ["lag_log"], "P0+M0": ["lag_log"] + SETS["M0"], "P0+M1": ["lag_log"] + SETS["M1"]}
EW_TARGET = 2023
EW_SETS = {"P0": ["lag_log"], "P0+M1": ["lag_log"] + SETS["M1"], "M0": SETS["M0"], "M1": SETS["M1"], "M2": SETS["M2"]}
EW_MIN_LAG_COUNT = 6
VILLA_MIN_SHARE = 0.5
FE_SIGNS = {"jr_cell": 1, "jeonse_ppsqm_yoy": -1, "jr_cell_chg": 1, "new_share": 1}
YEAR = lambda y: (pd.Timestamp(f"{y}-01-01"), pd.Timestamp(f"{y}-12-31"))


def as_dong(df):
    return df.assign(gu=df["gu"] + " " + df["dong"])


def unit_features(t, r, start, end):
    ts, rs = window_slice(t, start, end), window_slice(r, start, end)
    return basic(ts, rs).join(jr_cell(ts, rs))


def ladder(t, r, main, prev):
    m, p = unit_features(t, r, *main), unit_features(t, r, *prev)
    return pd.DataFrame({"jr_cell": m.jr_cell, "jeonse_ppsqm_yoy": m.jeonse_ppsqm / p.jeonse_ppsqm - 1,
                         "jr_cell_chg": m.jr_cell - p.jr_cell, "new_share": m.new_share,
                         "n_jeonse_rh": m.n_jeonse, "cell_level": m.cell_level})


def jeonse_counts():
    parts, full = [], None
    for h in JR_TYPES:
        r = pd.read_csv(INTERIM / f"{dataset_name(h, 'rent')}_clean.csv",
                        usecols=["gu", "dong", "contract_date", "is_jeonse"], parse_dates=["contract_date"])
        months = r.contract_date.dt.to_period("M").drop_duplicates()
        yrs = {y for y, n in months.dt.year.value_counts().items() if n == 12}
        full = yrs if full is None else full & yrs
        r = r[r.is_jeonse]
        parts.append(r.assign(unit=r.gu + " " + r.dong, year=r.contract_date.dt.year)[["unit", "year"]])
    c = pd.concat(parts).groupby(["unit", "year"]).size().unstack("year", fill_value=0)
    return c[sorted(full)]


def registry_counts():
    g = pd.read_csv(EXTERNAL / "lease_order_dong.csv")
    g = g.assign(unit=g.gu + " " + g.dong, year=g.ym.str[:4].astype(int))
    return g.groupby(["unit", "year"]).n_aggregate.sum().unstack("year", fill_value=0)


def rate(num, den, num_years, den_years):
    n = num.reindex(columns=num_years).fillna(0).sum(axis=1)
    d = den[den_years].sum(axis=1)
    n = n.reindex(d.index).fillna(0)
    return 100 * n / d, n, d


def cv(model, X, y_log, y_cls, groups):
    reg, clf = clone(MODELS[model][0]), clone(MODELS[model][1])
    p_reg, p_cls = np.zeros(len(X)), np.zeros(len(X))
    for tr, te in LeaveOneGroupOut().split(X, groups=groups):
        reg.fit(X[tr], y_log[tr])
        p_reg[te] = reg.predict(X[te])
        if y_cls[tr].min() == y_cls[tr].max():
            p_cls[te] = y_cls[tr].mean()
        else:
            clf.fit(X[tr], y_cls[tr])
            p_cls[te] = clf.predict_proba(X[te])[:, 1]
    return p_reg, p_cls


def within_rank(x, groups):
    return pd.Series(x).groupby(np.asarray(groups)).rank(pct=True).to_numpy()


def score(p_reg, p_cls, y, y_cls, groups):
    auc = roc_auc_score(y_cls, p_cls) if 0 < y_cls.sum() < len(y_cls) else np.nan
    return (spearmanr(p_reg, y).statistic, auc,
            spearmanr(within_rank(p_reg, groups), within_rank(y, groups)).statistic)


def high(y, cut):
    return (y >= np.quantile(y, cut)).astype(int)


def _perm_once(model, X, y, cut, groups, seed):
    yy = np.random.default_rng(seed).permutation(y)
    yc = high(yy, cut)
    return score(*cv(model, X, np.log1p(yy), yc, groups), yy, yc, groups)


def cluster_boot(a, b, y, y_cls, groups, rng):
    gus = np.unique(groups)
    idx = {g: np.flatnonzero(groups == g) for g in gus}
    out = []
    for _ in range(N_BOOT):
        i = np.concatenate([idx[g] for g in rng.choice(gus, len(gus))])
        if not 0 < y_cls[i].sum() < len(i):
            continue
        sa = score(a[0][i], a[1][i], y[i], y_cls[i], groups[i])
        sb = score(b[0][i], b[1][i], y[i], y_cls[i], groups[i])
        out.append(np.subtract(sa, sb))
    return np.array(out)


def level_ci(a, y, y_cls, groups, rng):
    gus = np.unique(groups)
    idx = {g: np.flatnonzero(groups == g) for g in gus}
    out = []
    for _ in range(N_BOOT):
        i = np.concatenate([idx[g] for g in rng.choice(gus, len(gus))])
        if 0 < y_cls[i].sum() < len(i):
            out.append(score(a[0][i], a[1][i], y[i], y_cls[i], groups[i])[:2])
    lo, hi = np.nanpercentile(np.array(out), [2.5, 97.5], axis=0)
    return {"ci_spearman_lo": lo[0], "ci_spearman_hi": hi[0], "ci_auc_lo": lo[1], "ci_auc_hi": hi[1]}


def paired_perm(a, b, y, y_cls, groups, rng, by_gu):
    a, b = _to_rank(a), _to_rank(b)
    obs = np.subtract(score(*a, y, y_cls, groups), score(*b, y, y_cls, groups))
    gus, inv = np.unique(groups, return_inverse=True)
    hits = np.zeros(3)
    for _ in range(N_PAIR_PERM):
        s = rng.random(len(gus))[inv] < 0.5 if by_gu else rng.random(len(y)) < 0.5
        pa = tuple(np.where(s, x, z) for x, z in zip(a, b))
        pb = tuple(np.where(s, z, x) for x, z in zip(a, b))
        hits += np.abs(np.subtract(score(*pa, y, y_cls, groups), score(*pb, y, y_cls, groups))) >= np.abs(obs) - 1e-12
    return (hits + 1) / (N_PAIR_PERM + 1)


def target_validity(num, den):
    rows = []
    for t, hug_file, den_years in [(TARGET_YEAR, "hug_target.csv", [TARGET_YEAR - LAG]),
                                   (TEST_YEAR, "hug_target_2026.csv", [TEST_YEAR - LAG])]:
        if not set(den_years) <= set(den.columns) or t not in num.columns:
            continue
        _, n, d = rate(num, den, [t], den_years)
        gu = pd.DataFrame({"n": n, "d": d}).groupby(lambda u: u.split(" ")[0]).sum()
        gu["lease_rate"] = 100 * gu.n / gu.d
        hug = pd.read_csv(EXTERNAL / hug_file).set_index("gu").reindex(gu.index)
        rows.append({"year": t, "n_gu": len(gu),
                     "rho_vs_hug_rate": spearmanr(gu.lease_rate, hug.accident_rate).statistic,
                     "rho_count_vs_hug_count": spearmanr(gu.n, hug.accident_count).statistic})
    return pd.DataFrame(rows)


def feature_corr(s, n_boot=N_BOOT):
    rng = np.random.default_rng(SEED + 2)
    groups = s.gu.to_numpy()
    idx = {g: np.flatnonzero(groups == g) for g in np.unique(groups)}
    gus = list(idx)
    picks = [np.concatenate([idx[g] for g in rng.choice(gus, len(gus))]) for _ in range(n_boot)]
    rows = []
    for c in ["jr_cell", "jeonse_ppsqm_yoy", "jr_cell_chg", "new_share", "lag_log"]:
        x, y = s[c].to_numpy(), s.lease_rate.to_numpy()
        ok = ~np.isnan(x)
        boot = [spearmanr(x[i][ok[i]], y[i][ok[i]]).statistic for i in picks]
        lo, hi = np.percentile(boot, [2.5, 97.5])
        rows.append({"feature": c, "n": int(ok.sum()), "spearman": spearmanr(x[ok], y[ok]).statistic,
                     "ci_lo": lo, "ci_hi": hi})
    return pd.DataFrame(rows)


def judge(d, n_perm):
    groups = d.gu.to_numpy()
    conditions = {
        "기본 (상위 사분위, 2023)": ("lease_rate", 0.75, "jr_cell", "new_share", True),
        "민감도 1: 중앙값 이상": ("lease_rate", 0.50, "jr_cell", "new_share", True),
        "민감도 2: 입력 창 2021.07~2023.12": ("lease_rate", 0.75, "jr_cell_prop", "new_share_prop", True),
        "참고: 누적 타깃 2024+2025": ("lease_rate_pool", 0.75, "jr_cell", "new_share", False),
    }
    rows, comp, diffs, oof = [], [], [], []
    rng = np.random.default_rng(SEED)
    for cond, (ycol, cut, jr, new, counts) in conditions.items():
        sets = {"M0": [jr], "M1": [jr, "jeonse_ppsqm_yoy", "jr_cell_chg"], "M2": [jr, new]}
        keep = d[sorted({f for fs in sets.values() for f in fs})].notna().all(axis=1).to_numpy()
        y, g = d[ycol].to_numpy(float)[keep], groups[keep]
        yc = high(y, cut)
        base = cond.startswith("기본")
        preds = {}
        for s, feats in sets.items():
            X = d.loc[keep, feats].to_numpy(float)
            for model in (MODELS if base else ["linear"]):
                p = preds[(s, model)] = cv(model, X, np.log1p(y), yc, g)
                rho, auc, wrho = score(*p, y, yc, g)
                if base:
                    null = np.array(Parallel(n_jobs=-1)(delayed(_perm_once)(model, X, y, cut, g, SEED + k)
                                                        for k in range(n_perm))) if model == "linear" else None
                    pv = (lambda j, obs: (np.sum(null[:, j] >= obs) + 1) / (n_perm + 1)) if null is not None \
                        else (lambda j, obs: np.nan)
                    comp.append({"feature_set": s, "model": model, "features": "+".join(feats), "n": int(keep.sum()),
                                 "n_high": int(yc.sum()), "spearman": rho, "p_perm_spearman": pv(0, rho),
                                 "auc": auc, "p_perm_auc": pv(1, auc), "within_gu_spearman": wrho,
                                 "p_perm_within_gu": pv(2, wrho)})
                    oof.append(pd.DataFrame({"unit": d.index[keep], "gu": g, "feature_set": s, "model": model,
                                             "lease_rate": y, "high_risk": yc,
                                             "pred_log1p_rate": p[0], "pred_prob_high": p[1]}))
        sc = {k: score(*v, y, yc, g) for k, v in preds.items()}
        for s in ["M1", "M2"]:
            a, b = sc[(s, "linear")], sc[("M0", "linear")]
            rows.append({"condition": cond, "in_verdict": counts, "n": int(keep.sum()), "n_high": int(yc.sum()),
                         "model": s, "features": "+".join(sets[s]),
                         "rho_m0": b[0], "rho": a[0], "rho_higher": a[0] > b[0],
                         "auc_m0": b[1], "auc": a[1], "auc_higher": a[1] > b[1],
                         "within_m0": b[2], "within": a[2]})
        if base:
            for k in preds:
                if k == ("M0", "linear"):
                    continue
                bt = cluster_boot(preds[k], preds[("M0", "linear")], y, yc, g, rng)
                lo, hi = np.nanpercentile(bt, [2.5, 97.5], axis=0)
                obs = np.subtract(sc[k], sc[("M0", "linear")])
                p_dong = paired_perm(preds[k], preds[("M0", "linear")], y, yc, g, rng, by_gu=False)
                p_gu = paired_perm(preds[k], preds[("M0", "linear")], y, yc, g, rng, by_gu=True)
                diffs.append({"a": "/".join(k), "b": "M0/linear",
                              "diff_spearman": obs[0], "ci_spearman_lo": lo[0], "ci_spearman_hi": hi[0],
                              "p_pair_dong_spearman": p_dong[0], "p_pair_gu_spearman": p_gu[0],
                              "diff_auc": obs[1], "ci_auc_lo": lo[1], "ci_auc_hi": hi[1],
                              "p_pair_dong_auc": p_dong[1], "p_pair_gu_auc": p_gu[1],
                              "diff_within_gu": obs[2], "ci_within_lo": lo[2], "ci_within_hi": hi[2],
                              "p_pair_dong_within": p_dong[2], "p_pair_gu_within": p_gu[2],
                              "meaningful_spearman": abs(obs[0]) >= MIN_DIFF and (lo[0] > 0 or hi[0] < 0),
                              "meaningful_auc": abs(obs[1]) >= MIN_DIFF and (lo[1] > 0 or hi[1] < 0)})
    res = pd.DataFrame(rows)
    res["both_higher"] = res.rho_higher & res.auc_higher
    verdict = res[res.in_verdict].groupby("model").both_higher.all().map({True: "RQ2 긍정", False: "RQ2 긍정 아님"})
    return res, verdict, pd.DataFrame(comp), pd.DataFrame(diffs), pd.concat(oof)


def lag_ladder(d):
    s = d[d.in_sample & (d.denom_lag >= MIN_DENOM)]
    y, g = s.lease_rate.to_numpy(float), s.gu.to_numpy()
    sets = {**LAG_SETS, "M0": SETS["M0"], "M1": SETS["M1"]}
    rng, rng_ci = np.random.default_rng(SEED), np.random.default_rng(SEED + 1)
    rows, diffs = [], []
    for cond, cut in [("기본 (상위 사분위)", 0.75), ("민감도: 중앙값 이상", 0.50)]:
        yc = high(y, cut)
        preds = {k: cv("linear", s[f].to_numpy(float), np.log1p(y), yc, g) for k, f in sets.items()}
        sc = {k: score(*p, y, yc, g) for k, p in preds.items()}
        rows += [{"condition": cond, "model": k, "features": "+".join(sets[k]), "n": len(s), "n_high": int(yc.sum()),
                  "spearman": v[0], "auc": v[1], "within_gu_spearman": v[2],
                  **level_ci(preds[k], y, yc, g, rng_ci)} for k, v in sc.items()]
        for k in ["P0+M0", "P0+M1"]:
            bt = cluster_boot(preds[k], preds["P0"], y, yc, g, rng)
            lo, hi = np.nanpercentile(bt, [2.5, 97.5], axis=0)
            pp = paired_perm(preds[k], preds["P0"], y, yc, g, rng, by_gu=True)
            obs = np.subtract(sc[k], sc["P0"])
            diffs.append({"condition": cond, "a": k, "b": "P0",
                          "diff_spearman": obs[0], "ci_spearman_lo": lo[0], "ci_spearman_hi": hi[0], "p_pair_gu_spearman": pp[0],
                          "diff_auc": obs[1], "ci_auc_lo": lo[1], "ci_auc_hi": hi[1], "p_pair_gu_auc": pp[1],
                          "both_higher": obs[0] > 0 and obs[1] > 0})
    diffs = pd.DataFrame(diffs)
    adds = diffs[diffs.a == "P0+M1"].both_higher.all()
    verdict = "시장 피처가 작년 값에 정보를 더함" if adds else "시장 피처가 작년 값에 더하는 정보 없음 → 기여는 설명 쪽"
    return pd.DataFrame(rows), diffs, verdict


def temporal(t, r, num, den, d):
    if TEST_YEAR - LAG not in den.columns:
        print(f"시점 외 검증 건너뜀: {TEST_YEAR - LAG}년 아파트·오피스텔 전월세 정제 자료 없음")
        return None
    X24 = ladder(t, r, YEAR(TEST_YEAR - LAG), YEAR(TEST_YEAR - LAG - 1))
    y26, _, d24 = rate(num, den, [TEST_YEAR], [TEST_YEAR - LAG])
    te = X24.assign(lease_rate_2026=y26, denom_2026=d24).join(d[["lease_rate"]].rename(columns={"lease_rate": "lease_rate_2025"}))
    te = te[(te.denom_2026 >= MIN_DENOM) & (te.n_jeonse_rh >= MIN_GU_N)].dropna(
        subset=["jr_cell", "jeonse_ppsqm_yoy", "jr_cell_chg", "new_share", "lease_rate_2026", "lease_rate_2025"])
    te["lag_log"] = np.log1p(te.lease_rate_2025)
    tr = d[d.in_sample]
    tr_lag = tr[tr.denom_lag >= MIN_DENOM]
    fitted = {}
    for name, feats, train in ([(s, f, tr) for s, f in SETS.items()] + [(s, f, tr_lag) for s, f in LAG_SETS.items()]):
        y_tr = train.lease_rate.to_numpy(float)
        reg = clone(MODELS["linear"][0]).fit(train[feats].to_numpy(float), np.log1p(y_tr))
        clf = clone(MODELS["linear"][1]).fit(train[feats].to_numpy(float), high(y_tr, 0.75))
        fitted[f"{name} ({'+'.join(feats)})"] = (feats, reg, clf)
    rng, rng_ci = np.random.default_rng(SEED), np.random.default_rng(SEED + 1)

    def run(sub, label):
        y_te, g_te = sub.lease_rate_2026.to_numpy(float), sub.index.str.split(" ").str[0].to_numpy()
        hi_te = high(y_te, 0.75)

        def ev(name, pr, pc):
            rho, auc, wrho = score(pr, pc, y_te, hi_te, g_te)
            null = np.array([score(pr, pc, yy, high(yy, 0.75), g_te)[:2] for yy in (rng.permutation(y_te) for _ in range(2000))])
            return {"sample": label, "model": name, "n": len(sub), "n_high": int(hi_te.sum()), "spearman_2026": rho,
                    "p_perm_spearman": (np.sum(null[:, 0] >= rho) + 1) / 2001, "auc_2026_top_quartile": auc,
                    "p_perm_auc": (np.sum(null[:, 1] >= auc) + 1) / 2001, "within_gu_spearman": wrho,
                    **level_ci((pr, pc), y_te, hi_te, g_te, rng_ci)}

        out = [ev("기준값: 2025 비율 그대로", sub.lease_rate_2025.to_numpy(), sub.lease_rate_2025.to_numpy()),
               ev("기준값: 2024 전세가율 원값", sub.jr_cell.to_numpy(), sub.jr_cell.to_numpy())]
        for name, (feats, reg, clf) in fitted.items():
            X = sub[feats].to_numpy(float)
            out.append(ev(name, reg.predict(X), clf.predict_proba(X)[:, 1]))
        return out

    both = te[te.index.isin(tr.index)]
    return pd.DataFrame(run(te, "a 2026 조건 충족 전체") + run(both, "b 판정 표본과 겹치는 동"))


def early_frame(t, r, num, den):
    ty = EW_TARGET
    d = ladder(t, r, YEAR(ty - LAG), YEAR(ty - LAG - 1))
    d["lease_rate"], d["num"], d["denom"] = rate(num, den, [ty], [ty - LAG])
    lag, d["num_lag"], d["denom_lag"] = rate(num, den, [ty - 1], [ty - 1 - LAG])
    d["lag_log"] = np.log1p(lag)
    d["gu"] = d.index.str.split(" ").str[0]
    feats = sorted({f for fs in EW_SETS.values() for f in fs})
    return d[(d.denom >= MIN_DENOM) & (d.denom_lag >= MIN_DENOM) & (d.n_jeonse_rh >= MIN_GU_N)
             & d[feats + ["lease_rate"]].notna().all(axis=1)]


def main_lag_frame(num):
    d = pd.read_csv(PROCESSED / "dong_features.csv").set_index("unit")
    d = d[d.in_sample & (d.denom_lag >= MIN_DENOM)].copy()
    d["num_lag"] = num.get(TARGET_YEAR - 1, pd.Series(dtype=float)).reindex(d.index).fillna(0)
    return d


def ladder_metrics(s, sets, rng, rng_ci, label, pairs):
    y, g = s.lease_rate.to_numpy(float), s.gu.to_numpy()
    yc = high(y, 0.75)
    preds = {k: cv("linear", s[f].to_numpy(float), np.log1p(y), yc, g) for k, f in sets.items()}
    sc = {k: score(*p, y, yc, g) for k, p in preds.items()}
    rows = [{"check": label[0], "period": label[1], "model": k, "n": len(s), "n_gu": int(pd.Series(g).nunique()),
             "n_high": int(yc.sum()), "spearman": v[0], "auc": v[1], **level_ci(preds[k], y, yc, g, rng_ci)}
            for k, v in sc.items()]
    diffs = []
    for a, b in pairs:
        bt = cluster_boot(preds[a], preds[b], y, yc, g, rng)
        lo, hi = np.nanpercentile(bt, [2.5, 97.5], axis=0)
        obs = np.subtract(sc[a], sc[b])
        diffs.append({"check": label[0], "period": label[1], "a": a, "b": b,
                      "diff_spearman": obs[0], "ci_spearman_lo": lo[0], "ci_spearman_hi": hi[0],
                      "diff_auc": obs[1], "ci_auc_lo": lo[1], "ci_auc_hi": hi[1],
                      "meets_rule": bool(obs[0] >= MIN_DIFF and obs[1] >= MIN_DIFF and lo[0] > 0 and lo[1] > 0)})
    return rows, diffs


def early_warning_sensitivity(t, r, num, den):
    before, after = early_frame(t, r, num, den), main_lag_frame(num)
    sets = {k: EW_SETS[k] for k in ["P0", "P0+M1", "M0", "M1"]}
    pairs = [("M1", "P0"), ("P0+M1", "P0")]
    rng, rng_ci = np.random.default_rng(SEED + 5), np.random.default_rng(SEED + 6)
    common = before.index.intersection(after.index)
    checks = [(("a 공통 동", "급증 직전 (2023)"), before.loc[common]), (("a 공통 동", "급증 이후 (2025)"), after.loc[common]),
              ((f"b 작년 {EW_MIN_LAG_COUNT}건 이상", "급증 직전 (2023)"), before[before.num_lag >= EW_MIN_LAG_COUNT]),
              ((f"b 작년 {EW_MIN_LAG_COUNT}건 이상", "급증 이후 (2025)"), after[after.num_lag >= EW_MIN_LAG_COUNT])]
    rows, diffs = [], []
    for label, s in checks:
        r_, d_ = ladder_metrics(s, sets, rng, rng_ci, label, pairs)
        rows += r_
        diffs += d_
    rows, diffs = pd.DataFrame(rows), pd.DataFrame(diffs)
    verdicts = []
    for chk in rows.check.unique():
        p = rows[(rows.check == chk) & (rows.model == "P0")].set_index("period")
        b, a = p.loc["급증 직전 (2023)"], p.loc["급증 이후 (2025)"]
        m1 = diffs[(diffs.check == chk) & (diffs.period == "급증 직전 (2023)") & (diffs.a == "M1")].iloc[0]
        inside = a.ci_spearman_lo <= b.spearman <= a.ci_spearman_hi
        if chk.startswith("a"):
            v = ("시기 차이가 아니라 표본 차이였을 수 있음" if inside else
                 "결론 유지: 공통 동에서도 작년 값 우위는 급증 이후" if b.spearman < a.spearman and not m1.meets_rule else
                 "결론 수정 필요")
        else:
            v = "측정 잡음(희소성)이 주원인일 수 있음" if inside else "희소성만으로는 설명되지 않음"
        verdicts.append(f"{chk}: 직전 P0 ρ {b.spearman:.3f} (n {int(b.n)}) vs 이후 {a.spearman:.3f} "
                        f"[{a.ci_spearman_lo:.3f}, {a.ci_spearman_hi:.3f}] (n {int(a.n)}), 직전 M1 − P0 기준 충족 {m1.meets_rule} → {v}")
    return rows, diffs, verdicts


def villa_counts():
    d = pd.read_csv(PROCESSED / "dong_features.csv")
    s = d[d.in_sample].assign(villa_share=lambda x: x.n_jeonse_rh / x.denom)
    v = s[s.villa_share >= VILLA_MIN_SHARE]
    c = v.groupby("gu").size().rename("n_dong").reindex(sorted(s.gu.unique()), fill_value=0).reset_index()
    summary = {"n_sample": len(s), "villa_share_median": float(s.villa_share.median()), "n_villa_dong": len(v),
               "n_gu_with_any": int((c.n_dong > 0).sum()), "n_gu_with_one": int((c.n_dong == 1).sum()),
               "n_gu_with_2plus": int((c.n_dong >= 2).sum()), "max_per_gu": int(c.n_dong.max())}
    return c, summary


def _ols(X, y, groups=None):
    if groups is not None:
        X = X - pd.DataFrame(X).groupby(groups).transform("mean").to_numpy()
        y = y - pd.Series(y).groupby(groups).transform("mean").to_numpy()
    else:
        X = np.column_stack([np.ones(len(y)), X])
    b = np.linalg.lstsq(X, y, rcond=None)[0]
    return b if groups is not None else b[1:]


def fixed_effects(n_boot=N_BOOT):
    d = pd.read_csv(PROCESSED / "dong_features.csv")
    s = d[d.in_sample].reset_index(drop=True)
    feats = list(FE_SIGNS)
    Z = ((s[feats] - s[feats].mean()) / s[feats].std(ddof=0)).to_numpy(float)
    g = s.gu.to_numpy()
    gus = np.unique(g)
    idx = {k: np.flatnonzero(g == k) for k in gus}
    rng = np.random.default_rng(SEED + 7)
    draws = [rng.choice(gus, len(gus), replace=True) for _ in range(n_boot)]
    rows = []
    for spec, target, fe in [("주: 구 고정효과, log(1+비율)", np.log1p(s.lease_rate.to_numpy(float)), True),
                             ("참고: 구 고정효과 없음, log(1+비율)", np.log1p(s.lease_rate.to_numpy(float)), False),
                             ("참고: 구 고정효과, 비율 그대로", s.lease_rate.to_numpy(float), True)]:
        est = _ols(Z, target, g if fe else None)
        bt = []
        for pick in draws:
            ii = np.concatenate([idx[k] for k in pick])
            lab = np.concatenate([np.full(len(idx[k]), j) for j, k in enumerate(pick)])
            bt.append(_ols(Z[ii], target[ii], lab if fe else None))
        lo, hi = np.percentile(np.array(bt), [2.5, 97.5], axis=0)
        for j, f in enumerate(feats):
            sign_ok = np.sign(est[j]) == FE_SIGNS[f]
            excl = lo[j] > 0 or hi[j] < 0
            rows.append({"spec": spec, "feature": f, "hyp_sign": "+" if FE_SIGNS[f] > 0 else "−", "coef_per_sd": est[j],
                         "ci_lo": lo[j], "ci_hi": hi[j], "sign_matches": bool(sign_ok), "ci_excludes_0": bool(excl),
                         "explains": bool(sign_ok and excl), "n": len(s), "n_gu": len(gus)})
    return pd.DataFrame(rows)


def early_warning(t, r, num, den):
    s = early_frame(t, r, num, den)
    y, g = s.lease_rate.to_numpy(float), s.gu.to_numpy()
    rng, rng_ci = np.random.default_rng(SEED + 3), np.random.default_rng(SEED + 4)
    rows, diffs = [], []
    for cond, cut in [("기본 (상위 사분위)", 0.75), ("민감도: 중앙값 이상", 0.50)]:
        yc = high(y, cut)
        preds = {k: cv("linear", s[f].to_numpy(float), np.log1p(y), yc, g) for k, f in EW_SETS.items()}
        sc = {k: score(*p, y, yc, g) for k, p in preds.items()}
        rows += [{"condition": cond, "model": k, "features": "+".join(EW_SETS[k]), "n": len(s),
                  "n_high": int(yc.sum()), "spearman": v[0], "auc": v[1], "within_gu_spearman": v[2],
                  **level_ci(preds[k], y, yc, g, rng_ci)} for k, v in sc.items()]
        for a, b in [("M1", "P0"), ("P0+M1", "P0"), ("M1", "M0")]:
            bt = cluster_boot(preds[a], preds[b], y, yc, g, rng)
            lo, hi = np.nanpercentile(bt, [2.5, 97.5], axis=0)
            pp = paired_perm(preds[a], preds[b], y, yc, g, rng, by_gu=True)
            obs = np.subtract(sc[a], sc[b])
            diffs.append({"condition": cond, "a": a, "b": b,
                          "diff_spearman": obs[0], "ci_spearman_lo": lo[0], "ci_spearman_hi": hi[0], "p_pair_gu_spearman": pp[0],
                          "diff_auc": obs[1], "ci_auc_lo": lo[1], "ci_auc_hi": hi[1], "p_pair_gu_auc": pp[1],
                          "meets_rule": bool(obs[0] >= MIN_DIFF and obs[1] >= MIN_DIFF and lo[0] > 0 and lo[1] > 0)})
    diffs = pd.DataFrame(diffs)
    rule = diffs[(diffs.a == "M1") & (diffs.b == "P0") & diffs.condition.str.startswith("기본")].meets_rule.iloc[0]
    verdict = "사고가 드물 때는 시장 피처가 유용" if rule else "사고가 드물 때도 시장 피처가 작년 값보다 낫다는 근거 없음"
    sparse = {"n": len(s), "lag_count_le5_share": float((s.num_lag <= 5).mean()),
              "lag_count_zero_share": float((s.num_lag == 0).mean()),
              "target_rate_median": float(s.lease_rate.median()), "lag_rate_median": float(np.expm1(s.lag_log).median())}
    return pd.DataFrame(rows), diffs, verdict, sparse


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-perm", type=int, default=1000)
    ap.add_argument("--skip-judge", action="store_true",
                    help="판정·모형 비교(순열, 몇 분)를 건너뛰고 작년 값 사다리·시점 외 검증만 다시 계산")
    ap.add_argument("--early-warning", action="store_true", help="조기 경보 검증(2021 입력 → 2023 비율)만 계산")
    ap.add_argument("--early-warning-sens", action="store_true", help="조기 경보 민감도(공통 동, 작년 6건 이상)만 계산")
    ap.add_argument("--villa-counts", action="store_true", help="빌라 비중 50% 이상 동의 구별 개수만 센다")
    ap.add_argument("--fixed-effects", action="store_true", help="구 고정효과 설명 모형(동 144개)만 계산")
    args = ap.parse_args()

    if args.fixed_effects:
        fe = fixed_effects()
        fe.to_csv(OUTPUTS / "dong_fixed_effects.csv", index=False, encoding="utf-8-sig")
        print(fe.round(3).to_string(index=False))
        return

    if args.villa_counts:
        c, sm = villa_counts()
        c.to_csv(OUTPUTS / "dong_villa_counts.csv", index=False, encoding="utf-8-sig")
        print(sm)
        print(c[c.n_dong > 0].to_string(index=False))
        return

    t, r = (as_dong(x) for x in load())
    num, den = registry_counts(), jeonse_counts()
    print(f"분모(3개 유형 전세)가 있는 해: {list(den.columns)}")

    if args.early_warning:
        ew, ewd, ewv, sp = early_warning(t, r, num, den)
        ew.to_csv(OUTPUTS / "dong_early_warning.csv", index=False, encoding="utf-8-sig")
        ewd.to_csv(OUTPUTS / "dong_early_warning_diff.csv", index=False, encoding="utf-8-sig")
        print("\n조기 경보 표본: " + ", ".join(f"{k} {v:.3g}" for k, v in sp.items()))
        print("\n조기 경보 사다리 (2021 입력 → 2023 비율)\n" + ew.drop(columns="features").round(3).to_string(index=False))
        print("\n차이 (구 군집 bootstrap 95% CI, 쌍체 순열 구 통째)\n" + ewd.round(3).to_string(index=False))
        print("\n판정: " + ewv)
        return

    if args.early_warning_sens:
        es, esd, ev = early_warning_sensitivity(t, r, num, den)
        es.to_csv(OUTPUTS / "dong_early_warning_sens.csv", index=False, encoding="utf-8-sig")
        esd.to_csv(OUTPUTS / "dong_early_warning_sens_diff.csv", index=False, encoding="utf-8-sig")
        print("\n조기 경보 민감도\n" + es.round(3).to_string(index=False))
        print("\n차이 (구 군집 bootstrap 95% CI)\n" + esd.round(3).to_string(index=False))
        print("\n판정\n" + "\n".join(ev))
        return

    d = ladder(t, r, WINDOWS["main"], WINDOWS["prev"])
    prop = unit_features(t, r, *WINDOWS["prop"])
    d["jr_cell_prop"], d["new_share_prop"] = prop.jr_cell, prop.new_share
    d["lease_rate"], d["num"], d["denom"] = rate(num, den, [TARGET_YEAR], [TARGET_YEAR - LAG])
    d["lease_rate_pool"], d["num_pool"], d["denom_pool"] = rate(num, den, [TARGET_YEAR - 1, TARGET_YEAR],
                                                                [TARGET_YEAR - LAG - 1, TARGET_YEAR - LAG])
    d["lease_rate_lag"], _, d["denom_lag"] = rate(num, den, [TARGET_YEAR - 1], [TARGET_YEAR - 1 - LAG])
    d["lag_log"] = np.log1p(d.lease_rate_lag)
    d.index.name = "unit"
    d["gu"] = d.index.str.split(" ").str[0]
    d["in_sample"] = ((d.denom >= MIN_DENOM) & (d.n_jeonse_rh >= MIN_GU_N)
                      & d[["jr_cell", "jeonse_ppsqm_yoy", "jr_cell_chg", "new_share", "lease_rate"]].notna().all(axis=1))
    d.reset_index().to_csv(PROCESSED / "dong_features.csv", index=False, encoding="utf-8-sig")
    s = d[d.in_sample]
    print(f"동 {len(d)}개 중 표본 {len(s)}개 ({s.gu.nunique()}개 구), 분자 0인 동 {int((s.num == 0).sum())}개, "
          f"셀 단계 {s.cell_level.value_counts().to_dict()}")
    print(f"lease_rate(2025, 100건당): 중앙값 {s.lease_rate.median():.2f}, Q3 {s.lease_rate.quantile(.75):.2f}, "
          f"최대 {s.lease_rate.max():.2f} ({s.lease_rate.idxmax()})")

    val = target_validity(num, den)
    val.to_csv(OUTPUTS / "dong_target_validity.csv", index=False, encoding="utf-8-sig")
    print("\n타깃 타당성 (구 합산 vs HUG)\n" + val.round(3).to_string(index=False))

    fc = feature_corr(s)
    fc.to_csv(OUTPUTS / "dong_feature_corr.csv", index=False, encoding="utf-8-sig")
    print("\n피처별 단순 상관 (2025 동 비율, 구 군집 bootstrap 95% CI)\n" + fc.round(3).to_string(index=False))

    if not args.skip_judge:
        res, verdict, comp, diffs, oof = judge(s, args.n_perm)
        res.to_csv(OUTPUTS / "dong_rq2_judgment.csv", index=False, encoding="utf-8-sig")
        comp.to_csv(OUTPUTS / "dong_model_comparison.csv", index=False, encoding="utf-8-sig")
        diffs.to_csv(OUTPUTS / "dong_paired_diff.csv", index=False, encoding="utf-8-sig")
        oof.to_csv(OUTPUTS / "dong_oof.csv", index=False, encoding="utf-8-sig")
        print("\n모형 비교 (leave-one-gu-out)\n" + comp.drop(columns="features").round(3).to_string(index=False))
        print("\nM0 선형 대비 (구 군집 bootstrap 95% CI)\n" + diffs.round(3).to_string(index=False))
        print("\nRQ2 판정 표\n" + res.drop(columns=["features"]).round(3).to_string(index=False))
        print("\n판정\n" + verdict.to_string())

    lad, ldiff, lverdict = lag_ladder(d)
    lad.to_csv(OUTPUTS / "dong_lag_ladder.csv", index=False, encoding="utf-8-sig")
    ldiff.to_csv(OUTPUTS / "dong_lag_diff.csv", index=False, encoding="utf-8-sig")
    print("\n작년 값 사다리\n" + lad.drop(columns="features").round(3).to_string(index=False))
    print("\nP0 대비 (구 군집 bootstrap 95% CI, 쌍체 순열 구 통째)\n" + ldiff.round(3).to_string(index=False))
    print("\n판정: " + lverdict)

    tmp = temporal(t, r, num, den, d)
    if tmp is not None:
        tmp.to_csv(OUTPUTS / "dong_temporal.csv", index=False, encoding="utf-8-sig")
        print("\n시점 외 검증 (2026)\n" + tmp.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
