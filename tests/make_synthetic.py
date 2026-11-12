import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from config import RAW, EXTERNAL, SEOUL_GU

rng = np.random.default_rng(0)
months = pd.period_range("2021-07", "2025-09", freq="M")
latent = {c: rng.normal() for c in SEOUL_GU}
size = {c: rng.integers(40, 250) for c in SEOUL_GU}

(RAW / "trade").mkdir(parents=True, exist_ok=True)
(RAW / "rent").mkdir(parents=True, exist_ok=True)
EXTERNAL.mkdir(parents=True, exist_ok=True)

for code in SEOUL_GU:
    base_p = rng.uniform(450, 900)
    jr = 0.72 + 0.06 * latent[code]
    dongs = [f"합성{i}동" for i in range(6)]
    bldgs = [(rng.choice(dongs), f"{rng.integers(1, 900)}", f"빌라{i}", int(rng.integers(1985, 2023)))
             for i in range(120)]
    for k, p in enumerate(months):
        n_t, n_r = max(3, int(size[code] * 0.15)), size[code]
        last_day = 1 if (k == len(months) - 1) else p.days_in_month
        def rows(n, kind):
            out = []
            for _ in range(n):
                d, j, b, by = bldgs[rng.integers(len(bldgs))]
                area = round(float(rng.choice([25, 33, 40, 48, 59, 75])) * rng.uniform(0.98, 1.02), 2)
                trend = 1 + 0.002 * k
                row = {"sggCd": code, "umdNm": d, "jibun": j, "mhouseNm": b, "excluUseAr": area,
                       "floor": int(rng.integers(1, 6)), "buildYear": by, "dealYear": p.year,
                       "dealMonth": p.month, "dealDay": int(rng.integers(1, last_day + 1))}
                age_adj = 1 + 0.004 * (by - 2005)
                if kind == "trade":
                    price = base_p * trend * age_adj * area * rng.lognormal(0, 0.12)
                    row.update(dealAmount=f"{int(price):,}", landAr=round(area * 0.4, 2),
                               cdealDay="" if rng.random() > 0.05 else "23.05.01", dealingGbn="중개거래")
                else:
                    jeonse = rng.random() > 0.6
                    dep = base_p * trend * age_adj * area * jr * rng.lognormal(0, 0.10)
                    y0 = p.year % 100
                    row.update(deposit=f"{int(dep if jeonse else dep * 0.1):,}",
                               monthlyRent=0 if jeonse else int(rng.integers(40, 90)),
                               contractTerm=f"{y0:02d}.{p.month:02d}~{y0 + 2:02d}.{p.month:02d}",
                               contractType=rng.choice(["신규", "갱신"], p=[0.7, 0.3]),
                               useRRRight="", preDeposit="", preMonthlyRent="")
                out.append(row)
            df = pd.DataFrame(out)
            if len(df) > 3:
                df = pd.concat([df, df.iloc[3:5]])
                df.iloc[0, df.columns.get_loc("excluUseAr")] = 500.0
                df.iloc[1, df.columns.get_loc("buildYear")] = 0
                col = "dealAmount" if kind == "trade" else "deposit"
                df.iloc[2, df.columns.get_loc(col)] = "50"
            return df
        rows(n_t, "trade").to_csv(RAW / "trade" / f"{code}_{p.strftime('%Y%m')}.csv", index=False)
        rows(n_r, "rent").to_csv(RAW / "rent" / f"{code}_{p.strftime('%Y%m')}.csv", index=False)

rate = np.exp(0.0 + 0.5 * np.array([latent[c] for c in SEOUL_GU]) + rng.normal(0, 0.35, 25))
pd.DataFrame({"gu": list(SEOUL_GU.values()), "accident_rate": rate.round(2),
              "accident_count": rng.integers(1, 80, 25), "accident_amount": rng.uniform(5, 180, 25).round(1)}) \
    .to_csv(EXTERNAL / "hug_target.csv", index=False, encoding="utf-8-sig")
print("합성 데이터 생성:", RAW)
