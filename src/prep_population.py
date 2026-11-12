import argparse
import os

import pandas as pd
import requests

from config import EXTERNAL, SEOUL_GU

URL = "https://apis.data.go.kr/1741000/stdgSexdAgePpltn/selectStdgSexdAgePpltn"


def fetch(ym: str, key: str) -> pd.DataFrame:
    params = {"serviceKey": key, "stdgCd": "1100000000", "srchFrYm": ym, "srchToYm": ym,
              "lv": "2", "regSeCd": "1", "type": "JSON", "numOfRows": "100", "pageNo": "1"}
    r = requests.get(URL, params=params, timeout=30)
    r.raise_for_status()
    body = r.json()["Response"]
    if body["head"]["resultCode"] not in ("0", "00"):
        raise RuntimeError(body["head"]["resultMsg"])
    items = pd.DataFrame(body["items"]["item"])
    items["lawd_cd"] = items["stdgCd"].str[:5]
    d = items[items["lawd_cd"].isin(SEOUL_GU)].copy()
    if len(d) != 25:
        raise SystemExit(f"{ym}: 서울 25개 구가 아님 ({len(d)}행)")
    num = lambda c: pd.to_numeric(d[c], errors="raise")
    d["total"] = num("totNmprCnt")
    d["age20_39"] = sum(num(f"{s}{a}AgeNmprCnt") for s in ("male", "feml") for a in (20, 30))
    d["youth_share"] = d["age20_39"] / d["total"]
    d["gu"] = d["lawd_cd"].map(SEOUL_GU)
    d["ym"] = ym
    return d[["gu", "lawd_cd", "ym", "total", "age20_39", "youth_share"]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ym", nargs="+", default=["202312", "202412"])
    args = ap.parse_args()
    key = os.environ.get("DATA_GO_KR_KEY")
    if not key:
        raise SystemExit("환경변수 DATA_GO_KR_KEY 가 없습니다.")
    out = pd.concat([fetch(ym, key) for ym in args.ym], ignore_index=True)
    EXTERNAL.mkdir(parents=True, exist_ok=True)
    path = EXTERNAL / "population.csv"
    out.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"저장: {path}")
    print(out.pivot(index="gu", columns="ym", values="youth_share").round(3).to_string())


if __name__ == "__main__":
    main()
