import argparse
import os
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import pandas as pd
import requests

from config import RAW, OUTPUTS, SEOUL_GU, GU_TO_CODE, dataset_name

BASE_URL = "https://apis.data.go.kr/1613000"
SERVICES = {"rh": "RH", "apt": "Apt", "sh": "SH", "offi": "Offi"}
KIND_SUFFIX = {"trade": "Trade", "rent": "Rent"}


def endpoint(htype: str, kind: str) -> str:
    svc = f"RTMSDataSvc{SERVICES[htype]}{KIND_SUFFIX[kind]}"
    return f"{BASE_URL}/{svc}/get{svc}"
NUM_ROWS = 1000
MAX_RETRY = 3


def fetch_month(htype: str, kind: str, lawd: str, ym: str, key: str):
    rows, page, total = [], 1, None
    while True:
        params = {"serviceKey": key, "LAWD_CD": lawd, "DEAL_YMD": ym,
                  "pageNo": page, "numOfRows": NUM_ROWS}
        r = requests.get(endpoint(htype, kind), params=params, timeout=30)
        r.raise_for_status()
        root = ET.fromstring(r.content)
        code = root.findtext(".//resultCode")
        if code not in ("00", "000"):
            raise RuntimeError(f"API resultCode={code} {root.findtext('.//resultMsg')}")
        total = int(root.findtext(".//totalCount") or 0)
        for item in root.iter("item"):
            rows.append({c.tag: (c.text or "").strip() for c in item})
        if page * NUM_ROWS >= total:
            break
        page += 1
    return pd.DataFrame(rows), total


def month_range(start: str, end: str):
    return [p.strftime("%Y%m") for p in pd.period_range(start, end, freq="M")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--type", dest="htype", choices=list(SERVICES), default="rh")
    ap.add_argument("--kind", choices=list(KIND_SUFFIX), required=True)
    ap.add_argument("--gu", nargs="*", default=[])
    ap.add_argument("--all-gu", action="store_true")
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--sleep", type=float, default=0.2)
    args = ap.parse_args()

    key = os.environ.get("DATA_GO_KR_KEY")
    if not key:
        raise SystemExit("환경변수 DATA_GO_KR_KEY 가 없습니다.")
    codes = list(SEOUL_GU) if args.all_gu else [GU_TO_CODE[g] for g in args.gu]

    name = dataset_name(args.htype, args.kind)
    outdir = RAW / name
    outdir.mkdir(parents=True, exist_ok=True)
    log_path = OUTPUTS / f"collect_log_{name}.csv"
    log = pd.read_csv(log_path, dtype=str) if log_path.exists() else pd.DataFrame()

    new_logs = []
    for lawd in codes:
        for ym in month_range(args.start, args.end):
            f = outdir / f"{lawd}_{ym}.csv"
            if f.exists():
                continue
            status, total, n, err = "ok", None, 0, ""
            for attempt in range(1, MAX_RETRY + 1):
                try:
                    df, total = fetch_month(args.htype, args.kind, lawd, ym, key)
                    n = len(df)
                    df.to_csv(f, index=False, encoding="utf-8-sig")
                    status = "ok" if n == total else "count_mismatch"
                    break
                except Exception as e:
                    status, err = "failed", str(e).replace(key, "***")[:200]
                    time.sleep(2 * attempt)
            new_logs.append({"type": args.htype, "kind": args.kind, "lawd_cd": lawd, "gu": SEOUL_GU[lawd],
                             "ym": ym, "total_count": total, "rows": n,
                             "status": status, "attempts": attempt, "error": err,
                             "fetched_at": pd.Timestamp.now().isoformat(timespec="seconds")})
            time.sleep(args.sleep)
        print(f"{SEOUL_GU[lawd]} 완료")

    log = pd.concat([log, pd.DataFrame(new_logs)], ignore_index=True)
    log.to_csv(log_path, index=False, encoding="utf-8-sig")
    failed = (pd.DataFrame(new_logs).get("status", pd.Series(dtype=str)) != "ok").sum()
    print(f"수집 로그 저장: {log_path} (이번 실행 실패·불일치 {failed}건)")


if __name__ == "__main__":
    main()
