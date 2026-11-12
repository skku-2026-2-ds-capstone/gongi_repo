import os
from pathlib import Path
import pandas as pd

ROOT = Path(os.environ.get("GONGI_DATA_ROOT", Path(__file__).resolve().parents[1]))
RAW = ROOT / "data" / "raw"
INTERIM = ROOT / "data" / "interim"
PROCESSED = ROOT / "data" / "processed"
EXTERNAL = ROOT / "data" / "external"
OUTPUTS = ROOT / "outputs"

_env = Path(__file__).resolve().parents[1] / ".env"
if _env.exists():
    for _line in _env.read_text(encoding="utf-8-sig").splitlines():
        if "=" in _line and not _line.lstrip().startswith("#"):
            _k, _v = _line.split("=", 1)
            os.environ.setdefault(_k.strip(), _v.strip().strip('"').strip("'"))
if "API_KEY" in os.environ:
    os.environ.setdefault("DATA_GO_KR_KEY", os.environ["API_KEY"])

TARGET_START = pd.Timestamp("2025-06-01")
TARGET_END = pd.Timestamp("2025-08-31")
HIGH_RISK_THRESHOLD = 1.8
LOW_RISK_THRESHOLD = 0.6

WINDOWS = {
    "main": (pd.Timestamp("2023-01-01"), pd.Timestamp("2023-12-31")),
    "prev": (pd.Timestamp("2022-01-01"), pd.Timestamp("2022-12-31")),
    "aux": (pd.Timestamp("2021-07-01"), pd.Timestamp("2023-06-30")),
    "long": (pd.Timestamp("2020-01-01"), pd.Timestamp("2023-12-31")),
    "prop": (pd.Timestamp("2021-07-01"), pd.Timestamp("2023-12-31")),
}
INPUT_CUTOFF = max(end for _, end in WINDOWS.values())
assert INPUT_CUTOFF < TARGET_START, "입력 창이 타깃 기간과 겹친다"

RENT_REPORTING_BREAK = pd.Timestamp("2021-06-01")

SEOUL_GU = {
    "11110": "종로구", "11140": "중구", "11170": "용산구", "11200": "성동구",
    "11215": "광진구", "11230": "동대문구", "11260": "중랑구", "11290": "성북구",
    "11305": "강북구", "11320": "도봉구", "11350": "노원구", "11380": "은평구",
    "11410": "서대문구", "11440": "마포구", "11470": "양천구", "11500": "강서구",
    "11530": "구로구", "11545": "금천구", "11560": "영등포구", "11590": "동작구",
    "11620": "관악구", "11650": "서초구", "11680": "강남구", "11710": "송파구",
    "11740": "강동구",
}
GU_TO_CODE = {v: k for k, v in SEOUL_GU.items()}

HOUSING_TYPES = {"rh": "연립다세대", "apt": "아파트", "offi": "오피스텔", "sh": "단독다가구"}
JR_TYPES = ["rh", "apt", "offi"]


def dataset_name(htype: str, kind: str) -> str:
    return kind if htype == "rh" else f"{htype}_{kind}"
DEEP_DIVE_GU = ["강서구", "중랑구", "은평구", "동작구"]

AREA_MIN, AREA_MAX = 10.0, 200.0
MAD_K = 5.0
MIN_CELL_N = 5
MIN_GU_N = 30
AREA_BINS = [0, 30, 45, 60, 85, 1000]
AREA_LABELS = ["~30", "30-45", "45-60", "60-85", "85~"]
BUILD_BINS = [0, 1990, 2000, 2010, 2015, 2100]
BUILD_LABELS = ["~1989", "1990s", "2000s", "2010-14", "2015~"]
MATCH_AREA_TOL = 0.05
MATCH_MAX_GAP_DAYS = 365
