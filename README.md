# 공.이. 데이터 파이프라인 (수집 → 정제 → 피처 → M0)

## 처음 실행하는 팀원용 준비

1. Python 3.10 이상에서 패키지 설치: `pip install -r requirements.txt`
2. 공공데이터포털(data.go.kr)에서 아래 두 API를 **활용신청**하고, 마이페이지의 **Decoding 키**를 복사한다.
   - 국토교통부_연립다세대 매매 실거래가 자료 (15126467)
   - 국토교통부_연립다세대 전월세 실거래가 자료 (15126473)
3. 프로젝트 루트에 `.env` 파일을 만들고 한 줄만 적는다. `config.py`가 자동으로 읽는다.
   ```
   API_KEY=발급받은_디코딩_키
   ```
   `.env`는 GitHub에 올리지 않는다 (`.gitignore`에 `.env`, `data/` 추가).
4. HUG 사고현황 엑셀을 `data/hug_target.xlsx`에 둔다.

수집을 건너뛰고 코드만 확인하려면 맨 아래 **합성 데이터 점검**을 쓴다.

## 실행 순서

Windows PowerShell 기준. 각 줄을 프로젝트 루트에서 실행한다. 출력 한글이 깨지면 먼저 `$env:PYTHONIOENCODING="utf-8"`.

```powershell
# 1) 수집: 25개 구 M0용 (구×월 1,500회 호출, 약 15분). 이미 받은 월은 건너뜀
python src/collect_molit.py --kind trade --all-gu --start 2021-07 --end 2023-12
python src/collect_molit.py --kind rent  --all-gu --start 2021-07 --end 2023-12
#    2단계 4개 구 장기분
python src/collect_molit.py --kind trade --gu 강서구 중랑구 은평구 동작구 --start 2016-01 --end 2025-12
python src/collect_molit.py --kind rent  --gu 강서구 중랑구 은평구 동작구 --start 2016-01 --end 2025-12
#    공개시스템에서 받은 CSV는 data/raw/csv_export/ 에 넣으면 같이 읽힘 (파일명에 '매매' 또는 '전월세')

# 2) HUG 타깃: data/hug_target.xlsx → data/external/hug_target.csv (서울 25개 구, 금액은 억원)
python src/prep_hug.py

# 3) 정제 → 피처 → 베이스라인
python src/prep_population.py     # 청년 비율 (없으면 features.py가 건너뜀)
python src/clean.py
python src/features.py
python src/baseline.py            # 순열 1,000회라 몇 분 걸림. 빠른 점검은 --n-perm 200

# 4) 구 단위 모형 비교, 민감도, 시점 외 검증 (2024 입력 → 2026 사고율)
python src/models.py
python src/sensitivity.py
python src/sensitivity.py --rq2-only
python src/prep_hug.py --xlsx data/hug_target_2026.xlsx --out hug_target_2026.csv
python src/temporal.py

# 5) 법정동 단위 대리 타깃(임차권등기명령), RQ1 4개 구 비교, 그림
python src/prep_registry.py
python src/dong.py                # 판정·순열에 몇 분. 빠른 점검은 --skip-judge
python src/rq1.py
python src/make_figures.py        # outputs/*.csv 에서만 그림을 그린다 → outputs/figures/
```

모든 재표본(bootstrap, 순열, RF·GBM)은 seed 42로 고정한다 (`models.py`·`sensitivity.py`의 `SEED`, `baseline.py`·`temporal.py`의 `default_rng(42)`).

macOS/Linux bash에서는 같은 명령을 그대로 쓰고, 인코딩 설정만 `export PYTHONIOENCODING=utf-8`로 바꾼다.

| 단계 | 산출물 |
|---|---|
| 수집 | `data/raw/{trade,rent}/{LAWD_CD}_{YYYYMM}.csv`, `outputs/collect_log_*.csv` |
| 타깃 | `data/external/hug_target.csv` (gu, lawd_cd, accident_rate(%), accident_count, accident_amount(억원)) |
| 정제 | `data/interim/*_clean.csv`, `outputs/cleaning_log.csv`, `outputs/missing_report.csv` |
| 피처 | `data/processed/gu_features.csv`(25구), `gu_halfyear_panel.csv`(구×반기), `outputs/feature_dictionary.csv` |
| M0 | `outputs/baseline_results.csv`, `outputs/baseline_oof.csv` |

기간·시차·임계값·정제 파라미터는 전부 `src/config.py` 한 곳에서만 바꾼다.
누수 가드: 입력 창(~2023.12) 밖 계약이 피처에 섞이거나 피처 창이 타깃 시작(2025.06) 이후면 코드가 멈춘다.

변수 사전(`outputs/feature_dictionary.csv`)은 `features.py`의 `FEATURE_DICT`에서 만들어진다. 정의·단위·입력 창·방향 가설·근거·모형 역할·예측 시점에 알 수 있는지를 함께 적는다. 고칠 때는 CSV가 아니라 `FEATURE_DICT`를 고친다.

**합성 데이터 점검** (실데이터 없이 코드만 확인):
```powershell
$env:GONGI_DATA_ROOT="$env:TEMP\gongi_synth"; python tests/make_synthetic.py
python src/clean.py; python src/features.py; python src/baseline.py --n-perm 200
Remove-Item Env:GONGI_DATA_ROOT     # 끝나면 실데이터 경로로 복귀
```
