# QQQ Realized Volatility Forecasting — 01~07 + src/ 정밀 감사 리포트

> 분석 전용(read-only) 리포트. 코드 수정 없음. 모든 판단은 노트북 셀 소스·저장된 출력·src/ 코드를 직접 읽고 검증한 결과임.

---

## 요약 (TL;DR)

| # | 질문 | 판정 |
|---|------|------|
| 1 | 05/06/07 frame·fold 일관성 | **대체로 일관** — 셋 다 3,831행 + src/splits 0-indexed f00–f10. 단, **노트북 03 안에 fold 정의가 두 벌** 있어 03 cell 1의 출력된 날짜표는 05/06/07과 21일 어긋남 |
| 2 | Frozen HAR 일관성 | 07은 06을 정확히 재현(β 동일, refit 없음). 단 **f00 calibration이 HAR 학습구간 내부**(in-sample residual)이고, 마지막 21행 라벨이 f00 테스트 기간과 겹침(03의 embargo 규율과 불일치) |
| 3 | Look-ahead / leakage | **1건 실질적 leak 발견**: 07의 `sigma_garch`가 전체 3,831행에 적합된 GARCH 파라미터 사용. 그 외 h=21 라벨 관측 지연 문제(calibration window·ACI 피드백)로 "온라인 배포 가능"이라 말할 수 없음 |
| 4 | ACI 수학 | α_t 업데이트 식은 Gibbs & Candès (2021)와 **정확히 일치**. 단 α_t를 [1e-3, 0.999]로 clip한 것은 논문과 다르고, **하한 clip이 실제로 binding**(α_path.min()=0.001)이라 논문의 coverage 보장이 그대로 성립하지 않음. h=21 지연 피드백도 논문 세팅(즉시 err_t 관측)과 다름 |
| 5 | Stage 09 준비물 | 예측값 아티팩트 저장 전무, trial registry 없음, 전략(PnL) 레이어 없음 — PBO/DSR 계산에 필요한 3대 재료가 모두 미비 |
| 6 | 진학·커리어 가치 | 연구 감각(leakage 규율, persistence baseline, falsifiable thesis)은 상위권. CS 대학원용으론 **일반화 증거**(멀티에셋·합성 drift), 퀀트용으론 **경제적 유의성**(QLIKE, DM test, 전략 PnL)이 각각 부족 |

---

## 1. 단계 간 논리적 일관성 — 3,831-row frame과 0-indexed fold

### 1-a. 05/06/07은 같은 frame과 fold를 쓴다 ✅

세 노트북 모두 `assert len(work/df) == 3831` 가드가 있고, fold는 모두 [src/splits.py](src/splits.py)의 `walk_forward_splits(n, min_train=1008, test_size=252, embargo=21)` → **11 folds, 0-indexed f00–f10**:

| 노트북 | frame 구성 방식 | fold 소스 |
|--------|----------------|-----------|
| 05 | `df.dropna(subset=feat_all+tgt)` | `src.splits` |
| 06 | `df.dropna()` (전 컬럼) | `src.splits` |
| 07 | `df.iloc[21:-21]` (위치 기반 trim) | `src.splits` (파라미터 명시 호출) |

fold 경계도 세 노트북 출력에서 동일 확인: f05 test = 2020-02-07→2021-02-05 (COVID), **f07 = 2022-02-07→2023-02-07** (2022 약세장), f08 = 2023-02-08→2024-02-08. 06의 `MIN_TRAIN=1008`은 deployment stream[0]을 정확히 f00 test 시작(행 1008)에 정렬시키고, 07 cell 2 출력(`stream head ... 2015-02-05`)이 이를 교차 확인함.

### 1-b. 발견된 불일치 두 가지 ⚠️

**(i) 노트북 03에 split 정의가 두 벌 존재.**
- [03 cell 0]: 로컬 함수 `test_start = min_train + embargo` → f00 test = 행 [1029:1281], **2015-03-09→2016-03-07**
- [03 cell 4 이후]: `from src.splits import walk_forward_splits` → `test_start = min_train` → f00 test = 행 [1008:1260], **2015-02-05→2016-02-04**

실제 결과(HAR/GARCH RMSE 표, f07의 -25.1% "LOST")는 모두 src/splits 기준이라 05/06/07과 정합. 하지만 **03 cell 1에 출력된 fold 날짜표는 구버전(+21일 offset)이라 06/07의 날짜표와 어긋남**. 03의 날짜표를 인용해서 06의 drift 표와 교차 대조하면 21 거래일 밀린 잘못된 매핑이 나옴. 정리(구버전 셀 삭제 또는 재실행) 필요.

**(ii) frame 구성 방식이 3가지로 갈라져 있음.** dropna(subset)/dropna()/iloc[21:-21]이 지금은 우연히(NaN이 정확히 head21+tail21이라) 같은 3,831행을 주지만, 상류 데이터가 바뀌어 내부 NaN이 하나라도 생기면 07의 위치 기반 trim은 assert를 **통과하면서** NaN 행을 포함하게 됨(dropna 계열만 count가 달라져 걸림). [src/data.py](src/data.py)의 `load_work_frame()`이 이미 있으므로 세 노트북이 이것 하나만 쓰는 게 맞는 구조.

### 1-c. 논지 서사와 05 결과의 미묘한 간극 ⚠️

사용자 논지는 "f05/f08 = P(X) false alarm"인데, 05의 최종 binarization(PSI-only, cell 5) 결과는:

```
cov-shift POSITIVE set (>=2 feats): [3, 7, 8]   ← f05는 미포함!
```

- **f08**: P(X) fire + P(Y|X) 무반응(p=0.583) → false alarm 서사 성립 ✅
- **f03**: P(X) fire + P(Y|X) 무반응(p=0.573) → 언급 안 된 **추가 false alarm** (사실 논지를 더 강화하는 사례)
- **f05**: 최종 binary rule에서는 fire하지 않음. 05 cell 6이 이를 "ranking 논증"으로 전환(f05 intensity 4위/11, vix_psi=7.71)했지만, "f05가 P(X) 경보를 울렸다"는 서술은 **binary 기준으로는 부정확**. 리포트/논문화 시 "f05는 P(X) 강도 상위권(4/11, VIX PSI 7.7)이지만 P(Y|X) 무반응" 식으로 정확히 써야 함.

또 06의 ADWIN·DDM은 모든 δ에서 **drift 0건 검출**이었고, 실제 f07 검출은 ADWIN이 아니라 **calm-baseline 대비 one-sided z-test**(overlap-deflated n/21)가 해냄. "ADWIN 기반 concept drift detector"라고 요약하면 부정확 — 커밋 메시지("Concept drift detector (Drift detection, ADWIN)")와 실제 기여가 다름.

---

## 2. Frozen HAR (β 최초 1008행 고정)의 07 전반 일관성

### 일관된 부분 ✅
- [src/conformal.py:4-32](src/conformal.py#L4-L32) `frozen_har_residuals(X, y, n_train=1008)`: intercept+rv1/rv5/rv21, 최초 1008행 OLS(정규방정식 solve), 전 구간 forward-predict, **refit 없음**. 06 cell 0(lstsq)과 수학적으로 동일한 추정(β = [0.0048, 0.0258, 0.2010, 0.2949]로 출력 일치).
- 07의 split/normalized/ACI/Bayes 전부 이 residual stream 하나를 소비. β 재적재·재추정 지점 없음.
- Bayes 셀도 `N_CAL=1008`로 같은 frozen 정보 집합 사용.

### 새는(어긋나는) 지점 ⚠️

**(i) f00 calibration이 in-sample.** 07 cell 3의 `cal_idx = np.arange(te_idx[0]-252, te_idx[0])`은 f00에서 행 [756:1008) = **HAR 학습구간 내부**. calibration score가 in-sample residual(체계적으로 작음)이라 q가 과소 → f00 undercoverage(split 0.845)의 개연적 원인. f01 이후는 모두 out-of-sample이라 문제없음. f00을 빼거나 calibration 시작을 행 1008 이후로 미는 게 정석.

**(ii) frozen HAR 학습에 embargo 없음.** y_rv21[t]는 t+1..t+21의 수익률을 쓰므로, 학습 행 987–1007의 라벨은 f00 테스트 구간(행 1008–1028)과 겹침. 03은 이걸 embargo=21로 정확히 제거(f00 train이 [0:987])했는데 06/07의 frozen HAR은 1008행 전부 사용. 영향은 21/1008행이라 작지만, (a) 03의 규율과 불일치, (b) "행 1008 시점에 배포했다"는 서사가 엄밀히는 성립 안 함 — 그 시점엔 마지막 21개 라벨이 아직 관측 전.

---

## 3. Look-ahead bias / Data leakage

### 깨끗한 부분 ✅
- **02 feature/target 구성**: rv{w}는 trailing(t 포함), y_rv{h}는 forward(t+1..t+h) — 겹침 없음. splits의 embargo=21=h_max가 train 라벨→test 누수를 정확히 차단(03 cell 4의 assert가 이를 검증).
- **04 LSTM**: StandardScaler(x, y 모두) train-only fit, val은 train의 마지막 15%(시간순), test 라벨은 raw 유지. 깨끗함.
- **03 GARCH walk-forward**: 파라미터는 train까지만 적합, σ² recursion은 관측 수익률로 causal하게 roll. 깨끗함.
- **rolling_sigma** ([src/conformal.py:35-55](src/conformal.py#L35-L55)): 윈도우 [t-21, t-1], t 제외. 깨끗함.

### 문제 지점 🔴

**(i) `sigma_garch` 전체 표본 적합 — 07의 가장 실질적인 leak.**
[07 cell 2]:
```python
am = arch_model(df['rv1'].values * 100, vol='Garch', p=1, q=1, mean='Zero')
garch_res = am.fit(disp='off')                     # ← 3,831행 전체로 MLE
sigma_garch = garch_res.conditional_volatility / 100.0
```
recursion 자체는 causal이지만 **(ω, α, β) 파라미터가 2026년까지의 전체 데이터로 추정**됨. 이 σ가 `cov_norm_garch`와 **ACI**(cell 4가 `sig_g`를 전달) 양쪽에 들어가므로, 07의 대표 결과(ACI mean coverage 0.898)에 look-ahead가 섞임. `sigma_roll`은 깨끗하므로 ACI를 sig_r로 다시 돌리거나, GARCH를 최초 1008행 frozen(frozen HAR와 동일 철학)으로 적합해 재검증해야 함. normalized conformal의 보장 조건("sigma must be leakage-free" — src/conformal.py:92 주석)을 스스로 위반한 셈.

**(ii) h=21 라벨 관측 지연 — "leak-free" 주석이 과장.**
resid[t] = y_rv21[t] − ŷ[t]는 **t+21에야 관측** 가능. 따라서:
- split conformal의 calibration window(테스트 직전 252행)의 **마지막 21행 residual은 테스트 시작 시점에 아직 미관측**. 겹치는 RV 윈도우 때문에 마지막 cal score들과 첫 test score들이 상관되어 exchangeability도 추가로 훼손.
- ACI의 `cal = score[t-252:t]`도 매 t마다 최근 21개 미관측 score 포함, `err_t` 피드백도 실제로는 21일 지연인데 즉시 반영.

**offline 진단으로는 유효하나 "실시간 배포 가능한 시스템"으로 서술하면 과장**. 표준 수정: window와 업데이트를 h만큼 lag (`score[t-252-h:t-h]`, delayed ACI).

**(iii) 회고적(retrospective) 설계 요소들** — leak이라기보다 서술 정확성 문제:
- 06의 calm baseline p0가 f09·f10 등 **미래 fold를 포함해 pooled**됨(f07 검정 시점 기준 미래 정보). 가설검정 연구로는 유효하지만 "온라인 트리거"로는 불가 — 온라인 버전은 과거 calm fold만 써야 함.
- 05의 fire threshold(THR_MULT=3, calm fold 지정)도 전체 timeline의 03/04 결과를 보고 정함.
- 05의 PSI는 fold의 252행 test window 전체를 쓰므로, 트리거로서는 "fold 종료 후" 판정임(실시간이면 부분 윈도우).

**(iv) 통계적 소소한 점**: 06의 overlap deflation(n/21)은 방향은 맞는 보수적 보정이나, n_eff=12에서 정규근사 z-test는 아슬아슬함 — exact binomial이나 block bootstrap이 더 방어적. 07의 split conformal도 같은 이유로 calibration 252행의 유효 표본이 ~12라서 finite-sample 보장이 명목보다 훨씬 느슨함.

---

## 4. ACI (Gibbs & Candès 2021) 수학적 검증

구현: [src/conformal.py:114-164](src/conformal.py#L114-L164) `aci_stream_coverage`.

### 논문과 일치 ✅
- **업데이트 식**: `alpha_t = alpha_t + gamma * (alpha - err)`, `err_t = 1[미커버]` — 논문 식 (2) `α_{t+1} = α_t + γ(α − err_t)` 와 **정확히 일치**. γ=0.01은 논문 실험 범위 내의 표준값.
- **스트림 연속성**: fold 경계에서 α_t 리셋 없음(all_test가 행 1008–3779로 연속 타일). "true online" 주석대로 동작.
- **quantile 추정기 자유도**: 논문의 보장은 quantile 추정기 Q̂의 유효성과 무관하게 업데이트 동역학에서 나오므로, trailing 252 window + `np.quantile`(선형보간, finite-sample 보정 없음)을 쓴 것 자체는 이론과 모순 아님. (split conformal 쪽은 `ceil((n+1)(1-α))` 정식 rank를 쓰고 있어 각자 맥락에 맞음.)
- **Exchangeability**: ACI는 exchangeability를 요구하지 않음(그게 논문의 요지) — 여기 적용은 개념적으로 올바른 use case. 반면 같은 노트북의 split/normalized conformal은 exchangeability를 요구하는데 h21 잔차의 21일 자기상관이 이를 위반(§3-iv) — 두 방법의 보장 수준 차이를 리포트에 명시할 것.

### 논문과 다른 부분 ⚠️
1. **α_t clipping [1e-3, 0.999]**: 논문은 α_t가 (0,1) 밖으로 나가는 것을 허용(α_t≤0 → 전 구간 interval, α_t≥1 → 공집합)하고, 장기 coverage 보장 |T⁻¹Σerr_t − α| ≤ (max(α₁,1−α₁)+γ)/(Tγ)는 **unclipped 동역학**에 의존. 출력 `alpha_t range [0.001, 0.293]`에서 **하한 clip이 실제로 binding** — 미커버 연속 구간에서 논문이라면 interval을 사실상 무한대로 벌렸을 것을, 여기선 252-window의 ~max score로 상한을 둠. 결과적으로 mean coverage 0.898은 경험적으로 훌륭하지만 **G&C 정리의 보호를 받는 수치가 아님**. (실무적 타협으로 흔하나, 논문 재현 주장 시 명시 필요.)
2. **지연 피드백**: 논문 세팅은 t 시점 예측 → err_t 즉시 관측 → t+1 업데이트. h=21에서는 err_t가 t+21에 관측되므로 즉시 업데이트는 불가능(§3-ii). 현재 구현은 사실상 "오라클 피드백 ACI".
3. (사소) score에 NaN 제거 후 quantile — 초기 sigma_roll NaN 대비 방어 코드로, ACI 실행은 sig_g만 쓰므로 실질 영향 없음.

**결론**: 업데이트 식 자체는 교과서적으로 정확. 다만 (clip binding + 전량 적합 σ_GARCH + 지연 피드백) 3개를 고치기 전에는 "Gibbs & Candès 보장이 성립하는 배포 가능 ACI"라고 쓰면 안 되고, "offline 재현 + 실무적 변형"으로 서술해야 함.

---

## 5. Stage 09 (PBO / Deflated Sharpe) 사전 준비 — 빠진 것

README 로드맵: `08_adaptive_retrain`(drift-triggered retraining) → `09_eval_report`(PBO, deflated Sharpe). 현재 08·09 노트북은 없고, repo 전체에서 PBO/DSR/CSCV/backtest/transaction cost 관련 코드는 **전무**. 09를 위해 지금부터 준비해야 할 것:

1. **예측 아티팩트 저장 규율 (가장 시급)** — 현재 03(HAR/GARCH), 04(LSTM), 07(interval, α_path)은 결과를 **print만 하고 저장하지 않음**(저장물은 covariate_shift.csv, concept_drift.csv 단 2개, 그것도 Colab Drive에만). 09는 모델·fold 재실행 없이 계산 가능해야 하므로 표준 스키마의 예측 저장소 필요: `(date, model, horizon, y, yhat, lo, hi, fold)` per-day parquet/csv.
2. **Trial registry** — PBO(CSCV, Bailey et al. 2016)는 "시도한 모든 설정" M개의 성과 행렬 (M×T)이 입력. 지금까지의 선택들(05에서 KS 버리고 PSI-only 채택, THR_MULT=3, 06의 h21 채택·CALM fold 지정·OVERLAP=21, 07의 γ=0.01·cal_window=252, 04의 LSTM 하이퍼파라미터/시드)이 **모두 기록되지 않은 trial**임. 각 설정 변형의 per-day 손실/수익 시계열을 남기는 실험 로그가 없으면 PBO는 계산 자체가 불가능.
3. **전략(PnL) 레이어** — deflated Sharpe는 Sharpe의 함수이므로 수익률 시계열이 필요한데, 현재는 예측→포지션 매핑이 없음. 최소 사양: vol-targeting(포지션 = σ_target/σ̂) 또는 vol 예측 기반 리스크 스케일링 + 거래비용 모델. 이게 없으면 09는 "deflated Sharpe" 대신 예측력 지표의 다중검정 보정(예: DM test + Bonferroni/White reality check)으로 대체 설계해야 함.
4. **DSR 입력 모멘트** — trial 수 N, trial 간 Sharpe 분산, 수익률 skew/kurtosis, track length. (3)과 (2)가 있어야 산출 가능.
5. **선택/검증 fold 분리 선언** — 지금까지 f07이 분석의 주인공으로 반복 사용되어 사실상 "선택에 오염된 fold". 09에서 f07을 다시 성과 검증에 쓰면 자기참조가 됨. 어떤 fold가 선택용이고 어떤 fold가 holdout인지 프로토콜 문서화 필요.
6. **08 자체** — 09의 비교 대상(retrain-always vs retrain-on-P(X) vs retrain-on-P(Y|X) vs never)이 08에서 나와야 함. 06의 회고적 z-test를 **온라인 버전**(과거 calm fold만으로 p0 추정)으로 바꾸는 작업이 08의 선행 과제.
7. **재현성 인프라** — requirements.txt/조건 고정 없음, tests 없음, Colab 경로(`/content/drive/...`) 하드코딩, 03의 이중 split 정의(§1-b). 09가 "최종 리포트"라면 이 상태로는 제3자가 숫자 하나도 재현 못 함.

---

## 6. CS 대학원 / 퀀트 커리어 관점 평가

### 지금 상태의 강점 (솔직하게, 실제로 상위권인 부분)
- **falsifiable한 단일 논지**("P(X)는 necessary but not sufficient")를 세우고 f05/f07/f08로 검증한 구조 — 대부분의 학생 포트폴리오("LSTM으로 주가 예측")와 격이 다름.
- walk-forward + embargo + overlap deflation + train-only scaling 등 **leakage 규율이 몸에 배어 있음**을 코드로 증명. 퀀트 면접에서 가장 먼저 찔러보는 지점을 선제 방어.
- ADWIN/DDM/ACI/NIG-Bayes를 라이브러리 호출이 아니라 **논문 보고 직접 구현** — 연구 역량 신호.
- persistence baseline을 이기는지부터 확인하는 태도(f07에서 모든 모델이 지는 걸 발견한 것 자체가 이 프로젝트의 발견).

### CS 대학원(ML 계열) 관점 — 지금은 "좋은 시작", 보완하면 "연구 실적"
현재의 약점은 **증거의 폭**: 자산 1개(QQQ), fold 11개, true drift 이벤트 **1개**(f07). 이벤트 1개로는 detector의 power/false-alarm rate을 주장할 수 없음. 보완:
1. **멀티에셋 패널** (SPY, IWM, 섹터 ETF, 국채, 원자재, 크립토) — drift 이벤트 수를 수십 개로 늘려 검출률·오경보율을 통계로 제시.
2. **합성 drift 주입 실험** — β를 인위적으로 바꾼 시뮬레이션으로 ground truth를 통제한 power 분석. distribution-shift 논문의 표준 구성이고 CS 심사자가 기대하는 것.
3. **경쟁 방법 비교** — BOCPD(Bayesian online change-point), online learning(재학습 always vs trigger), retrain 비용-편익 곡선.
4. **작성·공개** — 4~8쪽 workshop 포맷(예: NeurIPS/ICML의 distribution shift·conformal 워크숍) 또는 arXiv 테크리포트로 정리. SOP에 "repo + preprint" 링크가 있는 것과 없는 것의 차이가 큼. ACI·conformal은 현재 CS에서 매우 활발한 주제라 접점이 좋음.

### 퀀트 커리어 관점 — 방향은 정확, "돈"과 "업계 표준 지표"가 빠짐
1. **RV 정의가 업계 기준으로 약함**: 현재 RV는 일간 종가 수익률 제곱 기반(y_rv1은 사실상 |다음날 수익률|) — 노이즈가 큼. 5분봉 realized variance가 표준이고, 데이터가 없으면 최소 Parkinson/Garman-Klass 레인지 추정치로 업그레이드. HAR-RV 원전(Corsi 2009)은 log RV에 적합하는 것도 참고.
2. **손실함수**: vol 예측 평가의 표준은 RMSE가 아니라 **QLIKE**(robust to noise in proxy, Patton 2011). 지금 결과를 QLIKE로 다시 뽑고 **Diebold-Mariano 검정** 추가.
3. **경제적 유의성**: §5의 전략 레이어(vol targeting + 거래비용) + "drift 트리거로 재학습한 전략 vs 안 한 전략"의 PnL 차이 — 면접에서 "그래서 그 detector가 돈을 지켰나?"에 답할 수 있어야 함. 이게 stage 08/09이므로 **로드맵을 끝까지 완주하는 것 자체가 최대 보완책**.
4. **면접 서사**: "f07에서 HAR가 persistence에 25% 지는 걸 발견 → P(X) 경보만으론 f03/f08 오경보를 못 거름 → P(Y|X) 검정으로 f07만 분리 → 재학습 정책의 PnL 개선으로 연결"은 시니어 퀀트 앞에서도 통하는 스토리. 단, §1-c·§3의 서술 정확성 문제(f05 binary 미발화, 회고적 baseline, σ_GARCH leak)를 스스로 먼저 말할 수 있어야 신뢰를 얻음 — 지적당하면 감점, 선제 언급하면 가점.
5. **위생**: requirements 고정, 로컬 재현 경로, 03 구버전 셀 제거, `load_work_frame()` 일원화, 핵심 함수(splits, psi, adwin, aci) 단위 테스트. 퀀트 팀은 코드 위생을 연구 역량만큼 봄.

### 종합
- **도움이 되는가?** 된다 — 단, "완성된 06까지" 상태로는 아직 절반. 08(온라인 트리거 + 재학습 정책)과 09(PBO/DSR 또는 그에 준하는 다중검정 보정 평가)를 완주하고, σ_GARCH leak·f00 calibration·ACI clip 이슈를 고친 뒤 짧은 preprint로 묶으면 CS 대학원 SOP와 퀀트 지원서 양쪽에서 실질적 무기가 됨.
- 우선순위 제안: ① σ_GARCH frozen 재적합 + ACI 재실행(결과 방어) → ② 예측 아티팩트 저장 체계(09 준비) → ③ 08 온라인화 → ④ QLIKE/DM 추가 → ⑤ 멀티에셋 확장 → ⑥ write-up.
