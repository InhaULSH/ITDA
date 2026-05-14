# ITDA_KJM YelpZip Review Abuse GNN

## 1. 현재 실행 기준

현재 프로젝트에서 실제 실행 기준으로 사용하는 모델은 **기존 `topk15 window60 SAGE` 하나**이다.

```text
data/graph_relflag_edge_t15_w60_thr075/graph_rur_custom2.pt
experiments/relflag_edge_t15_w60_thr075_sage_inverse_seed42/
```

방향성 피처 적용 `topk15 window60 SAGE`의 리소스와 코드는 보존한다. 다만 이 모델은 현재 실행 모델이 아니라, 추후 설명 가능성 강화를 위해 다시 검토할 수 있는 후보이다. 기본 실행 스크립트와 실행 방법은 기존 `topk15 window60 SAGE`만 대상으로 한다.

LightGBM node-only 모델과 stacking 보정 모델은 현재 범위에서 제거했다.

## 2. 필수 조건 준수 요약

현재 실행 모델은 대회 필수 조건을 다음과 같이 만족한다.

| 필수 조건 | 현재 모델의 반영 |
|---|---|
| 샘플링 후 분할 | `data/sampled_relative_flags_q75_m2`를 먼저 생성한 뒤 `data/splits_relative_flags_q75_m2` mask를 생성 |
| 무작위 추출 금지 | 단순 무작위 노드 추출이 아니라 `prod_id × week` 단위의 밀도 gate와 상대 변화량 flag로 샘플링 |
| 1만~5만 노드 | 최종 샘플 노드 수 26,701개 |
| train/valid 80%, test 20% | train 64%, valid 16%, test 20% |
| 재현성 | split summary에 `method=temporal`, `random_state=42`, ratio 기록 |
| 라벨 변환 | 원본 `label=-1`을 `is_fake=1`, `label=1`을 `is_fake=0`으로 변환 |
| 리뷰 단위 노드 | 원본 리뷰 1행을 리뷰 노드 1개로 유지 |
| GNN 핵심 사용 | plain GraphSAGE 사용 |
| 기본 relation | R-U-R, edge type 0, 11,038 directed edges |
| 커스텀 relation | weak product rating shock, edge type 2, 4,266 directed edges |

주의할 점은 R-U-R 엣지가 현재 `rur_temporal=false`로 생성되어 있다는 것이다. 필수 조건 자체는 동일 사용자 기반 R-U-R을 요구할 뿐 방향성을 요구하지 않으므로 미준수는 아니다. 다만 보고서에서 “과거에서 현재 방향으로만 연결한다”고 표현하면 실제 산출물과 맞지 않으므로, 현재 모델 설명에서는 “동일 사용자 리뷰를 날짜 기준 가까운 리뷰 중심으로 연결한다”고 서술해야 한다.

## 3. 빠르게 결과 확인하기

재학습 없이 저장된 결과를 보려면 다음 파일을 확인한다.

```text
experiments/relflag_edge_t15_w60_thr075_sage_inverse_seed42/metrics.json
experiments/relflag_edge_t15_w60_thr075_sage_inverse_seed42/predictions_all.csv
experiments/relflag_edge_t15_w60_thr075_sage_inverse_seed42/prediction_test.csv
```

현재 test 성능은 다음과 같다.

| 모델 | PR-AUC | ROC-AUC | Macro F1 | Precision | Recall | Accuracy |
|---|---:|---:|---:|---:|---:|---:|
| 기존 topk15 window60 SAGE | 0.4901 | 0.7851 | 0.6793 | 0.4956 | 0.4965 | 0.7839 |

## 4. 환경 확인

현재 학습은 Intel XPU 사용을 기준으로 진행했다. 실행 전 PowerShell에서 다음 명령으로 XPU 사용 가능 여부를 확인한다.

```powershell
C:\Users\LSH\Downloads\ITDA\.venv\Scripts\python.exe -c "import torch; print(torch.__version__); print(hasattr(torch, 'xpu') and torch.xpu.is_available())"
```

마지막 출력이 `True`이면 `--device xpu`로 학습할 수 있다. 단순 smoke test만 필요할 때는 `--device cpu`로 바꿀 수 있지만, 최종 재현 기준은 XPU이다.

## 5. 현재 실행 모델 학습

이미 `metrics.json`이 있으면 기본적으로 재학습하지 않는다. 현재 저장된 결과를 확인하면서 요약 파일만 만들려면 다음 명령을 사용한다.

```powershell
C:\Users\LSH\Downloads\ITDA\.venv\Scripts\python.exe .\RunExperiments.py --device xpu
```

실제로 다시 학습하려면 `--force`를 붙인다.

```powershell
C:\Users\LSH\Downloads\ITDA\.venv\Scripts\python.exe .\RunExperiments.py --device xpu --force
```

실행 후 다음 요약 파일이 생성된다.

```text
experiments/active_sage_model_summary.csv
experiments/active_sage_model_summary.json
```

## 6. 단독 학습 명령

`RunExperiments.py`를 거치지 않고 직접 학습하려면 다음 명령을 사용한다.

```powershell
C:\Users\LSH\Downloads\ITDA\.venv\Scripts\python.exe .\TrainGNN.py `
  --graph-path .\data\graph_relflag_edge_t15_w60_thr075\graph_rur_custom2.pt `
  --model sage `
  --output-dir .\experiments\relflag_edge_t15_w60_thr075_sage_inverse_seed42 `
  --hidden-dim 128 `
  --num-layers 2 `
  --dropout 0.5 `
  --lr 0.001 `
  --weight-decay 0.0001 `
  --epochs 200 `
  --patience 30 `
  --seed 42 `
  --device xpu `
  --class-weight `
  --mask-mode split `
  --early-stop-metric valid_pr_auc `
  --threshold-strategy prevalence_constrained_macro_f1
```

## 7. 전체 파이프라인 재생성 순서

기존 산출물이 있으면 이 절을 실행할 필요가 없다. 샘플링부터 다시 만들 때만 아래 순서를 사용한다.

### 7.1 샘플링

```powershell
C:\Users\LSH\Downloads\ITDA\.venv\Scripts\python.exe .\Sampling.py `
  --processed-dir .\data\processed_rur_shock_context `
  --output-dir .\data\sampled_relative_flags_q75_m2 `
  --strategy legacy `
  --flag-mode relative `
  --flag-quantile 0.75 `
  --min-flags 2 `
  --min-reviews 10 `
  --min-users 8
```

### 7.2 Split

```powershell
C:\Users\LSH\Downloads\ITDA\.venv\Scripts\python.exe .\MakeSplits.py `
  --sampled-dir .\data\sampled_relative_flags_q75_m2 `
  --output-dir .\data\splits_relative_flags_q75_m2 `
  --method temporal `
  --train-ratio 0.64 `
  --valid-ratio 0.16 `
  --test-ratio 0.20 `
  --random-state 42
```

### 7.3 텍스트 임베딩

```powershell
C:\Users\LSH\Downloads\ITDA\.venv\Scripts\python.exe .\BuildTextEmbeddings.py `
  --sampled-dir .\data\sampled_relative_flags_q75_m2 `
  --split-dir .\data\splits_relative_flags_q75_m2 `
  --output-dir .\data\embeddings_relative_flags_q75_m2 `
  --svd-dim 128 `
  --random-state 42
```

### 7.4 엣지 생성

```powershell
C:\Users\LSH\Downloads\ITDA\.venv\Scripts\python.exe .\BuildEdges.py `
  --sampled-dir .\data\sampled_relative_flags_q75_m2 `
  --embedding-path .\data\embeddings_relative_flags_q75_m2\sampled_text_tfidf_svd.npy `
  --output-dir .\data\edges_relflag_edge_t15_w60_thr075 `
  --relation1-mode none `
  --custom2-edge-mode weak_product_shock `
  --shock-edge-style peer `
  --shock-score-mode standardized `
  --shock-min-abs-rating-dev 0.75 `
  --shock-date-window-days 60 `
  --shock-topk 15 `
  --shock-exclude-neutral `
  --max-neighbors-per-node 5
```

### 7.5 그래프 생성

```powershell
C:\Users\LSH\Downloads\ITDA\.venv\Scripts\python.exe .\BuildGraphDataset.py `
  --sampled-dir .\data\sampled_relative_flags_q75_m2 `
  --embedding-path .\data\embeddings_relative_flags_q75_m2\sampled_text_tfidf_svd.npy `
  --edge-dir .\data\edges_relflag_edge_t15_w60_thr075 `
  --split-dir .\data\splits_relative_flags_q75_m2 `
  --output-dir .\data\graph_relflag_edge_t15_w60_thr075 `
  --add-rating-history-text-features `
  --add-product-prior-context-features
```

## 8. 방향성 피처 후보의 위치

방향성 피처 적용 모델은 현재 실행 대상이 아니다. 관련 리소스는 아래 경로에 보존되어 있으며, 후속 연구에서 설명 가능성 강화 후보로만 다룬다.

```text
data/graph_relflag_edge_t15_w60_eda_logic_directional/
experiments/relflag_t15_w60_eda_logic_directional_sage_inverse_seed42/
Modeling_Second.md
```

## 9. 주요 문서

| 문서 | 내용 |
|---|---|
| `EDA_Report.md` | EDA 결과와 최종 모델 설계로 이어지는 논리 |
| `Sampling_Report.md` | 최종 상대 변화량 flag 기반 샘플링 기준 |
| `Modeling_Report.md` | 기존 topk15 window60 SAGE 기준 모델 |
| `Modeling_Second.md` | 방향성 피처 적용 후보와 기준 모델의 차이 |
| `Run_Topk15_Window60_SAGE.md` | 기존 topk15 window60 SAGE 단독 실행 안내 |

## 10. 현재 남긴 산출물

현재 실행 모델과 후보 보존에 필요한 주요 산출물은 다음과 같다.

```text
data/origin/
data/eda/
data/processed/
data/processed_rur_shock_context/
data/sampled_relative_flags_q75_m2/
data/splits_relative_flags_q75_m2/
data/embeddings_relative_flags_q75_m2/
data/edges_relflag_edge_t15_w60_thr075/
data/graph_relflag_edge_t15_w60_thr075/
data/graph_relflag_edge_t15_w60_eda_logic_directional/
experiments/relflag_edge_t15_w60_thr075_sage_inverse_seed42/
experiments/relflag_t15_w60_eda_logic_directional_sage_inverse_seed42/
```
