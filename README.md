# ITDA YelpZip Review Abuse GNN

이 폴더는 YelpZip 리뷰 데이터를 사용해 GNN 기반 조직적 어뷰징 리뷰 탐지 모델을 학습하고, 최종 제출 모델의 산출물을 검증할 수 있도록 정리한 코드 패키지입니다.

실행 목적은 세 가지로 구분합니다.

1. **검증용 실행**: 이미 포함된 최종 산출물이 보고서의 수치와 일치하는지 확인합니다.
2. **포함된 최종 그래프 기준 재학습**: 원천 데이터 전처리와 그래프 생성을 다시 하지 않고, 포함된 최종 그래프에서 모델만 다시 학습합니다.
3. **원천 데이터부터 전체 재생성**: `data/origin/yelpzip.csv`에서 시작해 최종 그래프, 모델, 운영 threshold, 거버넌스 레이어까지 모두 다시 생성합니다.

## 실행 환경

기본 패키지는 `requirements.txt`에 정리되어 있습니다. 먼저 아래 명령으로 의존성을 설치합니다.

```powershell
pip install -r requirements.txt
```

PyTorch와 PyTorch Geometric은 실행 환경의 CPU/CUDA/XPU 조건에 따라 설치 방식이 달라질 수 있습니다. `requirements.txt` 설치 중 PyTorch 계열 wheel 오류가 발생하면, 해당 평가 환경에 맞는 PyTorch와 PyTorch Geometric을 먼저 설치한 뒤 나머지 패키지를 설치하십시오.

## 최종 모델

| 항목 | 값 |
|---|---:|
| 모델 | `relation_sage_mlp` |
| 실험 폴더 | `experiments/campaign_quality_q60_relation_sage_mlp_equal_seed42` |
| 그래프 | `data/graph_campaign_quality_q60_top3_b6000_s020/graph_rur_custom2.pt` |
| 노드 | 26,701 |
| 피처 차원 | 182 |
| directed edge | 21,304 |
| best epoch | 60 |
| 운영 threshold | 0.794 |

최종 모델은 relation-aware GraphSAGE와 Self Branch를 결합한 `relation_sage_mlp`입니다. 그래프 branch는 `R-U-R`, `Filtered Campaign Pair`, `Weak Product Shock` 세 relation을 따로 처리하고, Self Branch는 리뷰 자체 피처를 MLP로 직접 처리합니다. 따라서 연결된 리뷰는 관계별 이웃 정보를 활용하고, 연결이 부족한 리뷰는 자기 피처 기반 판단을 유지할 수 있습니다.

## 최종 성능

| split | PR-AUC | ROC-AUC | Macro F1 | Precision | Recall | Accuracy |
|---|---:|---:|---:|---:|---:|---:|
| valid | 0.4441 | 0.7850 | 0.6601 | 0.4690 | 0.4036 | 0.8106 |
| test | 0.5288 | 0.8051 | 0.6962 | 0.5273 | 0.5149 | 0.7972 |

PR-AUC와 ROC-AUC는 ranking 성능이며, Precision, Recall, Macro F1, Accuracy는 운영 threshold `0.794`를 적용한 결과입니다.

## 주요 relation

| relation | edge 수 | 의미 |
|---|---:|---|
| R-U-R | 11,038 | 동일 사용자의 가까운 리뷰 흐름 |
| Filtered Campaign Pair | 6,000 | `prod_id x week x rating_direction` 단위의 캠페인성 리뷰 쌍 |
| Weak Product Shock | 4,266 | 상품 과거 평점 흐름 대비 이례적인 평점 이동 |

`Filtered Campaign Pair`는 q=0.60 기준으로 선별했습니다. 같은 상품과 같은 기간이라는 이유만으로 모든 리뷰를 묶지 않고, 신규 사용자 증가, 짧은 리뷰 증가, 리뷰 길이 하락, 평점 방향 집중이 함께 나타나는 경우를 중심으로 제한적으로 연결합니다.

## 제출 폴더 구조

```text
ITDA/
  README.md
  requirements.txt
  Preprocess.py
  Sampling.py
  MakeSplits.py
  BuildTextEmbeddings.py
  BuildEdges.py
  BuildGraphDataset.py
  TrainGNN.py
  RunExperiments.py
  ApplyOperatingThreshold.py
  BuildGovernanceLayer.py
  VerifyResults.py
  GvncLayer_Report.md
  data/
    origin/
      yelpzip.csv
    processed_rur_shock_context/
    sampled_relative_flags_q75_m2/
    splits_relative_flags_q75_m2/
    embeddings_relative_flags_q75_m2/
    edges_campaign_quality_q60_top3_b6000_s020/
    graph_campaign_quality_q60_top3_b6000_s020/
    governance_layer/
  experiments/
    campaign_quality_q60_relation_sage_mlp_equal_seed42/
```

## 1. 검증용 실행

이미 학습된 최종 모델 산출물이 포함되어 있으므로, 평가자는 아래 명령으로 핵심 산출물이 현행 최종 모델과 일치하는지 바로 확인할 수 있습니다.

```powershell
python .\VerifyResults.py --strict
```

이 명령은 재학습을 수행하지 않습니다. 전처리 행 수, 샘플 수, 그래프 구조, 최종 모델 설정, validation/test 성능, 주요 EDA 체크 값을 검증합니다.

## 2. 포함된 최종 그래프 기준 재학습

원천 데이터 처리와 그래프 생성을 다시 하지 않고, 포함된 최종 그래프에서 모델만 재학습하려면 아래 순서로 실행합니다. `RunExperiments.py`는 최종 그래프 `data/graph_campaign_quality_q60_top3_b6000_s020/graph_rur_custom2.pt`와 최종 실험 폴더를 사용하도록 고정되어 있습니다.

```powershell
python .\RunExperiments.py --device auto --force
```

학습이 끝난 뒤 운영 threshold `0.794`를 다시 적용합니다.

```powershell
python .\ApplyOperatingThreshold.py `
  --experiment-dir .\experiments\campaign_quality_q60_relation_sage_mlp_equal_seed42 `
  --threshold 0.794
```

스트림릿 대시보드에서 사용할 거버넌스 레이어 산출물을 다시 생성합니다.

```powershell
python .\BuildGovernanceLayer.py `
  --output-dir .\data\governance_layer `
  --experiment-dir .\experiments\campaign_quality_q60_relation_sage_mlp_equal_seed42 `
  --graph-dir .\data\graph_campaign_quality_q60_top3_b6000_s020 `
  --edge-dir .\data\edges_campaign_quality_q60_top3_b6000_s020
```

재학습 후 산출물 정합성을 확인하려면 다시 검증 명령을 실행합니다.

```powershell
python .\VerifyResults.py --strict
```

## 3. 원천 데이터부터 전체 재생성

원천 데이터부터 최종 산출물을 모두 다시 만들려면 `data/origin/yelpzip.csv`가 필요합니다. 전체 재생성에서는 기본 출력 경로가 아니라, 현행 최종 모델에서 사용하는 active 경로를 명시해야 합니다.

### 3-1. 전처리

```powershell
python .\Preprocess.py `
  --csv .\data\origin\yelpzip.csv `
  --output .\data\processed_rur_shock_context
```

### 3-2. 샘플링

```powershell
python .\Sampling.py `
  --processed-dir .\data\processed_rur_shock_context `
  --output-dir .\data\sampled_relative_flags_q75_m2 `
  --strategy legacy `
  --flag-mode relative `
  --flag-quantile 0.75 `
  --min-flags 2 `
  --min-reviews 10 `
  --min-users 8 `
  --max-nodes 50000
```

### 3-3. 시간 기준 split

```powershell
python .\MakeSplits.py `
  --sampled-dir .\data\sampled_relative_flags_q75_m2 `
  --output-dir .\data\splits_relative_flags_q75_m2 `
  --method temporal `
  --train-ratio 0.64 `
  --valid-ratio 0.16 `
  --test-ratio 0.20 `
  --random-state 42
```

### 3-4. 텍스트 임베딩

```powershell
python .\BuildTextEmbeddings.py `
  --sampled-dir .\data\sampled_relative_flags_q75_m2 `
  --split-dir .\data\splits_relative_flags_q75_m2 `
  --output-dir .\data\embeddings_relative_flags_q75_m2 `
  --max-features 50000 `
  --svd-dim 128 `
  --min-df 2 `
  --max-df 0.95 `
  --random-state 42
```

### 3-5. relation edge 생성

```powershell
python .\BuildEdges.py `
  --sampled-dir .\data\sampled_relative_flags_q75_m2 `
  --embedding-path .\data\embeddings_relative_flags_q75_m2\sampled_text_tfidf_svd.npy `
  --output-dir .\data\edges_campaign_quality_q60_top3_b6000_s020 `
  --max-neighbors-per-node 5 `
  --burst-min-group-size 3 `
  --relation1-mode filtered_campaign_pair `
  --train-mask-path .\data\splits_relative_flags_q75_m2\train_mask.npy `
  --campaign-filter-comp-quantile 0.60 `
  --campaign-filter-topk 3 `
  --campaign-filter-budget-directed 6000 `
  --campaign-filter-min-pair-score 0.20 `
  --campaign-filter-min-group-size 3 `
  --custom2-edge-mode weak_product_shock `
  --shock-edge-style peer `
  --shock-max-prior-product-reviews 30 `
  --shock-min-abs-rating-dev 0.75 `
  --shock-score-mode standardized `
  --shock-date-window-days 60 `
  --shock-topk 15 `
  --shock-exclude-neutral `
  --shock-min-behavior-shift-score 0.0
```

### 3-6. 최종 PyTorch Geometric 그래프 생성

```powershell
python .\BuildGraphDataset.py `
  --sampled-dir .\data\sampled_relative_flags_q75_m2 `
  --embedding-path .\data\embeddings_relative_flags_q75_m2\sampled_text_tfidf_svd.npy `
  --edge-dir .\data\edges_campaign_quality_q60_top3_b6000_s020 `
  --split-dir .\data\splits_relative_flags_q75_m2 `
  --output-dir .\data\graph_campaign_quality_q60_top3_b6000_s020 `
  --feature-profile legacy `
  --add-rating-history-text-features `
  --add-product-prior-context-features `
  --add-behavior-shift-features `
  --add-campaign-quality-features `
  --graph-keep-edge-types "0,1,2"
```

### 3-7. 모델 학습, threshold 적용, 거버넌스 레이어 생성, 검증

```powershell
python .\RunExperiments.py --device auto --force

python .\ApplyOperatingThreshold.py `
  --experiment-dir .\experiments\campaign_quality_q60_relation_sage_mlp_equal_seed42 `
  --threshold 0.794

python .\BuildGovernanceLayer.py `
  --output-dir .\data\governance_layer `
  --experiment-dir .\experiments\campaign_quality_q60_relation_sage_mlp_equal_seed42 `
  --graph-dir .\data\graph_campaign_quality_q60_top3_b6000_s020 `
  --edge-dir .\data\edges_campaign_quality_q60_top3_b6000_s020

python .\VerifyResults.py --strict
```

## 데이터 누수 관리

라벨, 원본 `tag`, 전체 기간 Fake Rate, 미래 리뷰 수는 입력 피처나 엣지 생성 기준에 사용하지 않습니다. TF-IDF/SVD, scaler, threshold 및 Filtered Campaign Pair의 주요 기준은 train/validation 기준으로만 선택하며, test split은 최종 일반화 성능 확인에만 사용합니다.
