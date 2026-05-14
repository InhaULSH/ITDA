# Modeling Report: topk15 window60 SAGE 기준 모델

## 1. 최종 기준 모델

이 문서는 현재 프로젝트에서 기준 후보로 유지하는 `topk15 window60 SAGE` 모델을 설명한다. 해당 모델의 실험 폴더는 다음과 같다.

```text
experiments/relflag_edge_t15_w60_thr075_sage_inverse_seed42
```

입력 그래프는 다음 파일이다.

```text
data/graph_relflag_edge_t15_w60_thr075/graph_rur_custom2.pt
```

이 모델은 plain GraphSAGE이다. relation별 파라미터를 따로 두는 `relation_sage`가 아니라, `edge_index`에 들어 있는 모든 엣지를 하나의 이웃 그래프로 보고 메시지를 전달한다. 그래프 안에는 R-U-R 엣지와 weak product shock 엣지가 함께 들어 있지만, 학습 단계에서는 두 엣지 타입이 별도 가중치로 분리되지는 않는다.

## 2. 입력 데이터

기준 모델은 `data/sampled_relative_flags_q75_m2` 샘플을 사용한다. 이 샘플은 `n_reviews >= 10`, `n_users >= 8`의 상품-주 gate를 통과한 후보 중 상대 변화량 flag를 2개 이상 만족하는 상품-주를 선택해 만든 것이다.

노드 수는 26,701개이며, split은 다음과 같다.

| split | 노드 수 | fake | real | fake rate |
|---|---:|---:|---:|---:|
| train | 17,088 | 2,372 | 14,716 | 0.1388 |
| valid | 4,272 | 768 | 3,504 | 0.1798 |
| test | 5,341 | 1,144 | 4,197 | 0.2142 |

라벨은 학습과 평가에만 사용되며, 샘플링 flag, 노드 피처, 엣지 생성 기준에는 사용하지 않는다.

## 3. 노드 피처

기준 모델의 전체 입력 피처 차원은 174차원이다. 이 중 숫자형 피처가 46차원이고, 텍스트 TF-IDF/SVD 임베딩이 128차원이다.

기본 숫자형 피처는 전처리된 리뷰 자체 특성, 사용자 과거 맥락, 상품 과거 맥락으로 구성된다. 예를 들어 `rating_norm`, `rating_direction`, `extreme_rating`, `log_word_len`, `log1p_prior_user_review_count`, `prior_user_avg_rating`, `log1p_prior_product_review_count`, `prior_product_avg_rating`, `prior_product_rating_std`, `rating_impact_abs` 등이 포함된다.

여기에 그래프 생성 단계에서 9개의 보조 피처를 추가했다.

| 추가 피처 | 의미 |
|---|---|
| `word_len_minus_train_rating_history_median` | 같은 평점대·상품 이력 구간의 train 기준 중앙값 대비 리뷰 길이 차이 |
| `word_len_ratio_to_train_rating_history_median` | 같은 맥락의 train 기준 중앙값 대비 리뷰 길이 비율 |
| `is_short_by_train_rating_history_q25` | 같은 맥락에서 하위 25%보다 짧은 리뷰 여부 |
| `extreme_rating_and_short_by_train_rating_history_q25` | 극단 평점이면서 상대적으로 짧은 리뷰 여부 |
| `standardized_rating_deviation_from_prior_product_mean` | 상품 과거 평균 대비 평점 이탈을 과거 표준편차로 나눈 값 |
| `abs_standardized_rating_deviation_from_prior_product_mean` | 표준화 평점 이탈의 절댓값 |
| `product_reviews_last_7d_share_of_prior_count` | 최근 7일 리뷰 수가 과거 누적 리뷰 수에 비해 큰 정도 |
| `product_reviews_last_30d_share_of_prior_count` | 최근 30일 리뷰 수가 과거 누적 리뷰 수에 비해 큰 정도 |
| `low_product_history_by_train_q25` | train 기준 하위 25% 수준의 상품 이력 여부 |

이 피처들은 모두 train mask 기준으로 기준값을 fit한 뒤 validation/test에 적용한다. 따라서 검증·평가 구간의 분포 정보를 피처 기준 계산에 직접 사용하지 않는다.

텍스트 임베딩은 `data/embeddings_relative_flags_q75_m2/sampled_text_tfidf_svd.npy`를 사용한다. TF-IDF와 SVD도 train 텍스트에 fit하고 전체 split에 transform하는 구조이다.

## 4. 엣지 설계

기준 모델은 `graph_rur_custom2.pt`를 사용한다. 이 그래프에는 relation 0인 R-U-R 엣지와 relation 2인 weak product shock 엣지만 들어 있다. relation 1 burst 엣지는 사용하지 않는다.

| relation | 엣지 수 | 의미 |
|---|---:|---|
| R-U-R, edge type 0 | 11,038 | 같은 사용자의 리뷰들을 날짜 기준 가까운 리뷰 중심으로 연결 |
| weak product shock, edge type 2 | 4,266 | 과거 상품 평점 흐름 대비 평점 shock이 큰 리뷰 후보끼리 연결 |
| 전체 | 15,304 | plain SAGE가 사용하는 최종 message passing 그래프 |

R-U-R 엣지는 같은 사용자의 리뷰 이력을 전달한다. 이것은 같은 사용자를 위험하다고 단정하기 위한 relation이 아니라, 사용자 과거 활동 맥락을 주변 정보로 전달하기 위한 relation이다.

weak product shock 엣지는 다음 기준으로 만들어졌다.

```text
custom2_edge_mode       = weak_product_shock
shock_edge_style        = peer
shock_score_mode        = standardized
shock_min_abs_rating_dev = 0.75
shock_date_window_days  = 60
shock_topk              = 15
shock_exclude_neutral   = True
```

여기서 standardized shock은 현재 리뷰의 평점이 과거 상품 평균에서 얼마나 벗어났는지를 과거 상품 평점 표준편차로 나눈 값이다. 절대 별점 차이가 아니라 상품별 평소 변동 수준을 고려하기 때문에, 평소 안정적인 상품에서의 작은 변화와 평소 변동이 큰 상품에서의 큰 변화를 더 공정하게 비교할 수 있다.

## 5. 모델 구조와 학습 설정

모델은 2-layer GraphSAGE이다. hidden dimension은 128, dropout은 0.5, learning rate는 0.001, weight decay는 0.0001이다. seed는 42이고, 최대 epoch는 200, early stopping patience는 30이다.

loss는 train split의 클래스 불균형을 고려해 inverse-frequency class weight를 사용한다. fake 리뷰가 정상 리뷰보다 적기 때문에, 가중치를 주지 않으면 모델이 정상 리뷰 예측에 과도하게 치우칠 수 있다.

early stopping 기준은 validation PR-AUC이다. threshold는 validation split에서 `prevalence_constrained_macro_f1` 기준으로 선택한다. 즉 test 성능을 보고 threshold를 고르지 않는다. 최종 threshold는 0.75이다.

## 6. 성능

기준 모델의 test 성능은 다음과 같다.

| 지표 | 값 |
|---|---:|
| PR-AUC | 0.4901 |
| ROC-AUC | 0.7851 |
| Macro F1 | 0.6793 |
| Precision | 0.4956 |
| Recall | 0.4965 |
| Accuracy | 0.7839 |
| Best epoch | 71 |
| Threshold | 0.75 |

검증 성능은 validation PR-AUC 0.3940, validation Macro F1 0.6458이다. train 성능은 PR-AUC 0.5889, ROC-AUC 0.8959, Macro F1 0.7331이다. train과 test 사이에 차이가 있으므로 과적합 가능성을 완전히 배제할 수는 없지만, 현재 후보 중에서는 threshold 적용 후 균형 성능이 가장 좋은 축에 속한다.

## 7. 모델 해석

이 모델은 세 수준의 정보를 결합한다.

첫째, 리뷰 자체의 정보이다. 평점 방향, 극단 평점 여부, 문장 길이, 문장부호, 텍스트 임베딩을 통해 리뷰 하나가 가진 표현적 특징을 본다.

둘째, 리뷰 작성 시점 이전의 사용자·상품 맥락이다. 사용자가 처음 등장한 계정인지, 과거에 어떤 평점 성향을 가졌는지, 상품의 과거 리뷰 수와 평점 변동성은 어떤지를 본다.

셋째, 리뷰 간 관계이다. 같은 사용자의 리뷰 흐름과 유사한 weak product shock 리뷰군을 함께 보면서, 단일 리뷰만 볼 때 애매한 신호를 주변 구조로 보완한다.

결과적으로 `topk15 window60 SAGE`는 전체 순위화 성능만 극대화하는 모델이라기보다, 상품의 과거 흐름 대비 평점 이동과 사용자 맥락을 함께 보아 실제 검토 대상 리뷰를 고르는 모델로 해석하는 것이 적절하다.

## 8. 실행 명령

현재 저장된 산출물을 그대로 사용하면 재학습할 필요가 없다. 같은 설정으로 재학습하려면 다음 명령을 사용한다.

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
