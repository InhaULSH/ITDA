# Modeling Second: 방향성 피처 적용 topk15 window60 SAGE

## 1. 문서의 역할

이 문서는 기존 `topk15 window60 SAGE` 기준 모델과 방향성 피처 적용 모델의 차이만 정리한다. 기준 모델 자체의 샘플링, 엣지, 학습 설정은 `Modeling_Report.md`에 설명되어 있다.

방향성 피처 적용 모델의 실험 폴더는 다음과 같다.

```text
experiments/relflag_t15_w60_eda_logic_directional_sage_inverse_seed42
```

입력 그래프는 다음 파일이다.

```text
data/graph_relflag_edge_t15_w60_eda_logic_directional/graph_rur_custom2.pt
```

두 모델은 같은 샘플, 같은 split, 같은 텍스트 임베딩, 같은 엣지를 사용한다. 차이는 그래프 생성 단계의 노드 피처 profile이다.

## 2. 왜 방향성 피처를 추가했는가

기존 기준 모델은 평점 방향, 극단 평점, 상품 과거 평균 대비 평점 이탈을 이미 사용한다. 그러나 긍정 조작과 부정 공격을 설명할 때는 두 현상을 완전히 대칭으로 보는 것이 자연스럽지 않다.

긍정 조작은 평판이 약하거나 리뷰 이력이 적은 상품에서 평균 평점과 소비자 인상을 끌어올릴 유인이 크다. 반대로 부정 공격은 이미 어느 정도 평판이 형성된 상품에 낮은 평점을 남겨 소비자 판단을 흔드는 방식으로 나타날 수 있다. 두 경우 모두 “평점 shock”이지만, 경제적 유인과 해석 맥락은 다르다.

또한 오탐 방지도 중요하다. 정상 사용자가 최근에 다시 방문해 자신의 평소 평점 성향과 크게 다르지 않은 리뷰를 쓰고, 설명도 충분하다면, 그 리뷰가 짧거나 특정 상품-주에 포함되었다는 이유만으로 과하게 의심해서는 안 된다. 방향성 피처는 이런 정상 사용자 맥락을 모델과 설명 산출물에 함께 제공하기 위해 추가했다.

## 3. 바뀌지 않은 부분

방향성 피처 적용 모델에서도 다음 요소는 기준 모델과 같다.

| 항목 | 값 |
|---|---|
| 샘플 | `data/sampled_relative_flags_q75_m2` |
| 노드 수 | 26,701 |
| split | train 17,088 / valid 4,272 / test 5,341 |
| 텍스트 임베딩 | `data/embeddings_relative_flags_q75_m2` |
| 엣지 | R-U-R 11,038개 + weak product shock 4,266개 |
| 전체 엣지 수 | 15,304 |
| 모델 구조 | plain GraphSAGE |
| hidden dimension | 128 |
| layer | 2 |
| dropout | 0.5 |
| learning rate | 0.001 |
| early stopping | validation PR-AUC |
| threshold 선택 | validation 기준 prevalence-constrained Macro F1 |

따라서 두 모델의 성능 차이는 샘플링이나 엣지 차이가 아니라 노드 피처 설계 차이로 해석할 수 있다.

## 4. 노드 피처 차이

기준 모델은 숫자형 피처 46차원과 텍스트 임베딩 128차원을 합쳐 총 174차원을 사용한다. 방향성 피처 적용 모델은 숫자형 피처 57차원과 텍스트 임베딩 128차원을 합쳐 총 185차원을 사용한다.

방향성 profile의 이름은 `eda_logic_directional`이다. 이 profile은 기존 sampled numeric feature를 보존하면서, EDA 논리에 맞춘 보조 피처를 추가한다. 특히 다음 피처들이 방향성 해석의 핵심이다.

| 피처 | 의미 |
|---|---|
| `positive_standardized_rating_deviation_from_prior_product_mean` | 상품 과거 평균보다 위로 벗어난 정도 |
| `negative_standardized_rating_deviation_from_prior_product_mean` | 상품 과거 평균보다 아래로 벗어난 정도 |
| `positive_promotion_low_reputation_shock` | 평판 이력이 약한 상품에서 위쪽 평점 shock이 발생한 정도 |
| `negative_attack_high_reputation_shock` | 이력이 많은 상품에서 아래쪽 평점 shock이 발생한 정도 |
| `positive_extreme_rating_shock` | 극단 긍정 평점과 위쪽 shock의 결합 |
| `negative_extreme_rating_shock` | 극단 부정 평점과 아래쪽 shock의 결합 |
| `disguised_high_effort_sparse_user_shock` | 설명은 충분하지만 신규·저이력 사용자의 평점 shock인 경우 |
| `returning_recent_user_consistent_rating` | 최근 재방문 사용자가 자기 평점 성향에서 크게 벗어나지 않은 경우 |
| `returning_recent_user_consistent_rating_and_text_sufficient` | 위 조건에 설명 충분성까지 함께 만족하는 정상 사용자 보호 맥락 |

사용자 평점 일관성 기준은 임의 계수가 아니라 train 구간에서 이전 리뷰가 있는 사용자들의 표준화 평점 이탈 분포로 정했다. 최종 방향성 그래프의 기준값은 `0.75`이다.

## 5. 설명 산출물 차이

방향성 피처 적용 모델은 예측 후 설명 산출물도 함께 생성한다.

```text
experiments/relflag_t15_w60_eda_logic_directional_sage_inverse_seed42/prediction_explanations.csv
experiments/relflag_t15_w60_eda_logic_directional_sage_inverse_seed42/prediction_explanation_summary.json
```

설명 태그는 학습 라벨이 아니라 진단용 태그이다. 라벨 기반 집계값을 쓰지 않고, 리뷰 시점의 피처와 train 기준 cutoff로 계산한다.

| 설명 태그 | 의미 | 전체 노드 중 개수 |
|---|---|---:|
| `평판취약상품_평점충격` | 상품 이력이 약한 상태에서 평점 shock이 큼 | 2,963 |
| `긍정홍보형_평점상승` | 위쪽 평점 shock | 7,972 |
| `부정공격형_평점하락` | 아래쪽 평점 shock | 3,425 |
| `긍정홍보형_취약평판상승` | 취약 평판 상품에서 긍정 방향 shock | 1,661 |
| `부정공격형_성숙상품하락` | 이력 많은 상품에서 부정 방향 shock | 1,237 |
| `평점대비_설명부족` | 평점·상품 이력 맥락 대비 설명이 짧음 | 7,287 |
| `긴리뷰형_위장가능성` | 설명은 충분하지만 신규·저이력 사용자의 shock | 3,868 |
| `약한템플릿반복` | 상품-주 안에서 약한 템플릿 유사성이 높음 | 2,763 |
| `정상사용자_보호맥락` | 최근 재방문·충분 설명·기존 사용자 맥락 | 4,537 |
| `정상사용자_일관평점충분설명` | 최근 재방문·평점 일관성·충분 설명 | 1,844 |
| `짧지만_기존사용자일관맥락` | 짧지만 기존 사용자의 평점 일관성이 있는 경우 | 729 |

이 설명 체계의 목적은 모델의 결론을 단일 위험 점수로만 제시하지 않고, 어떤 행동 논리 때문에 의심 또는 보호 맥락이 생겼는지를 사람이 검토할 수 있게 하는 것이다.

## 6. 성능 비교

두 후보의 test 성능은 다음과 같다.

| 모델 | Numeric dim | Total dim | PR-AUC | ROC-AUC | Macro F1 | Precision | Recall | Accuracy |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 기존 topk15 window60 SAGE | 46 | 174 | 0.4901 | 0.7851 | 0.6793 | 0.4956 | 0.4965 | 0.7839 |
| 방향성 피처 적용 topk15 window60 SAGE | 57 | 185 | 0.4882 | 0.7878 | 0.6744 | 0.4937 | 0.4790 | 0.7832 |

방향성 피처 적용 모델은 PR-AUC와 Macro F1이 기준 모델보다 약간 낮다. 반면 ROC-AUC는 소폭 높다. 순수 성능만 보면 기존 `topk15 window60 SAGE`가 1차 기준으로 더 적합하다. 다만 방향성 피처 적용 모델은 긍정 조작, 부정 공격, 정상 사용자 보호 맥락을 더 직접적으로 설명할 수 있다는 장점이 있다.

## 7. 최종 해석

현재 단계에서 두 모델을 모두 후보로 남기는 이유는 역할이 다르기 때문이다.

기존 `topk15 window60 SAGE`는 성능 기준 후보이다. Macro F1과 PR-AUC가 방향성 모델보다 높고, 모델 구조와 피처 수가 상대적으로 단순하다. 운영 성능과 계산 비용을 우선하면 이 모델이 기준이다.

방향성 피처 적용 `topk15 window60 SAGE`는 설명 가능성 기준 후보이다. 성능은 약간 손해를 보지만, EDA에서 도출한 평판 조작의 비대칭성, 평점 이탈과 설명 충분성의 결합, 위장 비용, 약한 템플릿 반복, 정상 사용자 보호 맥락을 보고서와 진단 산출물에 더 분명히 연결할 수 있다.

따라서 후속 단계에서 성능을 우선하는 실험은 기존 모델을 기준으로 진행하고, 발표 또는 해석 중심 산출물에서는 방향성 피처 적용 모델을 함께 비교 후보로 제시하는 것이 타당하다.

## 8. 보존 상태

방향성 피처 적용 모델은 현재 실행 대상이 아니다. 실행 방법은 `README.md`와 `RunExperiments.py` 모두 기존 `topk15 window60 SAGE` 기준으로만 정리되어 있다. 방향성 피처 적용 모델의 목적은 추후 설명 가능성 강화 후보를 검토할 때 비교 근거를 제공하는 것이다.

현재 보존된 산출물은 다음과 같다.

```text
data/graph_relflag_edge_t15_w60_eda_logic_directional/
experiments/relflag_t15_w60_eda_logic_directional_sage_inverse_seed42/
```
