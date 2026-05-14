# YelpZip 샘플링 설계 보고서

## 1. 최종 샘플링의 위치

현재 프로젝트에서 유지하는 기준 샘플은 `data/sampled_relative_flags_q75_m2`이다. 이 샘플은 기존 `topk15 window60 SAGE` 모델과 방향성 피처 적용 `topk15 window60 SAGE` 모델이 공통으로 사용한다. 두 모델의 차이는 노드 피처 profile뿐이며, 샘플링, split, 텍스트 임베딩, 엣지 구조는 동일하다.

샘플링의 목적은 전체 608,458개 리뷰 중 GNN이 학습할 수 있을 만큼 관계가 살아 있는 1만~5만 개 규모의 서브그래프를 만드는 것이다. 단순 무작위 추출은 리뷰 간 관계가 희박해질 수 있고, 반대로 너무 강한 의심 조건만 사용하면 정상 리뷰 분포를 잃어 오탐이 늘 수 있다. 따라서 최종 샘플링은 `상품-주 단위의 기본 gate`와 `상대 변화량 기반 flag`를 결합한다.

## 2. 기본 단위와 gate

최종 샘플링 단위는 `prod_id × week`이다. 리뷰 하나를 직접 뽑는 것이 아니라, 먼저 상품-주 단위를 평가한 뒤 선택된 상품-주 안의 리뷰를 모두 포함한다. 이렇게 하면 이후 R-U-R 엣지와 weak product shock 엣지가 단절되지 않고, 특정 상품의 특정 기간에 발생한 평점 이동 맥락을 보존할 수 있다.

기본 gate는 다음과 같다.

```text
상품-주 리뷰 수 n_reviews >= 10
상품-주 고유 사용자 수 n_users >= 8
```

이 기준은 너무 작은 상품-주에서 발생하는 우연한 변동을 줄이기 위한 최소 조건이다. 전체 상품-주 구간 290,477개 중 이 gate를 통과한 구간은 4,087개이며, 해당 구간에 포함된 리뷰는 59,467개이다. 여기서 바로 모든 리뷰를 쓰지 않고, 추가 flag를 적용해 최종 표본을 26,701개 리뷰로 줄였다.

## 3. 상대 변화량 flag를 사용한 이유

초기 기준은 리뷰 증가율, 신규 사용자 비율, 극단 평점 비율, 짧은 리뷰 비율, 약한 상품 이력, 평점 이탈, 평점 방향 집중도 같은 절댓값 중심 지표였다. 그러나 절댓값 기준은 상품별 차이를 충분히 반영하지 못한다. 예를 들어 원래부터 리뷰가 많고 평점 변동이 큰 상품과, 원래 조용하고 평점이 안정적인 상품은 같은 1점 변동이라도 의미가 다르다.

최종 샘플링은 기존 SAGE baseline의 틀을 유지하되, flag의 기준을 가능한 한 과거 흐름 대비 변화량으로 바꾸었다. 즉 “짧은 리뷰가 많다”보다 “평소보다 짧은 리뷰 비중이 올라갔다”, “평점이 많이 벗어났다”보다 “그 상품의 과거 평점 변동 수준에 비해 많이 벗어났다”를 더 중요하게 본다.

이 방식은 임의 가중치 점수 하나로 위험도를 만드는 방식이 아니다. 각 flag는 train 이전 또는 리뷰 작성 시점 이전에 계산 가능한 관측값을 사용하고, 기본 후보 안에서 75분위수 이상인 경우를 flag로 둔다. 최종 선택은 여러 flag 중 2개 이상이 동시에 관측되는 상품-주를 고르는 방식이다.

## 4. 최종 flag 구조

최종 샘플링은 `--flag-mode relative`, `--flag-quantile 0.75`, `--min-flags 2` 설정을 사용했다. 7개 numeric flag는 기본 후보 집단 내부의 75분위수 이상이면 1로 두고, 템플릿 반복 flag는 희소 사건이므로 동일 상품-주 안에서 같은 텍스트 반복이 하나라도 있으면 1로 둔다.

| flag | 상대 변화량 모드에서의 의미 | 실행 임계값 |
|---|---|---:|
| `review_growth_flag` | 같은 평점 방향 리뷰 수가 과거 4주 흐름 대비 증가 | 0.5534 |
| `new_user_ratio_flag` | 신규 사용자 비율이 과거 4주 흐름 대비 증가 | 0.1308 |
| `extreme_rating_ratio_flag` | 극단 평점 및 행동 변화가 결합된 버스트성 점수 | 0.9361 |
| `short_review_ratio_flag` | 짧은 리뷰 비율이 과거 4주 흐름 대비 증가 | 0.0629 |
| `weak_product_flag` | legacy 이름은 유지하지만 실제로는 평균 단어 길이 하락 신호로 사용 | 0.2228 |
| `rating_deviation_flag` | 상품 과거 평점 표준편차 대비 평균 평점 이탈 | 0.9078 |
| `rating_direction_concentration_flag` | 평점 방향 집중도가 과거 4주 흐름 대비 증가 | 0.1053 |
| `local_template_repeat_flag` | 같은 상품-주 안에서 동일 텍스트 반복 존재 | 동일 텍스트 2회 이상 |

`weak_product_flag`라는 이름은 legacy 코드와 산출물 호환성을 위해 그대로 두었다. 다만 최종 보고서에서는 이 flag를 “상품이 약하다”는 직접 의미보다 “상대 변화량 모드에서 리뷰 설명 길이 하락을 반영하는 legacy 이름의 flag”로 해석한다. 모델 내용에는 영향을 주지 않기 위해 컬럼명은 바꾸지 않았다.

최종 선택 규칙은 다음과 같다.

```text
기본 gate를 통과하고,
8개 flag 중 2개 이상을 만족하는 상품-주를 선택한다.
선택된 상품-주 안의 리뷰 노드는 모두 포함한다.
```

## 5. 최종 샘플 규모

최종 샘플 산출물은 `data/sampled_relative_flags_q75_m2`에 저장되어 있다.

| 항목 | 값 |
|---|---:|
| 전체 리뷰 노드 | 608,458 |
| 선택된 리뷰 노드 | 26,701 |
| 전체 대비 비율 | 4.39% |
| 선택된 상품-주 구간 | 1,957 |
| 선택된 상품 수 | 243 |
| 선택된 사용자 수 | 22,914 |
| 기간 | 2007-08-06 ~ 2015-01-10 |
| fake rate | 0.1604 |

split은 시간 기준으로 구성되며, 최종 그래프 파일 내부 mask로 저장된다.

| split | 노드 수 | fake | real | fake rate |
|---|---:|---:|---:|---:|
| train | 17,088 | 2,372 | 14,716 | 0.1388 |
| valid | 4,272 | 768 | 3,504 | 0.1798 |
| test | 5,341 | 1,144 | 4,197 | 0.2142 |

분할 비율은 train 64%, validation 16%, test 20%이다. 대회 필수 조건에서 요구하는 “학습용 80%, 시험용 20%”를 train과 validation을 합친 80%로 만족한다. split 생성 코드는 `MakeSplits.py`이며, 현재 산출물의 `split_summary.json`에는 `method=temporal`, `random_state=42`, `train_ratio=0.64`, `valid_ratio=0.16`, `test_ratio=0.20`이 기록되어 있다. 현재 사용한 temporal split은 날짜 순서로 나누기 때문에 `random_state`가 실제 섞기에는 거의 관여하지 않지만, 재현성 표기를 위해 하이퍼파라미터로 함께 명시한다.

fake rate는 샘플 선택에 사용하지 않고, 선택 후 진단용으로만 확인했다. 전체 데이터보다 fake rate가 다소 높지만, fake만 과도하게 모은 표본은 아니다. 이는 정상 리뷰 맥락을 유지하면서도 어뷰징 가능성이 있는 상품-주를 충분히 포함하기 위한 절충이다.

## 6. 산출물 구조

최종 샘플 폴더의 핵심 파일은 다음과 같다.

| 파일 | 역할 |
|---|---|
| `sampled_review_nodes.csv.gz` | 선택된 리뷰 노드 메타데이터 |
| `sampled_review_text.csv.gz` | 선택된 리뷰 원문 |
| `sampled_relation_candidate_keys.csv.gz` | 엣지 생성을 위한 relation 후보 키 |
| `sampled_node_features_numeric.npy` | 선택된 노드의 숫자형 피처, shape `(26701, 37)` |
| `sampled_node_labels.npy` | 선택된 노드의 라벨 |
| `sampled_node_review_ids.npy` | 샘플 행과 원본 review id 매핑 |
| `sampled_node_mapping.csv` | `sampled_node_idx`, `original_node_idx`, `review_id` 연결표 |
| `product_week_sampling_units.csv.gz` | 전체 상품-주 후보의 flag와 선택 여부 |
| `sampling_summary.json` | 샘플링 기준과 결과 요약 |

중요한 점은 이후 모든 엣지와 그래프가 `sampled_node_idx`를 기준으로 만들어진다는 것이다. `original_node_idx`는 원본 전처리 노드 번호를 보존하기 위한 값이며, `edge_index`에는 사용하지 않는다.

## 7. 재현 명령

최종 샘플을 다시 만들 때 사용하는 명령은 다음과 같다.

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

샘플링 이후에는 같은 샘플을 기준으로 split, 텍스트 임베딩, 엣지, 그래프를 생성한다. 이 단계들은 `README.md`에 최종 실행 순서로 정리했다.

split을 재생성할 때 사용하는 명령은 다음과 같다.

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

## 8. 최종 판단

최종 샘플링은 기존 SAGE baseline의 큰 틀인 “상품-주 gate + 다중 flag 선택”을 유지하면서, flag의 의미를 절댓값 중심에서 과거 흐름 대비 변화량 중심으로 옮긴 것이다. 이 설계는 모델이 전체 분포를 지나치게 좁히지 않으면서도, 평판을 움직일 가능성이 있는 상품-주를 우선적으로 학습하게 한다.

이 샘플은 현재 남긴 두 후보 모델의 공통 기반이다. 따라서 이후 모델 비교에서 성능 차이는 샘플링 차이가 아니라, 기존 feature set과 방향성 EDA feature set의 차이로 해석할 수 있다.
