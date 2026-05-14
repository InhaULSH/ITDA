# topk15 window60 SAGE 모델 실행 안내

이 문서는 현재 프로젝트에서 최종 후보로 남겨 둔 `topk15 window60 SAGE` 모델만 바로 사용할 수 있도록 정리한 실행 가이드이다. 다른 개발자가 프로젝트 폴더를 열었을 때 혼동하지 않도록, 기존 SAGE baseline, LightGBM node-only, stacking 보정 모델은 여기서 다루지 않는다.

## 0. AI 작업 지시용 핵심 규칙

이 파일은 다른 개발자뿐 아니라 AI에게 그대로 전달해도 되는 실행 기준 문서로 사용한다. AI에게 이 프로젝트의 후속 작업을 맡길 때에는, 특별한 추가 지시가 없는 한 반드시 아래 규칙을 우선 적용한다.

첫째, 후속 작업의 기준 모델은 `topk15 window60 SAGE` 하나이다. 기준 그래프는 `data/graph_relflag_edge_t15_w60_thr075/graph_rur_custom2.pt`이고, 기준 실험 폴더는 `experiments/relflag_edge_t15_w60_thr075_sage_inverse_seed42`이다. 비슷한 이름의 `data/graph/graph_rur_custom2.pt`는 기존 SAGE baseline용 그래프이므로, 이 문서의 작업 대상이 아니다.

둘째, 기존 SAGE baseline, LightGBM node-only, stacking 보정 모델은 비교 또는 보조 실험 산출물일 뿐이다. 사용자가 명시적으로 “baseline과 비교하라”, “LightGBM을 붙이라”, “stacking을 실험하라”고 요청하지 않는 한, AI는 해당 산출물을 기준 모델처럼 사용하거나 그쪽 파이프라인으로 작업 범위를 넓히면 안 된다.

셋째, 기존 최종 산출물은 원칙적으로 덮어쓰지 않는다. 재학습, 미세 조정, 추가 실험을 수행할 때에는 `experiments/relflag_edge_t15_w60_thr075_sage_inverse_seed42_rerun`처럼 새 output directory를 만들어 결과를 저장한다. 사용자가 명시적으로 기존 폴더 갱신을 요청한 경우에만 `experiments/relflag_edge_t15_w60_thr075_sage_inverse_seed42`를 직접 덮어쓴다.

넷째, 사용자가 샘플링 또는 그래프 재구성을 명시적으로 요청하지 않는 한, `Sampling.py`, `BuildTextEmbeddings.py`, `BuildEdges.py`, `BuildGraphDataset.py`를 다시 실행하지 않는다. 현재 topk15 window60 SAGE 작업은 이미 생성된 PyTorch Geometric 그래프 파일을 읽어 학습하거나 예측을 이어가는 작업이다. 샘플링을 다시 수행하면 노드 집합, split, fake rate, edge 수가 달라져 기존 성능과 직접 비교하기 어려워진다.

다섯째, 모델 평가를 보고할 때에는 `metrics.json`의 train, valid, test 성능을 분리해서 제시한다. 최종 비교는 test 성능으로 하되, threshold와 best epoch가 validation 기준으로 선택되었다는 점을 함께 설명한다. test 결과를 기준으로 threshold를 다시 고르는 방식은 사용하지 않는다.

여섯째, 정보 누수와 미래 정보 침범을 만들 수 있는 수정은 피한다. 이 프로젝트의 기준 그래프는 이미 train mask 기준 scaling과 split mask를 포함한다. 새 피처를 추가하거나 그래프를 다시 만들라는 별도 지시가 없다면, `graph_rur_custom2.pt` 내부의 `x`, `y`, `edge_index`, `train_mask`, `valid_mask`, `test_mask`를 그대로 사용한다.

일곱째, 정확한 재현을 목표로 할 때 XPU 사용이 불가능하면 임의로 CPU 실행으로 바꿔 진행하지 말고 사용자에게 알려야 한다. 단순 코드 점검이나 빠른 smoke test가 목적이라고 사용자가 명시한 경우에는 `--device cpu`를 사용할 수 있다.

AI에게 작업을 맡길 때 사용할 수 있는 최소 지시문은 다음과 같다.

```text
Run_Topk15_Window60_SAGE.md를 기준으로 현재 프로젝트의 topk15 window60 SAGE 모델만 사용해 작업하라.
기준 그래프는 data/graph_relflag_edge_t15_w60_thr075/graph_rur_custom2.pt이고,
기준 실험 폴더는 experiments/relflag_edge_t15_w60_thr075_sage_inverse_seed42이다.
별도 지시가 없는 한 baseline, LightGBM, stacking 산출물은 사용하지 말고,
샘플링이나 그래프 재구성도 수행하지 마라.
기존 최종 산출물은 덮어쓰지 말고 새 output directory에 저장하라.
성능 보고 시 train/valid/test를 구분하고, threshold는 validation 기준으로 선택된 값을 사용하라.
```

## 1. 이 문서에서 말하는 모델

이 프로젝트에는 여러 실험 산출물이 남아 있지만, 여기서 실행해야 하는 모델은 다음 하나이다.

```text
experiments/relflag_edge_t15_w60_thr075_sage_inverse_seed42
```

이 모델은 `plain SAGE` 모델이다. 즉 relation별로 다른 가중치를 두는 `relation_sage`가 아니라, 하나의 `edge_index` 위에서 GraphSAGE 메시지 전달을 수행한다. 그래프 파일 안에는 R-U-R 엣지와 shock 엣지가 함께 들어 있지만, plain SAGE는 `edge_type` 자체를 구분해서 다른 파라미터를 쓰지는 않는다. 다만 두 종류의 엣지가 모두 `edge_index`에 포함되어 있으므로, 메시지 전달 대상 이웃에는 두 관계가 모두 반영된다.

모델 이름의 `topk15 window60`은 shock 엣지 구성 방식에서 온 표현이다. 평점 변동 shock 후보를 만들 때 60일 시간 창을 사용하고, 후보별로 최대 15개 수준의 관련 노드를 연결한 설정이다.

## 2. 반드시 사용해야 하는 파일

이 모델을 그대로 이어서 사용하려면 아래 파일과 폴더를 기준으로 작업해야 한다.

```text
data/graph_relflag_edge_t15_w60_thr075/graph_rur_custom2.pt
experiments/relflag_edge_t15_w60_thr075_sage_inverse_seed42/best_model.pt
experiments/relflag_edge_t15_w60_thr075_sage_inverse_seed42/metrics.json
experiments/relflag_edge_t15_w60_thr075_sage_inverse_seed42/predictions_all.csv
experiments/relflag_edge_t15_w60_thr075_sage_inverse_seed42/prediction_test.csv
```

가장 중요한 입력 그래프는 다음 파일이다.

```text
data/graph_relflag_edge_t15_w60_thr075/graph_rur_custom2.pt
```

비슷한 이름의 아래 파일과 혼동하면 안 된다.

```text
data/graph/graph_rur_custom2.pt
```

`data/graph/graph_rur_custom2.pt`는 기존 SAGE baseline용 그래프이며, 이 문서의 topk15 window60 모델과 샘플링, 피처 차원, 엣지 구성이 다르다.

## 3. 현재 저장된 모델 결과를 바로 사용하는 방법

이미 학습된 모델의 예측 결과를 바로 확인하거나 후속 분석에 사용하려면 재학습할 필요가 없다. 다음 파일을 읽으면 된다.

```text
experiments/relflag_edge_t15_w60_thr075_sage_inverse_seed42/predictions_all.csv
```

이 파일에는 전체 노드에 대한 예측이 들어 있다. 주요 열은 다음과 같다.

```text
sampled_node_idx : 샘플링된 그래프 안에서의 노드 번호
y_true           : 실제 라벨. 0은 정상, 1은 어뷰징
prob_fake        : SAGE 모델이 예측한 어뷰징 확률
pred_label       : threshold를 적용한 최종 예측 라벨
split            : train, valid, test 구분
is_target_node   : 평가 대상 노드 여부
```

test split만 보고 싶으면 다음 파일을 사용하면 된다.

```text
experiments/relflag_edge_t15_w60_thr075_sage_inverse_seed42/prediction_test.csv
```

현재 저장된 최종 threshold는 `0.75`이다. 즉 `prob_fake >= 0.75`이면 어뷰징으로 예측한 것이다. 이 값은 test를 보고 정한 것이 아니라 validation split에서 `prevalence_constrained_macro_f1` 기준으로 선택된 값이다.

## 4. 같은 설정으로 다시 학습하는 방법

현재 프로젝트에는 별도의 inference-only 스크립트가 아니라, 학습 후 예측 파일까지 저장하는 `TrainGNN.py`가 있다. 같은 설정으로 모델을 다시 학습하려면 PowerShell에서 프로젝트 루트로 이동한 뒤 아래 명령을 실행한다.

```powershell
cd C:\Users\LSH\Downloads\Code\ITDA_KJM

C:\Users\LSH\Downloads\ITDA\.venv\Scripts\python.exe .\TrainGNN.py `
  --graph-path .\data\graph_relflag_edge_t15_w60_thr075\graph_rur_custom2.pt `
  --model sage `
  --output-dir .\experiments\relflag_edge_t15_w60_thr075_sage_inverse_seed42_rerun `
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

위 명령은 기존 최종 산출물을 덮어쓰지 않도록 `_rerun` 폴더에 결과를 저장한다. 기존 최종 산출물을 의도적으로 갱신하려면 `--output-dir`을 아래처럼 바꾸면 된다.

```text
.\experiments\relflag_edge_t15_w60_thr075_sage_inverse_seed42
```

다만 다른 개발자가 결과 비교를 해야 하는 상황에서는 기존 폴더를 바로 덮어쓰기보다 `_rerun`처럼 새 폴더를 쓰는 편이 안전하다.

## 5. 실행 전 환경 확인

현재 실험은 Intel XPU 버전 PyTorch 환경에서 실행한 설정이다. 실행 전에 다음 명령으로 PyTorch와 XPU 사용 가능 여부를 확인한다.

```powershell
C:\Users\LSH\Downloads\ITDA\.venv\Scripts\python.exe -c "import torch; print(torch.__version__); print(hasattr(torch, 'xpu') and torch.xpu.is_available())"
```

마지막 출력이 `True`이면 `--device xpu`로 실행할 수 있다. 만약 다른 개발자의 환경에서 XPU가 없고 단순 재현이 아니라 코드 확인이나 CPU 테스트만 목적이라면 `--device cpu`로 바꾸어 실행할 수 있다. 단, 이 프로젝트에서 저장된 최종 설정값은 `xpu` 기준이다.

## 6. 실행 후 생성되는 파일

학습이 끝나면 지정한 output directory에 다음 파일들이 생성된다.

```text
best_model.pt        : validation 기준으로 선택된 최종 SAGE 모델 가중치
config_used.json     : 실제 실행 설정
metrics.json         : train, valid, test 성능 요약
predictions_all.csv  : 전체 split 예측 결과
prediction_test.csv  : test split 예측 결과
training_log.csv     : epoch별 학습 로그
```

정상적으로 재현되었는지 빠르게 확인하려면 `metrics.json`에서 아래 값들이 현재 저장된 기준과 크게 다르지 않은지 보면 된다. XPU, PyTorch 버전, 난수 처리 차이 때문에 소폭 변동은 가능하다.

```text
best_epoch        : 71 근처
best_threshold    : 0.75 근처
valid PR-AUC      : 약 0.3940
valid Macro F1    : 약 0.6458
test PR-AUC       : 약 0.4901
test Macro F1     : 약 0.6793
test ROC-AUC      : 약 0.7851
```

현재 저장된 기준 성능은 다음과 같다.

```text
Test PR-AUC   = 0.4900669033
Test ROC-AUC  = 0.7851307169
Test Macro F1 = 0.6792789764
Precision     = 0.4956369983
Recall        = 0.4965034965
Accuracy      = 0.7839355926
```

## 7. 그래프와 샘플 구성 요약

이 모델의 입력 그래프 요약은 다음과 같다.

```text
노드 수                 : 26,701
전체 피처 차원           : 174
숫자 피처 차원           : 46
텍스트 SVD 피처 차원      : 128
전체 directed edge 수    : 15,304
R-U-R edge 수            : 11,038
weak product shock edge 수: 4,266
```

split 구성은 다음과 같다.

```text
train : 17,088 nodes, fake rate 0.1388
valid :  4,272 nodes, fake rate 0.1798
test  :  5,341 nodes, fake rate 0.2142
```

이 split은 이미 그래프 파일 내부의 mask로 저장되어 있다. 따라서 `TrainGNN.py`를 실행할 때 별도로 train/valid/test를 다시 나누지 않는다.

## 8. 후속 작업 시 주의사항

후속 작업에서 topk15 window60 SAGE만 사용하려면 아래 경로만 기준으로 삼는다.

```text
data/graph_relflag_edge_t15_w60_thr075/
experiments/relflag_edge_t15_w60_thr075_sage_inverse_seed42/
```

아래 경로들은 다른 실험이므로, 이 모델을 이어서 사용할 때 기본값처럼 사용하면 안 된다.

```text
data/graph/
experiments/baseline_original_rur_custom2_sage_prevalence_seed42/
experiments/lightgbm_node_only_*
experiments/stacking_sage_lgbm_*
```

LightGBM이나 stacking 모델은 보조 실험 산출물이다. 현재 문서의 목적이 topk15 window60 SAGE 모델만 실행하고 이어서 작업하는 것이라면, `TrainNodeLightGBM.py`나 `TrainStackingCalibrator.py`를 실행할 필요가 없다.
