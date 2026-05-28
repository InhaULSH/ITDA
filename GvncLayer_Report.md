# Governance Layer Streamlit Development Guide

## 1. 목적

이 문서는 Streamlit 대시보드 개발자가 governance layer만 보고도 모델 거버넌스 탭을 구현할 수 있도록 작성한 개발 지침이다. 리뷰 단위 설명, 캠페인 네트워크, 캠페인 evidence 화면은 다른 개발자가 별도로 구현하므로 이 문서의 범위에서 제외한다.

거버넌스 탭의 목표는 다음 세 가지이다.

1. 현재 운영 중인 모델과 threshold를 명확하게 보여준다.
2. 모델 성능, 데이터 계보, artifact 무결성, 접근 권한을 추적한다.
3. threshold 변경, 조회, export 같은 운영 행위를 감사 가능한 형태로 기록한다.

## 2. 현재 모델

| 항목 | 값 |
|---|---|
| model id | `campaign_quality_q60_relation_sage_mlp_equal_seed42` |
| model type | `relation_sage_mlp` |
| graph | `data/graph_campaign_quality_q60_top3_b6000_s020/graph_rur_custom2.pt` |
| operating threshold | 0.794 |
| validation selected threshold | 0.78 |
| governance layer | `data/governance_layer/` |

운영 threshold 0.794는 모델 가중치를 바꾸는 값이 아니라, 저장된 `prob_fake`를 어뷰징/정상 판정으로 변환하는 기준이다.

현행 모델은 relation-aware SAGE와 Self Branch를 결합한 구조이다. Streamlit은 모델을 다시 학습하거나 추론하지 않고, 이미 생성된 예측 결과와 governance layer를 읽어서 현재 운영 모델의 상태를 설명한다. 따라서 UI에서는 `relation_sage_mlp`를 “관계별 메시지와 리뷰 자체 피처 판단을 함께 쓰는 최종 모델”로 설명하면 된다.

| 구조 요소 | 대시보드 설명 |
|---|---|
| R-U-R | 같은 사용자의 인접 리뷰 흐름 |
| Filtered Campaign Pair | 상품-주-평점방향 단위의 캠페인성 리뷰 쌍 |
| Weak Product Shock | 상품 과거 평점 흐름 대비 이례적인 평점 이동 |
| Self Branch | 연결이 부족한 리뷰도 자기 피처로 안정적으로 판단하는 보조 경로 |
| relation aggregation | 세 relation의 메시지를 동일 비중으로 결합하는 equal 방식 |
| relation dropout | 0.0, 보수적으로 선별한 relation의 해석 일관성을 유지하기 위해 미적용 |

## 3. 파일 구조

Streamlit 앱은 다음 파일을 읽는다.

```text
data/governance_layer/
  governance_manifest.json
  model_card.json
  model_metrics.csv
  threshold_policy.csv
  data_lineage.csv
  artifact_registry.csv
  access_policy.csv
```

각 파일의 역할은 다음과 같다.

| 파일 | 형식 | 역할 |
|---|---|---|
| `governance_manifest.json` | JSON | governance layer 진입점, 버전, 파일 목록, Streamlit contract |
| `model_card.json` | JSON | 모델 구조, threshold, 성능, 그래프 요약, 누수 방지 정책 |
| `model_metrics.csv` | CSV | train/valid/test 성능 표 |
| `threshold_policy.csv` | CSV | threshold 선택 및 변경 원칙 |
| `data_lineage.csv` | CSV | raw data부터 governance layer까지의 데이터 흐름 |
| `artifact_registry.csv` | CSV | 주요 artifact 존재 여부, 크기, SHA-256 hash |
| `access_policy.csv` | CSV | viewer/reviewer/admin 권한 정책 |

## 4. Streamlit 기본 로딩 코드

다음 코드를 governance tab의 데이터 로딩 모듈로 사용할 수 있다.

```python
from pathlib import Path
import json
import pandas as pd
import streamlit as st

GOV_DIR = Path("data/governance_layer")


@st.cache_data(show_spinner=False)
def load_json_file(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_data(show_spinner=False)
def load_csv_file(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


@st.cache_data(show_spinner=False)
def load_governance_layer(base_dir: str = str(GOV_DIR)) -> dict:
    base = Path(base_dir)
    manifest = load_json_file(str(base / "governance_manifest.json"))
    model_card = load_json_file(str(base / manifest["files"]["model_card"]))
    metrics = load_csv_file(str(base / manifest["files"]["model_metrics"]))
    threshold_policy = load_csv_file(str(base / manifest["files"]["threshold_policy"]))
    lineage = load_csv_file(str(base / manifest["files"]["data_lineage"]))
    artifacts = load_csv_file(str(base / manifest["files"]["artifact_registry"]))
    access = load_csv_file(str(base / manifest["files"]["access_policy"]))
    return {
        "manifest": manifest,
        "model_card": model_card,
        "metrics": metrics,
        "threshold_policy": threshold_policy,
        "lineage": lineage,
        "artifacts": artifacts,
        "access": access,
    }
```

Streamlit 공식 문서에서는 데이터 계산 결과에는 `st.cache_data`, DB 연결이나 전역 리소스에는 `st.cache_resource`를 쓰는 구조를 권장한다. governance layer의 CSV/JSON은 읽기 전용 데이터이므로 `st.cache_data`가 적합하다.

## 5. Governance 탭 화면 구성

추천 화면 구조는 다음과 같다.

```text
Governance
  1. Model Status
  2. Performance Summary
  3. Threshold Policy
  4. Data Lineage
  5. Artifact Integrity
  6. Access Policy
  7. Audit Log
  8. Admin Actions
```

## 6. Model Status 섹션

`model_card.json`과 `governance_manifest.json`을 사용한다.

표시할 항목:

| UI 요소 | 데이터 |
|---|---|
| 모델 ID 카드 | `manifest.model_id` |
| 레이어 버전 카드 | `manifest.governance_layer_version` |
| 운영 threshold 카드 | `manifest.operating_threshold` |
| 모델 구조 | `model_card.model_type` |
| best epoch | `model_card.training.best_epoch` |
| graph 요약 | `model_card.graph.n_nodes`, `total_directed_edges`, `overall_isolated_nodes` |
| relation별 edge 수 | `model_card.graph.relation_edge_counts`, `model_card.graph.relation_stats` |

예시 코드:

```python
gov = load_governance_layer()
manifest = gov["manifest"]
card = gov["model_card"]

cols = st.columns(4)
cols[0].metric("Model ID", manifest["model_id"])
cols[1].metric("Threshold", manifest["operating_threshold"])
cols[2].metric("Nodes", card["graph"]["n_nodes"])
cols[3].metric("Edges", card["graph"]["total_directed_edges"])
```

relation별 edge 수는 작은 표로 함께 보여주는 것이 좋다. 이 표는 모델이 어떤 구조적 근거를 갖고 있는지 설명하는 데 쓰이며, 운영자가 개별 리뷰나 캠페인 화면으로 이동하기 전에 현재 모델의 relation 구성을 이해하도록 돕는다.

```python
relation_names = {
    "0": "R-U-R",
    "1": "Filtered Campaign Pair",
    "2": "Weak Product Shock",
}
relation_rows = []
for rel_id, count in card["graph"]["relation_edge_counts"].items():
    stats = card["graph"]["relation_stats"].get(rel_id, {})
    relation_rows.append({
        "relation": relation_names.get(rel_id, rel_id),
        "directed_edges": count,
        "mean_degree": stats.get("mean_degree"),
        "isolated_nodes": stats.get("isolated_nodes"),
    })
st.dataframe(pd.DataFrame(relation_rows), use_container_width=True)
```

## 7. Performance Summary 섹션

`model_metrics.csv`를 사용한다. governance/admin 화면에서는 valid와 test 지표를 나란히 볼 수 있지만, test 지표는 최종 보고용이라는 표시를 붙인다. 일반 운영 화면에서는 valid 중심 요약을 기본으로 두고, test 세부 지표는 접어두거나 관리자 권한에서만 노출하는 편이 안전하다.

표시 지표:

| 지표 | 의미 |
|---|---|
| PR-AUC | 클래스 불균형 상황에서 어뷰징 리뷰를 상위에 올리는 순위 성능 |
| ROC-AUC | 전체 ranking 분리 성능 |
| Macro F1 | 정상/어뷰징 양쪽 클래스를 균형 있게 본 threshold 기반 성능 |
| Precision | 어뷰징으로 예측한 리뷰 중 실제 어뷰징 비율 |
| Recall | 실제 어뷰징 리뷰 중 모델이 잡아낸 비율 |
| Accuracy | 전체 정답률 |

주의: test metric은 최종 보고용이다. threshold 변경 의사결정은 validation 기준으로 설명해야 한다. `model_card.json`의 `streamlit_usage.hide_eval_columns_by_default`가 `true`이므로, 대시보드 구현 시 평가 전용 정보가 일반 운영 판단에 섞이지 않도록 기본 표시 범위를 조절한다.

예시 코드:

```python
metrics = gov["metrics"]
st.dataframe(
    metrics[["split", "threshold", "pr_auc", "roc_auc", "macro_f1", "precision", "recall", "accuracy"]],
    use_container_width=True,
)
```

현재 핵심 운영 지표는 test PR-AUC 0.5288, ROC-AUC 0.8051, Macro F1 0.6962, Precision 0.5273, Recall 0.5149이다. 이 값은 모델의 최종 보고 성능을 보여주는 용도이며, threshold를 다시 고르는 근거로 사용하지 않는다.

## 8. Threshold Policy 섹션

`threshold_policy.csv`를 사용한다.

이 섹션은 운영자가 threshold의 의미를 오해하지 않도록 설계해야 한다. slider는 simulation일 뿐이고, 실제 threshold 변경은 admin 승인과 로그 기록 이후에만 반영한다.

권장 UI:

1. 현재 운영 threshold 표시
2. validation-selected threshold와 operating threshold 차이 설명
3. threshold 변경 정책 표
4. admin 전용 변경 요청 폼

`validation_selected_threshold` 0.78은 validation 기준 threshold 탐색 과정에서 선택된 값이고, `operating_threshold` 0.794는 이후 운영상 precision을 조금 더 중시해 고정한 기준이다. 둘 다 모델 가중치를 바꾸는 값은 아니며, 저장된 예측 확률을 어떤 기준으로 의심 후보로 표시할지 정하는 후처리 정책이다.

admin 변경 요청 폼 필드:

| 필드 | 설명 |
|---|---|
| requested_threshold | 새 threshold |
| reason | 변경 사유 |
| expected_effect | Precision/Recall/검토량 변화 예상 |
| approver | 승인자 |
| created_at | 생성 시각 |

실제 운영에서는 변경 요청을 바로 모델 파일에 쓰지 말고, 먼저 `threshold_change_log`에 저장한다. 이후 승인된 변경만 `ApplyOperatingThreshold.py`와 `BuildGovernanceLayer.py` 재실행으로 반영한다.

## 9. Data Lineage 섹션

`data_lineage.csv`를 사용한다. Streamlit에서는 단계별 table 또는 timeline 형태가 적합하다.

표시 목적:

1. 원천 데이터부터 모델 결과까지의 재현 경로를 보여준다.
2. 어떤 산출물이 어떤 단계에서 만들어졌는지 확인한다.
3. 발표나 감사 상황에서 “이 모델이 어떤 데이터 흐름으로 만들어졌는가”를 설명한다.

예시 코드:

```python
lineage = gov["lineage"].sort_values("stage_order")
st.dataframe(lineage, use_container_width=True)
```

## 10. Artifact Integrity 섹션

`artifact_registry.csv`를 사용한다. 이 파일은 모델 운영에 필요한 핵심 artifact의 존재 여부, 파일 크기, SHA-256 hash를 담는다.

권장 UI:

| 표시 | 조건 |
|---|---|
| 정상 | `exists == True` |
| 경고 | 파일은 있지만 hash가 governance layer 생성 시점과 다름 |
| 오류 | `exists == False` |

현재 `artifact_registry.csv`는 governance layer 생성 시점의 hash를 저장한다. Streamlit에서 실시간 hash 재계산을 추가하면 artifact 변조나 잘못된 파일 교체를 감지할 수 있다.

실시간 검증 예시:

```python
import hashlib


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
```

큰 파일의 hash 계산은 비용이 있으므로, 기본 화면에서는 저장된 registry를 보여주고 admin이 “무결성 재검사” 버튼을 눌렀을 때만 재계산하는 편이 낫다.

## 11. Access Policy 섹션

`access_policy.csv`를 사용한다.

권장 권한:

| role | 설명 |
|---|---|
| viewer | 발표/읽기 전용 사용자 |
| reviewer | 운영 검토자, governance-safe export 가능 |
| admin | threshold 변경 승인, eval-only 정보 접근 가능 |

Streamlit은 기본적으로 강한 권한 관리 시스템이 아니므로, 실제 배포에서는 사내 인증 프록시, Streamlit Community Cloud secrets, 또는 별도 인증 계층과 결합해야 한다. 데모 환경에서는 `st.session_state["role"]`로 role을 관리할 수 있다.

간단한 role gate 예시:

```python
def has_permission(access_df: pd.DataFrame, role: str, permission: str) -> bool:
    row = access_df.loc[access_df["role"].eq(role)]
    if row.empty or permission not in row.columns:
        return False
    return bool(row.iloc[0][permission])


role = st.session_state.get("role", "viewer")
if has_permission(gov["access"], role, "can_change_threshold"):
    st.button("Approve threshold change")
```

## 12. 운영 상태 DB

governance layer의 CSV/JSON은 읽기 전용으로 유지한다. 사용자가 남기는 운영 상태는 별도 SQLite DB에 저장한다.

권장 경로:

```text
ops/governance_ops.db
```

권장 테이블:

```sql
CREATE TABLE IF NOT EXISTS threshold_change_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    model_id TEXT NOT NULL,
    old_threshold REAL NOT NULL,
    requested_threshold REAL NOT NULL,
    status TEXT NOT NULL,
    reason TEXT,
    requested_by TEXT,
    approved_by TEXT,
    created_at TEXT NOT NULL,
    approved_at TEXT
);

CREATE TABLE IF NOT EXISTS audit_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    event_time TEXT NOT NULL,
    actor TEXT,
    role TEXT,
    event_type TEXT NOT NULL,
    model_id TEXT,
    object_type TEXT,
    object_id TEXT,
    detail_json TEXT
);

CREATE TABLE IF NOT EXISTS model_review_notes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    model_id TEXT NOT NULL,
    author TEXT,
    note_type TEXT,
    note TEXT,
    created_at TEXT NOT NULL
);
```

Streamlit에서 SQLite 연결은 `st.cache_resource`로 관리한다.

```python
import sqlite3


@st.cache_resource
def get_ops_connection(path: str = "ops/governance_ops.db"):
    return sqlite3.connect(path, check_same_thread=False)
```

## 13. Audit Log

OWASP Logging Cheat Sheet는 보안 관련 이벤트, 접근 실패, 중요한 운영 변경을 기록하고 로그 무결성과 접근 통제를 고려하라고 권고한다. 이 프로젝트에서는 최소한 다음 이벤트를 기록한다.

| event_type | 기록 시점 |
|---|---|
| `view_governance_tab` | 사용자가 governance tab을 열 때 |
| `export_governance_report` | governance 정보를 export할 때 |
| `request_threshold_change` | threshold 변경 요청 생성 |
| `approve_threshold_change` | admin 승인 |
| `integrity_check_run` | artifact hash 재검사 |
| `access_denied` | 권한 없는 기능 접근 시도 |

로그에는 raw review text나 불필요한 개인정보를 넣지 않는다. 모델 ID, 객체 ID, 이벤트 타입, actor, role, 시간, 요약 JSON 정도만 저장한다.

## 14. Threshold 변경 workflow

권장 workflow는 다음과 같다.

```text
1. reviewer/admin이 threshold simulation을 확인한다.
2. admin이 변경 요청을 생성한다.
3. threshold_change_log에 pending 상태로 저장한다.
4. 승인 후 ApplyOperatingThreshold.py를 실행한다.
5. BuildGovernanceLayer.py를 실행한다.
6. Streamlit cache를 clear하거나 layer version 변경을 감지해 reload한다.
7. audit_log에 변경 완료 이벤트를 남긴다.
```

로컬 데모에서는 Streamlit에서 subprocess로 스크립트를 실행할 수 있지만, 실제 운영에서는 별도 job runner 또는 배포 파이프라인에서 실행하는 편이 안전하다. 대시보드 서버가 모델 artifact를 직접 수정하는 구조는 권한과 장애 위험이 크기 때문이다.

## 15. 캐시 전략

권장 cache key:

```text
governance_layer_version + model_id + operating_threshold
```

`governance_manifest.json`의 `governance_layer_version`이 바뀌면 Streamlit cache를 무효화해야 한다.

```python
version = gov["manifest"]["governance_layer_version"]
st.caption(f"Governance layer version: {version}")
```

## 16. 데이터 아키텍처

권장 구조는 다음과 같다.

```text
model artifacts: read-only
  experiments/campaign_quality_q60_relation_sage_mlp_equal_seed42/
  data/graph_campaign_quality_q60_top3_b6000_s020/
  data/edges_campaign_quality_q60_top3_b6000_s020/

governance mart: read-only
  data/governance_layer/

operational state: mutable
  ops/governance_ops.db

Streamlit app
  reads governance mart with st.cache_data
  writes only operational state DB
```

이 구조의 장점은 모델 산출물과 운영 상태가 섞이지 않는다는 것이다. 모델을 재학습하거나 threshold를 승인 변경할 때만 governance mart를 재생성하고, 평소 대시보드는 읽기 전용 governance layer를 빠르게 조회한다.

향후 에이전틱 AI 기반 검수 보조 시스템으로 확장하더라도 같은 원칙을 유지한다. AI 에이전트는 governance mart와 모델 예측 결과를 읽어 검수 보고서나 변경 제안을 만들 수 있지만, threshold 변경이나 artifact 교체 같은 운영 행위는 `ops/governance_ops.db`에 요청과 승인 로그를 남긴 뒤 별도 승인 절차로 처리해야 한다.

## 17. Parquet와 DuckDB 확장안

현재 governance layer는 파일이 작으므로 CSV/JSON으로 충분하다. 다만 리뷰/캠페인 레이어처럼 행 수가 큰 테이블은 Parquet + DuckDB가 더 적합하다. Apache Parquet는 column-oriented file format이고, DuckDB는 Parquet를 직접 조회하며 filter pushdown을 지원한다. 따라서 다른 개발자가 만든 리뷰/캠페인 레이어와 결합할 때는 다음 구조를 권장한다.

```text
governance layer: CSV/JSON
large review/campaign layers: Parquet
query engine: DuckDB
Streamlit: selected object only query
```

governance tab 자체는 작은 파일 중심이므로 pandas read_csv/read_json으로 충분하다.

## 18. 배포 전 체크리스트

| 체크 | 기준 |
|---|---|
| governance files 존재 | 7개 파일 모두 존재 |
| model_id 일치 | manifest, model_card, experiment dir가 동일 |
| threshold 일치 | manifest, model_card, metrics threshold가 0.794 |
| relation 요약 | R-U-R 11,038개, Filtered Campaign Pair 6,000개, Weak Product Shock 4,266개 |
| artifact integrity | 핵심 파일 exists True |
| role gate | viewer/reviewer/admin 권한 분기 확인 |
| audit log | threshold 요청, export, 접근 실패 기록 |
| eval-only 통제 | test 지표는 보고용임을 표시하고 threshold 선택 근거로 사용하지 않음 |
| cache invalidation | governance_layer_version 변경 시 reload |

## 19. 참고 자료

- Streamlit caching: https://docs.streamlit.io/develop/concepts/architecture/caching
- Apache Parquet documentation: https://parquet.apache.org/docs/
- DuckDB Parquet documentation: https://duckdb.org/docs/current/data/parquet/overview.html
- OWASP Logging Cheat Sheet: https://cheatsheetseries.owasp.org/cheatsheets/Logging_Cheat_Sheet.html
