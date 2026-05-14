# Processed YelpZip Node Data

이 폴더는 `Preprocess.py`가 생성한 리뷰 노드 중심 전처리 산출물이다.

- `review_nodes.csv.gz`: 리뷰 노드 메타데이터, 라벨, 숫자 파생 피처. 원문 텍스트는 제외.
- `review_text.csv.gz`: `node_idx`, `review_id`, 원문 `text`. 향후 텍스트 임베딩과 대시보드 증거용.
- `relation_candidate_keys.csv.gz`: 아직 엣지는 만들지 않고, 다음 단계에서 relation을 만들기 위한 후보 키만 저장.
- `node_features_numeric.npy`: raw ID와 라벨 누수 정보를 제외한 숫자형 노드 피처 행렬.
- `node_labels.npy`: `is_fake` 타겟 배열.
- `node_review_ids.npy`: 행렬 행과 원본 리뷰 ID의 매핑.
- `review_node_graph_no_edges.pt`: PyTorch Geometric `Data` 객체. `x`, `y`, 빈 `edge_index`만 포함.
- `feature_columns.json`: 숫자 피처 컬럼 순서와 제외한 정보의 이유.
- `preprocess_summary.json`: 전처리 품질과 규모 요약.

주의: 이 단계에서는 샘플링과 엣지 연결을 수행하지 않는다.
다음 단계 순서는 샘플링 -> 그래프 네트워크 설계 -> GNN 모델링 및 최적화이다.
