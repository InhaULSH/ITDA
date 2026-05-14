# Sampled YelpZip Review Nodes

이 폴더는 `Sampling.py`가 생성한 상품-주 기반 서브그래프 샘플링 산출물이다.

- `sampled_review_nodes.csv.gz`: 선택된 리뷰 노드 메타데이터와 샘플링용 `sampled_node_idx`.
- `sampled_review_text.csv.gz`: 선택된 리뷰의 원문 텍스트.
- `sampled_relation_candidate_keys.csv.gz`: 선택된 노드의 relation 후보 키. 아직 엣지는 만들지 않았다.
- `sampled_node_features_numeric.npy`: 선택된 노드의 숫자형 피처 행렬.
- `sampled_node_labels.npy`: 선택된 노드의 라벨 배열.
- `sampled_node_review_ids.npy`: 선택된 노드 행과 원본 `review_id`의 매핑.
- `sampled_review_node_graph_no_edges.pt`: PyTorch Geometric `Data` 객체. `edge_index`는 빈 텐서이다.
- `sampled_node_mapping.csv`: `sampled_node_idx`, `original_node_idx`, `review_id` 연결표.
- `product_week_sampling_units.csv.gz`: 전체 상품-주 후보의 flag와 선택 여부.
- `sampling_summary.json`: 샘플링 기준과 결과 요약.

주의: 다음 단계에서 엣지를 만들 때는 `sampled_node_idx`를 기준으로 `edge_index`를 구성해야 한다.
