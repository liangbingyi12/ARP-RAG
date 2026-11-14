# -*- coding: utf-8 -*-
"""
工业级 GraphRAG 子图抽取 + 路径向量剪枝（批处理优化版）
- 返回节点除 embedding 外的所有属性
- 返回关系除 embedding 外的所有属性
- 无向边去重（基于节点对和关系类型，忽略属性差异）
- 兼容 Neo4j 5.x
- 新增：根据多跳问题对路径进行逐层向量剪枝（每层保留 top-2 关系）
- 优化：使用批量 embedding 接口，减少 API 调用次数
"""

from neo4j import GraphDatabase
from collections import deque
import json
import openai
import numpy as np
from typing import Dict, List, Set, Tuple

# ================== Neo4j 配置 ==================
import config
from config import (
    BFS_NEO4J_URI as NEO4J_URI,
    RETRIEVAL_NEO4J_USERNAME as NEO4J_USERNAME,
    RETRIEVAL_NEO4J_PASSWORD as NEO4J_PASSWORD,
    RETRIEVAL_OPENAI_API_KEY as OPENAI_API_KEY,
    RETRIEVAL_OPENAI_BASE_URL as OPENAI_BASE_URL,
    RETRIEVAL_EMBEDDING_MODEL as EMBEDDING_MODEL,
    RETRIEVAL_EMBEDDING_DIMENSIONS as EMBEDDING_DIMENSIONS,
)



driver = GraphDatabase.driver(
    NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD)
)

# ================== OpenAI 配置（用于向量剪枝） ==================





openai_client = openai.OpenAI(
    api_key=OPENAI_API_KEY,
    base_url=OPENAI_BASE_URL
)


def get_embedding(text: str) -> List[float]:
    """获取单个文本的向量表示（降级用）"""
    try:
        response = openai_client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=text,
            dimensions=EMBEDDING_DIMENSIONS
        )
        return response.data[0].embedding
    except Exception as e:
        print(f"❌ 获取 embedding 失败: {e}")
        return [0.0] * EMBEDDING_DIMENSIONS


def batch_get_embeddings(texts: List[str]) -> List[List[float]]:
    """
    批量获取多个文本的 embedding
    如果批量请求失败，降级为逐个调用 get_embedding
    """
    if not texts:
        return []
    try:
        response = openai_client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=texts,
            dimensions=EMBEDDING_DIMENSIONS
        )
        # 保持输入顺序
        return [data.embedding for data in response.data]
    except Exception as e:
        print(f"❌ 批量获取 embedding 失败，降级为逐个处理: {e}")
        return [get_embedding(t) for t in texts]


def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """计算余弦相似度"""
    v1 = np.array(vec1)
    v2 = np.array(vec2)
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return float(np.dot(v1, v2) / (norm1 * norm2))


# ================== 邻居查询（返回节点和关系的完整属性，Python 中移除 embedding） ==================

def get_neighbors(tx, node_id: str):
    """
    获取一个节点的所有邻居，返回邻居的 elementId、labels、所有属性（不含 embedding），
    以及关系的类型、所有属性（不含 embedding）、边的两端。
    """
    query = """
    MATCH (n)-[r]-(m)
    WHERE elementId(n) = $id
    RETURN elementId(m) AS nid,
           labels(m) AS labels,
           m AS node,
           type(r) AS rel_type,
           properties(r) AS rel_props_raw,
           elementId(n) AS src,
           elementId(m) AS dst
    LIMIT $neighbor_limit
    """
    result = tx.run(query, id=node_id, neighbor_limit=config.BFS_NEIGHBOR_LIMIT)
    records = []
    for record in result:
        data = record.data()
        # 处理节点属性（排除 embedding）
        node_obj = data["node"]
        node_props = dict(node_obj.items())
        node_props.pop("embedding", None)
        data["props"] = node_props
        del data["node"]
        # 处理关系属性：从 rel_props_raw 中移除 embedding
        rel_props = data["rel_props_raw"]
        rel_props.pop("embedding", None)
        data["rel_props"] = rel_props
        del data["rel_props_raw"]
        records.append(data)
    return records


def get_entity_vector_index_name(session) -> str:
    """Return an Entity embedding vector index name if Neo4j has one."""
    try:
        record = session.run(
            """
            SHOW INDEXES YIELD name, type, labelsOrTypes, properties
            WHERE type = 'VECTOR'
              AND 'Entity' IN labelsOrTypes
              AND 'embedding' IN properties
            RETURN name
            LIMIT 1
            """
        ).single()
        if record:
            return record["name"]
    except Exception as e:
        print(f"⚠️ 自动查找 Entity 向量索引失败: {e}")

    for index_name in config.ENTITY_INDEX_CANDIDATES:
        try:
            record = session.run(
                """
                SHOW INDEXES YIELD name, type
                WHERE type = 'VECTOR' AND name = $index_name
                RETURN name
                LIMIT 1
                """,
                index_name=index_name,
            ).single()
            if record:
                return record["name"]
        except Exception:
            continue
    return ""


def find_similar_start_node(session, start_name: str, top_k: int = config.START_NODE_DEFAULT_K):
    """
    When exact/contains lookup misses, find the closest Entity by embedding.
    Prefer Neo4j vector index, then fall back to Python cosine over Entity.embedding.
    """
    query_embedding = get_embedding(start_name)
    if not query_embedding or not any(query_embedding):
        return None

    index_name = get_entity_vector_index_name(session)
    if index_name:
        try:
            record = session.run(
                """
                CALL db.index.vector.queryNodes($index_name, $top_k, $query_embedding)
                YIELD node, score
                WHERE node:Entity
                RETURN elementId(node) AS id,
                       labels(node) AS labels,
                       node AS node,
                       score AS similarity_score
                ORDER BY score DESC
                LIMIT 1
                """,
                index_name=index_name,
                top_k=top_k,
                query_embedding=query_embedding,
            ).single()
            if record:
                return record
        except Exception as e:
            print(f"⚠️ Entity 向量索引查询失败，改用 Python 相似度扫描: {e}")

    best_record = None
    best_score = -1.0
    result = session.run(
        """
        MATCH (n:Entity)
        WHERE n.embedding IS NOT NULL AND size(n.embedding) = $dimensions
        RETURN elementId(n) AS id,
               labels(n) AS labels,
               n AS node,
               n.embedding AS embedding
        """,
        dimensions=EMBEDDING_DIMENSIONS,
    )
    for record in result:
        score = cosine_similarity(query_embedding, record["embedding"])
        if score > best_score:
            best_score = score
            best_record = record

    if best_record:
        return {
            "id": best_record["id"],
            "labels": best_record["labels"],
            "node": best_record["node"],
            "similarity_score": best_score,
        }
    return None


# ================== BFS 子图抽取（含边去重） ==================

def extract_subgraph_bfs(start_name: str,
                         max_depth: int = config.BFS_DEPTH,
                         max_nodes: int = config.BFS_DEFAULT_MAX_NODES):
    """
    BFS 子图抽取，节点去重，边去重（基于无向节点对和关系类型，忽略属性差异）
    返回 (graph_dict, start_id)
    """
    with driver.session() as session:

        # 1️⃣ 查找起始节点（获取完整属性，排除 embedding）
        record = session.run(
            """
            MATCH (n:Entity)
            WHERE n.name CONTAINS $name
            RETURN elementId(n) AS id,
                   labels(n) AS labels,
                   n AS node
            LIMIT 1
            """,
            name=start_name
        ).single()

        if not record:
            print(f"❌ 未找到节点: {start_name}")
            record = find_similar_start_node(session, start_name, top_k=config.START_NODE_SEARCH_K)
            if not record:
                print(f"❌ 向量匹配也未找到相似起点: {start_name}")
                return None, None

            node_name = dict(record["node"].items()).get("name", record["id"])
            score = record.get("similarity_score")
            if score is None:
                print(f"✅ 使用向量匹配起点: {node_name} (原始起点: {start_name})")
            else:
                print(
                    f"✅ 使用向量匹配起点: {node_name} "
                    f"(原始起点: {start_name}, 相似度: {float(score):.4f})"
                )

        node_obj = record["node"]
        start_props = dict(node_obj.items())
        start_props.pop("embedding", None)
        start_id = record["id"]
        start_labels = record["labels"]

        print(f"✅ 起始节点: {start_props.get('name', start_id)} (id={start_id})")

        # BFS 数据结构
        queue = deque([(start_id, 0)])
        visited: Set[str] = set([start_id])

        nodes = {}  # id -> node dict
        edges = []  # 边列表
        edges_set = set()  # 用于边去重的集合（标准化键）

        # 保存起始节点
        nodes[start_id] = {
            "id": start_id,
            "labels": start_labels,
            **start_props
        }

        # 2️⃣ BFS 遍历
        while queue and len(nodes) < max_nodes:
            current_id, depth = queue.popleft()
            if depth >= max_depth:
                continue

            neighbors = session.execute_read(get_neighbors, current_id)

            for rec in neighbors:
                nid = rec["nid"]
                rel_type = rec["rel_type"]
                src = rec["src"]
                dst = rec["dst"]
                n_labels = rec["labels"]
                n_props = rec["props"]  # 已排除 embedding
                rel_props = rec["rel_props"]  # 关系的其他属性（排除 embedding）

                # 保存节点（若尚未保存）
                if nid not in nodes:
                    nodes[nid] = {
                        "id": nid,
                        "labels": n_labels,
                        **n_props
                    }

                # 边去重：标准化键 (min(src,dst), max(src,dst), rel_type)
                edge_key = (min(src, dst), max(src, dst), rel_type)
                if edge_key not in edges_set:
                    edges_set.add(edge_key)
                    # 构建边字典：包含 src, dst, type 以及关系的其他属性
                    edge_dict = {
                        "src": src,
                        "dst": dst,
                        "type": rel_type,
                        **rel_props
                    }
                    edges.append(edge_dict)

                # BFS 扩展
                if nid not in visited:
                    visited.add(nid)
                    queue.append((nid, depth + 1))

            if len(nodes) >= max_nodes:
                print("⚠️ 达到节点上限，停止扩展")
                break

        print(f"📊 抽取节点数: {len(nodes)}")
        print(f"📊 抽取关系数: {len(edges)}")

        graph_dict = {
            "nodes": list(nodes.values()),
            "edges": edges
        }
        return graph_dict, start_id


# ================== 路径收集（不含剪枝，返回原始路径列表） ==================

def get_all_paths(graph_data: Dict, start_id: str, max_depth: int = config.BFS_DEPTH) -> Tuple[List, Dict, int]:
    """
    从起始节点开始，收集所有从起点出发、长度在 1 到 max_depth 之间的路径，
    每条路径包含节点名称序列和关系类型序列。
    返回 (paths, id_to_name) 其中 paths 为 [(node_names_list, rel_types_list), ...]
    """
    # 建立节点 id -> 显示名称 的映射
    id_to_name = {}
    for node in graph_data["nodes"]:
        node_id = node["id"]
        name = node.get("name")
        if not name:
            name = node.get("title")
        if not name:
            name = node_id
        id_to_name[node_id] = name

    # 构建邻接表：{node_id: [(neighbor_id, rel_type), ...]}
    adj = {}
    for edge in graph_data["edges"]:
        src = edge["src"]
        dst = edge["dst"]
        rel = edge["type"]
        if src not in adj:
            adj[src] = []
        if dst not in adj:
            adj[dst] = []
        adj[src].append((dst, rel))
        adj[dst].append((src, rel))  # 无向图，双向添加

    # 收集所有从起点出发、长度不超过 max_depth 的路径
    all_paths = []
    visited_set = set()
    raw_path_count = 0

    def dfs(current_id, depth, node_path_ids, rel_path):
        nonlocal raw_path_count
        if depth >= 1:
            raw_path_count += 1
            node_names = tuple(id_to_name.get(nid, nid) for nid in node_path_ids)
            rel_tuple = tuple(rel_path)
            key = (node_names, rel_tuple)
            if key not in visited_set:
                visited_set.add(key)
                all_paths.append((list(node_names), list(rel_path)))
        if depth >= max_depth:
            return
        for neighbor_id, rel_type in adj.get(current_id, []):
            if neighbor_id in node_path_ids:
                continue
            node_path_ids.append(neighbor_id)
            rel_path.append(rel_type)
            dfs(neighbor_id, depth + 1, node_path_ids, rel_path)
            node_path_ids.pop()
            rel_path.pop()

    dfs(start_id, 0, [start_id], [])
    return all_paths, id_to_name, raw_path_count


def format_paths_table(paths: List[Tuple[List[str], List[str]]],
                       max_depth: int = config.BFS_DEPTH,
                       raw_path_count: int = None,
                       deduplicated_path_count: int = None,
                       display_stage: str = "剪枝后") -> str:
    """
    将路径列表格式化为表格字符串，不截断名称。
    """
    lines = []
    lines.append("=" * 100)
    raw_count = len(paths) if raw_path_count is None else raw_path_count
    deduplicated_count = len(paths) if deduplicated_path_count is None else deduplicated_path_count
    lines.append(
        f"从起始节点出发，深度 ≤ {max_depth} 的完整路径"
        f"（去重前共 {raw_count} 条，去重后共 {deduplicated_count} 条，{display_stage}展示 {len(paths)} 条）"
    )
    lines.append("-" * 100)

    for node_names, rel_types in paths:
        row = []
        for i in range(len(rel_types)):
            node_name = node_names[i]
            rel_name = rel_types[i]
            # 直接使用原始名称，不截断
            row.append(node_name)
            row.append(f"[{rel_name}]-->")
        # 最后一个节点
        last_node = node_names[-1]
        row.append(last_node)
        lines.append("  ".join(row))

    lines.append("=" * 100)
    return "\n".join(lines)

# ================== 向量剪枝：根据子问题列表过滤路径（批处理优化版） ==================

def prune_paths_by_questions(paths: List[Tuple[List[str], List[str]]],
                             questions: List[str],
                             top_k: int = config.PRUNE_DEFAULT_K) -> List[Tuple[List[str], List[str]]]:
    """
    根据多跳问题列表逐层剪枝路径。
    第 i 跳问题（索引 i）用于剪枝路径中第 i+1 层的关系（关系列表索引 i）。
    每层保留相似度最高的 top_k 种关系类型，只有该层关系属于这些类型的路径才被保留。

    优化：批量获取所有问题和所有关系类型的 embedding。
    """
    if not questions:
        return paths

    # 1. 批量获取所有子问题的 embedding
    q_embeddings = batch_get_embeddings(questions)

    # 收集所有路径中出现的所有关系类型（按层索引）
    max_layer = max(len(rel_list) for _, rel_list in paths) if paths else 0
    layer_relations: List[Set[str]] = [set() for _ in range(max_layer)]
    for _, rel_list in paths:
        for idx, rel in enumerate(rel_list):
            if idx < len(layer_relations):
                layer_relations[idx].add(rel)

    # 2. 对于每一层，收集该层所有关系类型并批量获取 embedding
    top_relations_per_layer = []
    for layer_idx in range(min(len(questions), max_layer)):
        rels = list(layer_relations[layer_idx])
        if not rels:
            top_relations_per_layer.append(set())
            continue

        # 批量获取该层所有关系类型的 embedding
        rel_embeddings = batch_get_embeddings(rels)
        # 计算相似度
        sim_list = []
        q_emb = q_embeddings[layer_idx]
        for rel, emb in zip(rels, rel_embeddings):
            sim = cosine_similarity(q_emb, emb)
            sim_list.append((rel, sim))

        # 按相似度降序排序，取前 top_k
        sim_list.sort(key=lambda x: x[1], reverse=True)
        top_rels = {rel for rel, _ in sim_list[:top_k]}
        top_relations_per_layer.append(top_rels)
        print(f"📌 第 {layer_idx + 1} 层问题: {questions[layer_idx]}")
        print(f"   保留的关系类型: {top_rels}")

    # 3. 过滤路径
    pruned_paths = []
    for node_names, rel_list in paths:
        keep = True
        for layer_idx, rel in enumerate(rel_list):
            if layer_idx >= len(top_relations_per_layer):
                break
            if rel not in top_relations_per_layer[layer_idx]:
                keep = False
                break
        if keep:
            pruned_paths.append((node_names, rel_list))

    print(f"\n✂️ 剪枝前路径数: {len(paths)}，剪枝后路径数: {len(pruned_paths)}")
    return pruned_paths


# ================== JSON 导出 ==================

def save_graph_json(graph_data: Dict, filename=config.DEFAULT_SUBGRAPH_FILE):
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(graph_data, f, ensure_ascii=False, indent=2)
    print(f"✅ 子图已保存: {filename}")


def neo4j_BFS(
        subquestions=config.EXAMPLE_NEO4J_BFS):
    # 解析参数字符串，格式：START_NODE_NAME:...;DEPTH:...;q1:...;q2:...;...
    if not subquestions:
        return "缺少参数 START_NODE_NAME、DEPTH，无法进行BFS检索。"

    parts = subquestions.split(';')
    start_node_name = None
    depth = None
    questions = []
    invalid_params = []
    depth_provided = False
    for part in parts:
        part = part.strip()
        if not part or ':' not in part:
            continue
        key, value = part.split(':', 1)
        key = key.strip()
        value = value.strip()
        if key == "START_NODE_NAME":
            start_node_name = value
        elif key == "DEPTH":
            depth_provided = True
            try:
                depth = int(value)
            except ValueError:
                invalid_params.append("DEPTH")
        elif key.startswith('q'):  # q1, q2, ...
            questions.append(value)

    missing_params = []
    if not start_node_name:
        missing_params.append("START_NODE_NAME")
    if not depth_provided:
        missing_params.append("DEPTH")

    if missing_params or invalid_params:
        problems = []
        if missing_params:
            problems.append(f"缺少参数 {', '.join(missing_params)}")
        if invalid_params:
            problems.append(f"参数格式错误 {', '.join(invalid_params)}")
        return "；".join(problems) + "，无法进行BFS检索。"

    MAX_NODES = config.BFS_MAX_NODES  # 固定最大节点数

    # 1. 抽取子图
    graph, start_id = extract_subgraph_bfs(
        start_node_name,
        max_depth=depth,
        max_nodes=MAX_NODES
    )

    if graph:
        # 2. 保存子图 JSON
        save_graph_json(graph, config.SUBGRAPH_FILE)

        # 3. 获取所有原始路径
        all_paths, id_to_name, raw_path_count = get_all_paths(graph, start_id, max_depth=depth)

        # 4. 向量剪枝（根据子问题列表，已优化为批量获取 embedding）
        pruned_paths = prune_paths_by_questions(all_paths, questions, top_k=config.PRUNE_RELATION_K)

        # 5. 仅输出剪枝后的完整路径
        # 如需核对去重后、剪枝前的所有完整路径，可取消下面代码的注释。
        # deduplicated_table = format_paths_table(
        #     all_paths,
        #     max_depth=depth,
        #     raw_path_count=raw_path_count,
        #     deduplicated_path_count=len(all_paths),
        #     display_stage="剪枝前"
        # )
        pruned_table = format_paths_table(
            pruned_paths,
            max_depth=depth,
            raw_path_count=raw_path_count,
            deduplicated_path_count=len(all_paths),
            display_stage="剪枝后"
        )
        return pruned_table


# ================== 主程序 ==================
if __name__ == "__main__":
    print(neo4j_BFS())
