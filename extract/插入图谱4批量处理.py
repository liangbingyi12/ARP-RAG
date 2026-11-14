# neo4j_batch_insert_fixed.py
import config as settings
import json
import os
from neo4j import GraphDatabase
from typing import List, Dict, Any, Generator
import time

# Neo4j数据库连接配置
uri = settings.NEO4J_URI
username = settings.NEO4J_USERNAME
password = settings.NEO4J_PASSWORD

# 文件路径配置
extraction_file = settings.IMPORT_EXTRACTION_FILE
embeddings_file = settings.IMPORT_EMBEDDINGS_FILE

# 注意：行号从1开始计数
start_item = settings.IMPORT_START_ITEM
end_item = settings.IMPORT_END_ITEM  # 可以调整为None处理所有数据
batch_size = settings.IMPORT_BATCH_SIZE  # 每个批次处理的文档数，可根据内存调整


def quote_index(name):
    return "`" + name.replace("`", "``") + "`"


class Neo4jBatchGraph:
    """Neo4j图数据库批量操作类"""

    def __init__(self, uri: str, username: str, password: str):
        # 初始化Neo4j数据库连接
        self.driver = GraphDatabase.driver(uri, auth=(username, password))
        # 设置向量维度
        self.vector_dimensions = settings.VECTOR_DIMENSIONS  # OpenAI嵌入的维度
        self.batch_size = settings.NEO4J_BATCH_SIZE  # Neo4j批量操作的大小

    def close(self):
        # 关闭数据库连接
        self.driver.close()

    def setup_vector_indices(self):
        """设置向量索引 - 为实体、文档和关系都创建向量索引"""
        with self.driver.session() as session:
            try:
                # 创建实体向量索引
                entity_index_query = """
                CREATE VECTOR INDEX __INDEX__ IF NOT EXISTS
                FOR (n:Entity) ON (n.embedding)
                OPTIONS {
                    indexConfig: {
                        `vector.dimensions`: $dimensions,
                        `vector.similarity_function`: $similarity
                    }
                }
                """
                session.run(entity_index_query.replace("__INDEX__", quote_index(settings.ENTITY_INDEX)), dimensions=self.vector_dimensions, similarity=settings.VECTOR_SIMILARITY)
                print("✅ 实体向量索引创建成功")

                # 创建文档向量索引
                document_index_query = """
                CREATE VECTOR INDEX __INDEX__ IF NOT EXISTS
                FOR (n:Document) ON (n.embedding)
                OPTIONS {
                    indexConfig: {
                        `vector.dimensions`: $dimensions,
                        `vector.similarity_function`: $similarity
                    }
                }
                """
                session.run(document_index_query.replace("__INDEX__", quote_index(settings.DOCUMENT_INDEX)), dimensions=self.vector_dimensions, similarity=settings.VECTOR_SIMILARITY)
                print("✅ 文档向量索引创建成功")

                # 创建关系向量索引（备选语法）
                try:
                    relation_index_query = """
                    CREATE VECTOR INDEX __INDEX__ IF NOT EXISTS
                    FOR ()-[r:RELATES]-()
                    ON (r.embedding)
                    OPTIONS {
                        indexConfig: {
                            `vector.dimensions`: $dimensions,
                            `vector.similarity_function`: $similarity
                        }
                    }
                    """
                    session.run(relation_index_query.replace("__INDEX__", quote_index(settings.RELATIONSHIP_INDEX)), dimensions=self.vector_dimensions, similarity=settings.VECTOR_SIMILARITY)
                    print("✅ 关系向量索引创建成功")
                except Exception as rel_index_error:
                    print(f"⚠️  创建关系向量索引失败: {rel_index_error}")

            except Exception as e:
                print(f"❌ 设置向量索引时出错: {e}")

    def create_constraints(self):
        """创建唯一性约束，确保节点不重复"""
        with self.driver.session() as session:
            # 为实体创建唯一性约束
            session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (e:Entity) REQUIRE e.name IS UNIQUE")
            # 为文档创建唯一性约束
            session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (d:Document) REQUIRE d.doc_id IS UNIQUE")
            print("✅ 约束创建成功")

    def batch_create_documents(self, documents_batch: List[Dict[str, Any]]):
        """批量创建文档节点"""
        if not documents_batch:
            return 0

        query = """
        UNWIND $batch AS doc
        MERGE (d:Document {doc_id: doc.doc_id})
        SET d.title = doc.title,
            d.document_text = doc.document_text,
            d.embedding = doc.embedding
        RETURN count(d) as created_count
        """

        # 准备批量数据
        batch_data = []
        placeholder_vector = [0.0] * self.vector_dimensions

        for doc in documents_batch:
            doc_id = doc['doc_id']
            title = doc.get('title', '')
            document_text = doc['embedding_texts']['document_embedding_text']

            # 从embeddings中获取文档的embedding向量
            document_embedding = doc.get('embeddings', {}).get('document_embedding')
            if not document_embedding:
                document_embedding = placeholder_vector

            batch_data.append({
                'doc_id': doc_id,
                'title': title,
                'document_text': document_text,
                'embedding': document_embedding
            })

        with self.driver.session() as session:
            result = session.run(query, batch=batch_data)
            return result.single()["created_count"]

    def batch_create_entities(self, entities_batch: List[Dict[str, Any]]):
        """批量创建实体节点"""
        if not entities_batch:
            return 0

        query = """
        UNWIND $batch AS entity
        MERGE (e:Entity {name: entity.name})
        ON CREATE SET 
            e.type = entity.entity_type,
            e.context = entity.context,
            e.doc_ids = [entity.doc_id],
            e.embedding = entity.embedding
        ON MATCH SET 
            e.type = CASE 
                WHEN e.type = 'Unknown' OR e.type = 'OTHER' THEN entity.entity_type 
                ELSE e.type 
            END,
            e.context = CASE 
                WHEN e.context IS NULL OR e.context = '' THEN entity.context 
                ELSE e.context 
            END,
            e.doc_ids = CASE 
                WHEN entity.doc_id IN e.doc_ids THEN e.doc_ids 
                ELSE e.doc_ids + entity.doc_id 
            END
        RETURN count(e) as processed_count
        """

        # 准备批量数据
        batch_data = []
        placeholder_vector = [0.0] * self.vector_dimensions

        for entity_info in entities_batch:
            entity_name = entity_info.get('name', '').strip()
            if not entity_name:
                continue

            batch_data.append({
                'name': entity_name,
                'entity_type': entity_info.get('type', 'OTHER'),
                'context': entity_info.get('context', ''),
                'doc_id': entity_info['doc_id'],
                'embedding': entity_info.get('embedding', placeholder_vector)
            })

        if not batch_data:
            return 0

        with self.driver.session() as session:
            result = session.run(query, batch=batch_data)
            return result.single()["processed_count"]

    def batch_create_document_entity_relationships(self, relationships_batch: List[Dict[str, str]]):
        """批量创建文档和实体之间的关系"""
        if not relationships_batch:
            return 0

        query = """
        UNWIND $batch AS rel
        MATCH (d:Document {doc_id: rel.doc_id})
        MATCH (e:Entity {name: rel.entity_name})
        MERGE (d)-[:CONTAINS]->(e)
        RETURN count(*) as created_count
        """

        with self.driver.session() as session:
            result = session.run(query, batch=relationships_batch)
            return result.single()["created_count"]

    def batch_create_entity_relationships(self, relations_batch: List[Dict[str, Any]]):
        """批量创建实体之间的关系"""
        if not relations_batch:
            return 0

        # 先确保所有实体都存在
        ensure_entities_query = """
        UNWIND $entities AS entity
        MERGE (e:Entity {name: entity.name})
        ON CREATE SET e.type = 'Unknown', e.doc_ids = [entity.doc_id]
        ON MATCH SET e.doc_ids = CASE 
            WHEN entity.doc_id IN e.doc_ids THEN e.doc_ids 
            ELSE e.doc_ids + entity.doc_id 
        END
        """

        # 批量创建关系
        create_relations_query = """
        UNWIND $batch AS rel
        MATCH (e1:Entity {name: rel.subject})
        MATCH (e2:Entity {name: rel.object})
        CREATE (e1)-[r:RELATES {
            type: rel.predicate,
            context: rel.context,
            confidence: rel.confidence,
            source_doc: rel.doc_id,
            embedding: rel.embedding
        }]->(e2)
        RETURN count(r) as created_count
        """

        with self.driver.session() as session:
            # 1. 确保所有实体存在
            unique_entities = set()
            for rel in relations_batch:
                unique_entities.add((rel['subject'], rel['doc_id']))
                unique_entities.add((rel['object'], rel['doc_id']))

            entities_batch = [{'name': name, 'doc_id': doc_id} for name, doc_id in unique_entities]
            if entities_batch:
                # 分批确保实体存在
                for i in range(0, len(entities_batch), self.batch_size):
                    batch = entities_batch[i:i + self.batch_size]
                    session.run(ensure_entities_query, entities=batch)

            # 2. 分批创建关系
            total_created = 0
            for i in range(0, len(relations_batch), self.batch_size):
                batch = relations_batch[i:i + self.batch_size]
                result = session.run(create_relations_query, batch=batch)
                total_created += result.single()["created_count"]

            return total_created

    def batch_update_entity_embeddings(self, updates_batch: List[Dict[str, Any]]):
        """批量更新实体的embedding向量"""
        if not updates_batch:
            return 0

        query = """
        UNWIND $batch AS update
        MATCH (e:Entity {name: update.entity_name})
        SET e.embedding = update.embedding
        RETURN count(e) as updated_count
        """

        total_updated = 0
        with self.driver.session() as session:
            # 分批更新
            for i in range(0, len(updates_batch), self.batch_size):
                batch = updates_batch[i:i + self.batch_size]
                result = session.run(query, batch=batch)
                total_updated += result.single()["updated_count"]

        return total_updated

    def batch_update_relation_embeddings(self, updates_batch: List[Dict[str, Any]]):
        """批量更新关系的embedding向量"""
        if not updates_batch:
            return 0

        query = """
        UNWIND $batch AS update
        MATCH (e1:Entity {name: update.subject})-[r:RELATES]->(e2:Entity {name: update.object})
        WHERE r.type = update.predicate AND r.source_doc = update.source_doc
        SET r.embedding = update.embedding
        RETURN count(r) as updated_count
        """

        total_updated = 0
        with self.driver.session() as session:
            # 分批更新
            for i in range(0, len(updates_batch), self.batch_size):
                batch = updates_batch[i:i + self.batch_size]
                result = session.run(query, batch=batch)
                total_updated += result.single()["updated_count"]

        return total_updated


def read_jsonl_file(file_path: str, start_item: int = settings.IMPORT_DEFAULT_START_ITEM, end_item: int = settings.IMPORT_DEFAULT_END_ITEM):
    """流式读取JSONL文件"""
    if not os.path.exists(file_path):
        print(f"❌ 文件不存在: {file_path}")
        return

    with open(file_path, 'r', encoding='utf-8') as f:
        line_count = 0

        for line in f:
            line_count += 1

            # 跳过开始位置之前的行
            if line_count < start_item:
                continue

            # 如果指定了结束位置，且超过则停止
            if end_item is not None and line_count > end_item:
                break

            try:
                data = json.loads(line.strip())
                data['original_line'] = line_count
                yield data
            except json.JSONDecodeError as e:
                print(f"❌ 解析第 {line_count} 行时出错: {e}")


def load_embeddings_map(file_path: str):
    """加载embeddings结果到内存映射"""
    embeddings_map = {}

    if not os.path.exists(file_path):
        print(f"⚠️  Embeddings文件不存在: {file_path}")
        return embeddings_map

    print("📦 正在加载embedding结果到内存...")
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data = json.loads(line.strip())
                doc_id = data['doc_id']
                embeddings_map[doc_id] = data
            except json.JSONDecodeError as e:
                print(f"❌ 解析embedding行时出错: {e}")

    print(f"✅ 加载了 {len(embeddings_map)} 条embedding结果到内存")
    return embeddings_map


def match_entity_with_embedding(extraction_entities, embedding_entities):
    """匹配抽取的实体和embedding文本"""
    matched_embeddings = {}

    # 创建embedding文本到向量的映射
    text_to_embedding = {}
    for item in embedding_entities:
        text = item['text']
        embedding = item['embedding']
        text_to_embedding[text] = embedding

    # 尝试匹配抽取的实体
    for entity in extraction_entities:
        entity_name = entity.get('normalized_name', entity.get('name', ''))
        entity_context = entity.get('context', '')

        if not entity_name:
            continue

        # 尝试不同的匹配策略
        match_patterns = [
            f"{entity_name} ({entity.get('type', 'OTHER')}): {entity_context}",
            f"{entity_name} ({entity.get('type', 'OTHER')})",
            entity_name
        ]

        for pattern in match_patterns:
            if pattern in text_to_embedding:
                matched_embeddings[entity_name] = text_to_embedding[pattern]
                break

    return matched_embeddings


def match_relation_with_embedding(extraction_relations, embedding_relations):
    """匹配抽取的关系和embedding文本"""
    matched_embeddings = {}

    # 简单按索引匹配（假设顺序一致）
    for idx, relation in enumerate(extraction_relations):
        if idx < len(embedding_relations):
            matched_embeddings[idx] = embedding_relations[idx]['embedding']

    return matched_embeddings


def match_relation_with_embedding_advanced(extraction_relations, embedding_relations, doc_id=None):
    """高级匹配抽取的关系和embedding文本"""
    matched_embeddings = {}

    if not embedding_relations:
        return matched_embeddings

    if all('relation_index' in item for item in embedding_relations):
        for item in embedding_relations:
            idx = item['relation_index']
            if 0 <= idx < len(extraction_relations):
                context = (extraction_relations[idx].get('context') or '').strip()
                if context and context == item['text']:
                    matched_embeddings[idx] = item['embedding']
        return matched_embeddings

    # 创建embedding文本到向量的映射
    text_to_embedding = {}
    for item in embedding_relations:
        text = item['text']
        embedding = item['embedding']
        text_to_embedding[text] = embedding

    # 尝试多种匹配策略
    for idx, relation in enumerate(extraction_relations):
        subject = relation.get('subject', '')
        predicate = relation.get('predicate', '')
        object_ = relation.get('object', '')
        context = relation.get('context', '')

        if not subject or not predicate or not object_:
            continue

        # 策略1：构建标准关系描述文本进行匹配
        # 关系embedding文本通常格式："subject predicate object (context)"
        standard_patterns = [
            f"{subject} {predicate} {object_} ({context})" if context else f"{subject} {predicate} {object_}",
            f"{subject} {predicate} {object_}",
            f"{subject} {predicate} {object_}: {context}" if context else None,
            f"{subject} {predicate} {object_} 在文档{doc_id}" if doc_id else None,
        ]

        # 策略2：尝试从embedding文本中提取关键信息进行匹配
        matched = False
        for pattern in standard_patterns:
            if pattern and pattern in text_to_embedding:
                matched_embeddings[idx] = text_to_embedding[pattern]
                matched = True
                break

        # 策略3：如果标准匹配失败，尝试模糊匹配
        if not matched:
            for emb_text, embedding in text_to_embedding.items():
                # 检查文本中是否包含主体和客体
                if subject in emb_text and object_ in emb_text:
                    # 进一步检查是否包含关系谓词
                    if predicate in emb_text or any(word in emb_text for word in predicate.split()):
                        matched_embeddings[idx] = embedding
                        matched = True
                        break

    # 策略4：按索引顺序匹配（如果其他策略都失败）
    if len(matched_embeddings) < min(len(extraction_relations), len(embedding_relations)):
        for idx, relation in enumerate(extraction_relations):
            if idx < len(embedding_relations) and idx not in matched_embeddings:
                matched_embeddings[idx] = embedding_relations[idx]['embedding']

    return matched_embeddings


def process_data_batches(graph: Neo4jBatchGraph, extraction_file: str, embeddings_file: str,
                         start_item: int = settings.IMPORT_DEFAULT_START_ITEM, end_item: int = settings.IMPORT_DEFAULT_END_ITEM, doc_batch_size: int = settings.IMPORT_DEFAULT_DOC_BATCH_SIZE):
    """批量处理数据并插入到Neo4j"""

    # 加载embedding映射
    embeddings_map = load_embeddings_map(embeddings_file)

    # 初始化统计
    stats = {
        "total_docs": 0,
        "docs_processed": 0,
        "entities_created": 0,
        "doc_entity_rels_created": 0,
        "entity_rels_created": 0,
        "entity_embeddings_updated": 0,
        "relation_embeddings_updated": 0,
        "start_time": time.time()
    }

    print(f"\n🚀 开始批量处理数据 (批次大小: {doc_batch_size})")
    print(f"{'=' * 60}")

    # 流式读取抽取结果
    doc_batch = []
    batch_counter = 0

    for doc_data in read_jsonl_file(extraction_file, start_item, end_item):
        doc_batch.append(doc_data)
        stats["total_docs"] += 1

        # 当达到批次大小时处理
        if len(doc_batch) >= doc_batch_size:
            batch_counter += 1
            stats = process_single_batch(graph, doc_batch, embeddings_map, stats, batch_counter)
            doc_batch = []  # 清空当前批次

            # 显示进度
            progress_pct = (stats["docs_processed"] / stats["total_docs"]) * 100 if stats["total_docs"] > 0 else 0
            elapsed_time = time.time() - stats["start_time"]
            docs_per_sec = stats["docs_processed"] / elapsed_time if elapsed_time > 0 else 0

            print(f"📊 进度: {stats['docs_processed']}/{stats['total_docs']} ({progress_pct:.1f}%) | "
                  f"速率: {docs_per_sec:.1f} docs/sec")

    # 处理最后一批（如果有）
    if doc_batch:
        batch_counter += 1
        stats = process_single_batch(graph, doc_batch, embeddings_map, stats, batch_counter)

    # 计算总时间
    total_time = time.time() - stats["start_time"]

    # 打印最终统计
    print(f"\n{'=' * 60}")
    print("🎉 批量数据处理完成!")
    print(f"{'=' * 60}")
    print(f"📊 最终统计信息:")
    print(f"  - 总文档数: {stats['total_docs']}")
    print(f"  - 已处理文档数: {stats['docs_processed']}")
    print(f"  - 实体节点数: {stats['entities_created']}")
    print(f"  - 文档-实体关系数: {stats['doc_entity_rels_created']}")
    print(f"  - 实体间关系数: {stats['entity_rels_created']}")
    print(f"  - 实体embedding更新数: {stats['entity_embeddings_updated']}")
    print(f"  - 关系embedding更新数: {stats['relation_embeddings_updated']}")
    print(f"  - 总处理时间: {total_time:.2f}秒")
    print(f"  - 平均速率: {stats['docs_processed'] / total_time:.2f} docs/秒")

    return stats


def process_single_batch(graph: Neo4jBatchGraph, doc_batch: List[Dict], embeddings_map: Dict,
                         stats: Dict, batch_num: int):
    """处理单个文档批次"""

    print(f"\n🔨 处理批次 #{batch_num}: {len(doc_batch)} 个文档")

    # 准备批量数据
    documents_for_batch = []
    entities_for_batch = []
    doc_entity_rels_for_batch = []
    entity_rels_for_batch = []
    entity_embedding_updates = []
    relation_embedding_updates = []

    # 遍历当前批次的文档
    for doc_data in doc_batch:
        doc_id = doc_data['doc_id']

        # 获取对应的embedding数据
        embedding_data = embeddings_map.get(doc_id)
        if not embedding_data:
            continue

        # 合并数据用于文档创建
        merged_doc = {**doc_data, **embedding_data}
        documents_for_batch.append(merged_doc)

        # 提取抽取结果
        extraction_result = doc_data.get('extraction_result', {})

        # 收集实体信息
        entities = extraction_result.get('entities', [])
        for entity in entities:
            entity_name = entity.get('normalized_name', entity.get('name', ''))
            if not entity_name:
                continue

            # 添加到实体批次
            entities_for_batch.append({
                **entity,
                'doc_id': doc_id,
                'name': entity_name
            })

            # 添加到文档-实体关系批次
            doc_entity_rels_for_batch.append({
                'doc_id': doc_id,
                'entity_name': entity_name
            })

        # 收集关系信息
        relations = extraction_result.get('relations', [])

        # 匹配关系embedding
        relation_embeddings_list = embedding_data.get('embeddings', {}).get('relation_embeddings', [])

        # 使用高级匹配函数
        matched_relation_embeddings = match_relation_with_embedding_advanced(
            relations,
            relation_embeddings_list,
            doc_id
        )

        for idx, relation in enumerate(relations):
            subject = relation.get('subject', '')
            predicate = relation.get('predicate', '')
            object_ = relation.get('object', '')

            if not subject or not predicate or not object_:
                continue

            # 确定使用的embedding
            if idx in matched_relation_embeddings:
                relation_embedding = matched_relation_embeddings[idx]
                # 添加到关系embedding更新批次
                relation_embedding_updates.append({
                    'subject': subject,
                    'predicate': predicate,
                    'object': object_,
                    'source_doc': doc_id,
                    'embedding': relation_embedding
                })
            else:
                relation_embedding = [0.0] * graph.vector_dimensions  # 使用占位符

            # 添加到关系批次
            entity_rels_for_batch.append({
                'subject': subject,
                'predicate': predicate,
                'object': object_,
                'context': relation.get('context', ''),
                'confidence': relation.get('confidence', '中'),
                'doc_id': doc_id,
                'embedding': relation_embedding
            })

    # 批量插入文档节点
    if documents_for_batch:
        created = graph.batch_create_documents(documents_for_batch)
        stats["docs_processed"] += created
        print(f"  ✅ 批量创建了 {created} 个文档节点")

    # 批量插入实体节点
    if entities_for_batch:
        # 首先为实体匹配embedding
        for entity_info in entities_for_batch:
            doc_id = entity_info['doc_id']
            entity_name = entity_info['name']

            # 从embedding数据中查找匹配的embedding
            if doc_id in embeddings_map:
                embedding_data = embeddings_map[doc_id]
                extraction_entities = [e for e in entities_for_batch if e['doc_id'] == doc_id]
                embedding_entities = embedding_data.get('embeddings', {}).get('entity_embeddings', [])

                matched = match_entity_with_embedding(extraction_entities, embedding_entities)
                if entity_name in matched:
                    entity_info['embedding'] = matched[entity_name]
                    entity_embedding_updates.append({
                        'entity_name': entity_name,
                        'embedding': matched[entity_name]
                    })

        # 批量创建实体节点
        created = graph.batch_create_entities(entities_for_batch)
        stats["entities_created"] += created
        print(f"  ✅ 批量创建/更新了 {created} 个实体节点")

    # 批量创建文档-实体关系
    if doc_entity_rels_for_batch:
        created = graph.batch_create_document_entity_relationships(doc_entity_rels_for_batch)
        stats["doc_entity_rels_created"] += created
        print(f"  ✅ 批量创建了 {created} 个文档-实体关系")

    # 批量创建实体间关系
    if entity_rels_for_batch:
        created = graph.batch_create_entity_relationships(entity_rels_for_batch)
        stats["entity_rels_created"] += created
        print(f"  ✅ 批量创建了 {created} 个实体间关系")

    # 批量更新实体embedding
    if entity_embedding_updates:
        updated = graph.batch_update_entity_embeddings(entity_embedding_updates)
        stats["entity_embeddings_updated"] += updated
        print(f"  ✅ 批量更新了 {updated} 个实体embedding")

    # 批量更新关系embedding
    if relation_embedding_updates:
        updated = graph.batch_update_relation_embeddings(relation_embedding_updates)
        stats["relation_embeddings_updated"] += updated
        print(f"  ✅ 批量更新了 {updated} 个关系embedding")
    else:
        print(f"  ⚠️  本批次没有匹配到关系embedding")

    return stats


def get_database_stats(graph: Neo4jBatchGraph):
    """获取数据库统计信息"""
    with graph.driver.session() as session:
        # 统计实体数量
        result = session.run("MATCH (e:Entity) RETURN count(e) as entity_count")
        entity_count = result.single()["entity_count"]

        # 统计有embedding的实体数量
        result = session.run(
            "MATCH (e:Entity) WHERE e.embedding IS NOT NULL AND size(e.embedding) > 0 RETURN count(e) as entity_with_embedding")
        entity_with_embedding = result.single()["entity_with_embedding"]

        # 统计关系数量
        result = session.run("MATCH ()-[r:RELATES]->() RETURN count(r) as relation_count")
        relation_count = result.single()["relation_count"]

        # 统计有embedding的关系数量
        result = session.run(
            "MATCH ()-[r:RELATES]->() WHERE r.embedding IS NOT NULL AND size(r.embedding) > 0 RETURN count(r) as relation_with_embedding")
        relation_with_embedding = result.single()["relation_with_embedding"]

        # 统计文档数量
        result = session.run("MATCH (d:Document) RETURN count(d) as doc_count")
        doc_count = result.single()["doc_count"]

        # 统计CONTAINS关系数量
        result = session.run("MATCH ()-[r:CONTAINS]->() RETURN count(r) as contains_count")
        contains_count = result.single()["contains_count"]

        print(f"\n📊 数据库统计信息:")
        print(f"  - 文档数量: {doc_count}")
        print(f"  - 实体数量: {entity_count} ({entity_with_embedding} 个有embedding)")
        print(f"  - 实体间关系数量: {relation_count} ({relation_with_embedding} 个有embedding)")
        print(f"  - 文档-实体关系数量: {contains_count}")

        # 查询示例实体
        print(f"\n🔍 示例实体 (按出现文档数排序):")
        result = session.run("""
            MATCH (e:Entity)
            RETURN e.name as name, e.type as type, size(e.doc_ids) as doc_count
            ORDER BY doc_count DESC
            LIMIT 5
        """)

        for record in result:
            print(f"  - {record['name']} ({record['type']}) - 出现在 {record['doc_count']} 个文档中")

        # 查询示例关系（有embedding的）
        print(f"\n🔍 示例关系 (有embedding的):")
        result = session.run("""
            MATCH (e1:Entity)-[r:RELATES]->(e2:Entity)
            WHERE r.embedding IS NOT NULL AND size(r.embedding) > 0
            RETURN e1.name as subject, r.type as predicate, e2.name as object, 
                   r.source_doc as doc_id
            LIMIT 5
        """)

        for record in result:
            print(f"  - {record['subject']} --[{record['predicate']}]--> {record['object']} (文档{record['doc_id']})")


def main():
    """主函数"""
    try:
        print("🔗 正在连接Neo4j数据库...")
        graph = Neo4jBatchGraph(uri, username, password)

        # 创建约束
        graph.create_constraints()

        # 设置向量索引
        print("\n🔧 设置向量索引...")
        graph.setup_vector_indices()

        # 批量处理数据
        print("\n📤 开始批量处理数据并插入到Neo4j...")
        stats = process_data_batches(
            graph=graph,
            extraction_file=extraction_file,
            embeddings_file=embeddings_file,
            start_item=start_item,
            end_item=end_item,
            doc_batch_size=batch_size
        )

        # 显示数据库统计
        if stats.get("docs_processed", 0) > 0:
            get_database_stats(graph)

    except Exception as e:
        print(f"❌ 程序执行过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # 关闭数据库连接
        if 'graph' in locals():
            graph.close()
            print("\n🔒 数据库连接已关闭")


if __name__ == "__main__":
    main()