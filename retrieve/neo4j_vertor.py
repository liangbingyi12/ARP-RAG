# neo4j_vector.py
import openai
import numpy as np
import re
from neo4j import GraphDatabase
from typing import List, Dict, Any, Tuple, Optional

# ==================== 配置 ====================
import config
from config import (
    VECTOR_NEO4J_URI as NEO4J_URI,
    RETRIEVAL_NEO4J_USERNAME as NEO4J_USERNAME,
    RETRIEVAL_NEO4J_PASSWORD as NEO4J_PASSWORD,
    RETRIEVAL_OPENAI_API_KEY as OPENAI_API_KEY,
    RETRIEVAL_OPENAI_BASE_URL as OPENAI_BASE_URL,
    RETRIEVAL_EMBEDDING_MODEL as EMBEDDING_MODEL,
    RETRIEVAL_EMBEDDING_DIMENSIONS as EMBEDDING_DIMENSIONS,
)








openai_client = openai.OpenAI(
    api_key=OPENAI_API_KEY,
    base_url=OPENAI_BASE_URL
)


class SemanticGraphSearcher:
    def __init__(self, uri: str, username: str, password: str):
        self.driver = GraphDatabase.driver(uri, auth=(username, password))
        self.vector_dimensions = EMBEDDING_DIMENSIONS

    def close(self):
        self.driver.close()

    def get_embedding(self, text: str) -> List[float]:
        """单个文本的 embedding（降级用）"""
        try:
            response = openai_client.embeddings.create(
                model=EMBEDDING_MODEL,
                input=text,
                dimensions=EMBEDDING_DIMENSIONS
            )
            return response.data[0].embedding
        except Exception as e:
            print(f"❌ 获取 embedding 失败: {e}")
            return [0.0] * self.vector_dimensions

    def batch_get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """批量获取多个文本的 embedding，失败时降级为逐个调用"""
        if not texts:
            return []
        try:
            response = openai_client.embeddings.create(
                model=EMBEDDING_MODEL,
                input=texts,
                dimensions=EMBEDDING_DIMENSIONS
            )
            return [data.embedding for data in response.data]
        except Exception as e:
            print(f"❌ 批量获取 embedding 失败，降级为逐个处理: {e}")
            return [self.get_embedding(t) for t in texts]

    def cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        vec1 = np.array(vec1)
        vec2 = np.array(vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return np.dot(vec1, vec2) / (norm1 * norm2)

    def check_vector_index_exists(self, index_name: str, entity_type: str = "relationship") -> bool:
        with self.driver.session() as session:
            try:
                query = """
                SHOW INDEXES YIELD name, type
                WHERE type = 'VECTOR' AND name = $index_name
                RETURN count(*) as cnt
                """
                result = session.run(query, index_name=index_name)
                record = result.single()
                return record and record["cnt"] > 0
            except Exception:
                return False

    def search_similar_relations(self, question_embedding: List[float], top_k: int = config.VECTOR_RELATION_K) -> List[Dict]:
        relation_index_exists = self.check_vector_index_exists(config.RELATION_INDEX, "relationship")
        with self.driver.session() as session:
            try:
                if relation_index_exists:
                    query = """
                    CALL db.index.vector.queryRelationships($index_name, $top_k, $question_embedding)
                    YIELD relationship, score
                    RETURN startNode(relationship).name AS subject,
                           relationship.type AS predicate,
                           endNode(relationship).name AS object,
                           relationship.context AS context,
                           relationship.confidence AS confidence,
                           score AS similarity_score
                    ORDER BY score DESC
                    """
                    result = session.run(query, index_name=config.RELATION_INDEX, top_k=top_k, question_embedding=question_embedding)
                else:
                    query = """
                    MATCH (e1:Entity)-[r:RELATES]->(e2:Entity)
                    WHERE r.embedding IS NOT NULL AND size(r.embedding) = $dimensions
                    RETURN e1.name AS subject, r.type AS predicate, e2.name AS object,
                           r.context AS context, r.confidence AS confidence, r.embedding AS rel_emb
                    """
                    result = session.run(query, dimensions=self.vector_dimensions)
                    relations = []
                    for record in result:
                        sim = self.cosine_similarity(question_embedding, record['rel_emb'])
                        relations.append({
                            'subject': record['subject'],
                            'predicate': record['predicate'],
                            'object': record['object'],
                            'context': record['context'],
                            'confidence': record['confidence'],
                            'similarity_score': sim
                        })
                    relations.sort(key=lambda x: x['similarity_score'], reverse=True)
                    return relations[:top_k]

                relations = []
                for record in result:
                    relations.append({
                        'subject': record['subject'],
                        'predicate': record['predicate'],
                        'object': record['object'],
                        'context': record['context'],
                        'confidence': record['confidence'],
                        'similarity_score': float(record['similarity_score'])
                    })
                return relations
            except Exception as e:
                print(f"❌ 关系搜索失败: {e}")
                return []

    def split_into_sentences(self, text: str) -> List[str]:
        """将文本按句子分割（支持中英文标点：。！？；\n.）"""
        sentences = re.split(r'(?<=[。!?；\n\.])\s*', text)
        return [s.strip() for s in sentences if s.strip()]

    def retrieve_top_k_documents_and_texts(self, question_embedding: List[float], top_k: int = config.VECTOR_DOCUMENT_K) -> List[Dict]:
        """
        检索最相似的 top_k 个文档，对每个文档：
        - 若文档有多个段落（段落数>1），返回最相似的一个完整段落
        - 若文档只有一个段落，则拆分为句子，返回最相似的3个句子（按原顺序拼接）
        返回列表，每个元素包含：doc_info, best_text, seg_info
        """
        # 1. 检索最相似的 top_k 个文档
        doc_index_exists = self.check_vector_index_exists(config.DOCUMENT_INDEX, "document")
        docs_info = []  # 存储原始文档信息

        with self.driver.session() as session:
            try:
                if doc_index_exists:
                    query = """
                    CALL db.index.vector.queryNodes($index_name, $top_k, $question_embedding)
                    YIELD node, score
                    RETURN node.doc_id AS doc_id, node.title AS title, node.document_text AS document_text, score
                    """
                    result = session.run(query, index_name=config.DOCUMENT_INDEX, top_k=top_k, question_embedding=question_embedding)
                    for record in result:
                        docs_info.append({
                            'doc_id': record['doc_id'],
                            'title': record['title'],
                            'document_text': record['document_text'],
                            'similarity_score': record['score']
                        })
                else:
                    query = """
                    MATCH (d:Document)
                    WHERE d.embedding IS NOT NULL AND size(d.embedding) = $dimensions
                    RETURN d.doc_id AS doc_id, d.title AS title, d.document_text AS document_text, d.embedding AS emb
                    """
                    result = session.run(query, dimensions=self.vector_dimensions)
                    all_docs = []
                    for record in result:
                        sim = self.cosine_similarity(question_embedding, record['emb'])
                        all_docs.append({
                            'doc_id': record['doc_id'],
                            'title': record['title'],
                            'document_text': record['document_text'],
                            'similarity_score': sim,
                        })
                    all_docs.sort(key=lambda x: x['similarity_score'], reverse=True)
                    docs_info = all_docs[:top_k]
            except Exception as e:
                print(f"❌ 文档检索失败: {e}")
                return []

        # 2. 对每个文档提取最相似的段落/句子
        results = []
        for doc_info in docs_info:
            document_text = doc_info.get('document_text', '')
            if not document_text:
                results.append({
                    'doc_info': doc_info,
                    'best_text': None,
                    'seg_info': {}
                })
                continue

            # 分段
            paragraphs = []
            if '\n\n' in document_text:
                paragraphs = [p.strip() for p in document_text.split('\n\n') if p.strip()]
            if len(paragraphs) <= 1:
                paragraphs = [document_text.strip()]

            # 根据段落数量决定策略
            if len(paragraphs) > 1:
                # 多段落：批量获取所有段落的 embedding，找出最相似的一段
                para_embeddings = self.batch_get_embeddings(paragraphs)
                best_sim = -1.0
                best_idx = -1
                best_paragraph = None
                for i, emb in enumerate(para_embeddings):
                    if np.linalg.norm(emb) == 0:
                        continue
                    sim = self.cosine_similarity(question_embedding, emb)
                    if sim > best_sim:
                        best_sim = sim
                        best_paragraph = paragraphs[i]
                        best_idx = i
                seg_info = {
                    'mode': 'paragraphs',
                    'total_paragraphs': len(paragraphs),
                    'selected_paragraph_index': best_idx + 1
                }
                results.append({
                    'doc_info': doc_info,
                    'best_text': best_paragraph,
                    'seg_info': seg_info
                })
            else:
                # 单段落：拆分为句子，批量获取句子 embedding，返回最相似的3个句子（按原顺序）
                sentences = self.split_into_sentences(document_text)
                if not sentences:
                    results.append({
                        'doc_info': doc_info,
                        'best_text': None,
                        'seg_info': {}
                    })
                    continue

                sent_embeddings = self.batch_get_embeddings(sentences)
                sent_sims = []
                for i, (sent, emb) in enumerate(zip(sentences, sent_embeddings)):
                    if np.linalg.norm(emb) == 0:
                        sent_sims.append((i, sent, -1.0))
                    else:
                        sim = self.cosine_similarity(question_embedding, emb)
                        sent_sims.append((i, sent, sim))

                # 按相似度排序取前3，再按原顺序排列
                sent_sims.sort(key=lambda x: x[2], reverse=True)
                top3 = sent_sims[:3]
                top3_indices = [idx for idx, _, _ in top3]
                top3_sentences = [sent for _, sent, _ in top3]

                original_order = [sent for i, sent in enumerate(sentences) if i in top3_indices]
                best_text = ' '.join(original_order)

                selected_indices_sorted = sorted([i+1 for i in top3_indices])
                seg_info = {
                    'mode': 'sentences',
                    'total_sentences': len(sentences),
                    'selected_sentence_indices': selected_indices_sorted
                }
                results.append({
                    'doc_info': doc_info,
                    'best_text': best_text,
                    'seg_info': seg_info
                })
        return results

    def multi_hop_search(self, question_templates: List[str], top_k_relation: int = config.MULTIHOP_RELATION_K) -> List[Dict]:
        steps = []
        answers = []

        for idx, template in enumerate(question_templates):
            question = template
            for i, ans in enumerate(answers, start=1):
                placeholder = f"#{i}"
                if placeholder in question:
                    question = question.replace(placeholder, ans)

            q_emb = self.get_embedding(question)
            if np.linalg.norm(q_emb) == 0:
                print("❌ 问题向量获取失败，停止")
                break

            relations = self.search_similar_relations(q_emb, top_k=top_k_relation)
            if not relations:
                print("❌ 未检索到任何关系，停止多跳")
                break

            best_rel = max(relations, key=lambda r: r['similarity_score'])

            subj = best_rel['subject']
            obj = best_rel['object']

            subj_emb, obj_emb = self.batch_get_embeddings([subj, obj])
            sim_subj = self.cosine_similarity(q_emb, subj_emb)
            sim_obj = self.cosine_similarity(q_emb, obj_emb)

            if sim_subj < sim_obj:
                answer = subj
            else:
                answer = obj

            answers.append(answer)

            # 检索最相似的 top_k 个文档（默认2个）
            similar_docs = self.retrieve_top_k_documents_and_texts(q_emb, top_k=config.VECTOR_DOCUMENT_K)

            steps.append({
                'hop': idx + 1,
                'original_template': template,
                'resolved_question': question,
                'best_relation': {
                    'subject': best_rel['subject'],
                    'predicate': best_rel['predicate'],
                    'object': best_rel['object'],
                    'context': best_rel.get('context', ''),
                    'similarity_score': best_rel['similarity_score']
                },
                'answer': answer,
                'similar_documents': similar_docs   # 多个文档列表
            })

        return steps


def run_multi_hop_from_list(question_templates: List[str]) -> str:
    searcher = SemanticGraphSearcher(NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD)
    result_lines = []
    try:
        steps = searcher.multi_hop_search(question_templates, top_k_relation=config.MULTIHOP_RELATION_K)
        result_lines.append("")
        for step in steps:
            result_lines.append(f"第 {step['hop']} 跳问题: {step['resolved_question']}")
            result_lines.append("检索到的内容：")
            rel = step['best_relation']
            result_lines.append(f"向量匹配到的最佳关系: {rel['subject']} --[{rel['predicate']}]--> {rel['object']}")
            ctx = rel['context'] if rel['context'] else "（无上下文）"
            result_lines.append(f"context：{ctx}")
            # result_lines.append(f"第 {step['hop']} 向量匹配到的实体(不一定准确，也许下面 最相似文本里 或者 BFS检索结果 里能找到更合适的实体): {step['answer']}")
            result_lines.append(f"向量匹配到的相似实体(不一定准确，也许下面 最相似文本里 或者 BFS检索结果 里能找到更合适的实体): {step['answer']}")

            # 输出多个文档
            for i, doc_item in enumerate(step.get('similar_documents', []), start=1):
                doc = doc_item['doc_info']
                para = doc_item['best_text']
                seg_info = doc_item['seg_info']
                result_lines.append(f"向量匹配到的最相似文档 {i}: {doc['title']} (ID: {doc['doc_id']}, 相似度: {doc['similarity_score']:.4f})")
                if para:
                    if seg_info.get('mode') == 'paragraphs':
                        result_lines.append(f"  取自第 {seg_info['selected_paragraph_index']} / {seg_info['total_paragraphs']} 段落")
                    elif seg_info.get('mode') == 'sentences':
                        indices = seg_info['selected_sentence_indices']
                        result_lines.append(f"  取自句子索引 {indices} (共 {seg_info['total_sentences']} 句)")
                    result_lines.append(f"  文本: {para}")
                else:
                    result_lines.append("  文本: 未找到")
            result_lines.append("-" * 60)
    except Exception as e:
        result_lines.append(f"❌ 多跳问答出错: {e}")
    finally:
        searcher.close()
    return "\n".join(result_lines)


def neo4j_vertor(
        subquestions=config.EXAMPLE_NEO4J_VERTOR):
    # 解析参数字符串，格式：START_NODE_NAME:...;DEPTH:...;q1:...;q2:...;...
    if not subquestions:
        return "缺少参数 q1/q2/...，无法进行向量检索。"

    parts = subquestions.split(';')
    questions = []
    for part in parts:
        part = part.strip()
        if not part or ':' not in part:
            continue
        key, value = part.split(':', 1)
        key = key.strip()
        if key.startswith('q'):  # q1, q2, ...
            value = value.strip()
            if value:
                questions.append(value)

    if not questions:
        return "缺少参数 q1/q2/...，无法进行向量检索。"

    result_string = run_multi_hop_from_list(questions)
    return result_string


if __name__ == "__main__":
    print(neo4j_vertor())
