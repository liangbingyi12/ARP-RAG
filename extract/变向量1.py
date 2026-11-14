import config as settings
import json
import openai
from openai import OpenAI
import numpy as np
from typing import List, Dict, Any
import time

# 配置OpenAI客户端
client = OpenAI(
    api_key=settings.EMBED_API_KEY,  # 请确保这是有效的API密钥
    base_url=settings.EMBED_BASE_URL
)


def get_embedding(text: str, model: str = settings.EMBED_MODEL, dimensions: int = settings.VECTOR_DIMENSIONS) -> List[
    float]:  # 修改点：更新默认模型，添加维度参数
    """
    获取单个文本的向量
    """
    try:
        response = client.embeddings.create(
            model=model,
            input=text,
            dimensions=dimensions  # 修改点：添加维度参数
        )
        return response.data[0].embedding
    except Exception as e:
        print(f"获取向量时出错: {e}")
        return None


def batch_get_embeddings(texts: List[str], model: str = settings.EMBED_MODEL,  # 修改点：更新默认模型
                         batch_size: int = settings.EMBED_DEFAULT_BATCH_SIZE, dimensions: int = settings.VECTOR_DIMENSIONS) -> List[List[float]]:  # 修改点：添加维度参数
    """
    批量获取文本向量
    """
    embeddings = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        try:
            response = client.embeddings.create(
                model=model,
                input=batch,
                dimensions=dimensions  # 修改点：添加维度参数
            )
            batch_embeddings = [data.embedding for data in response.data]
            embeddings.extend(batch_embeddings)

            # 添加延迟以避免速率限制
            time.sleep(settings.EMBED_DELAY_SECONDS)

            print(f"已处理 {min(i + batch_size, len(texts))}/{len(texts)} 个文本")

        except Exception as e:
            print(f"处理批次 {i // batch_size} 时出错: {e}")
            # 如果批量失败，尝试逐个处理
            for text in batch:
                embedding = get_embedding(text, model, dimensions)  # 修改点：传递维度参数
                # 动态生成对应维度的零向量作为占位符
                placeholder_vector = [0.0] * dimensions if dimensions else [0.0] * settings.VECTOR_DIMENSIONS  # 修改点：修复硬编码维度
                embeddings.append(embedding if embedding else placeholder_vector)

    return embeddings


def process_jsonl_file(input_path: str, output_path: str, embedding_model: str = settings.EMBED_MODEL,
                       dimensions: int = settings.VECTOR_DIMENSIONS):  # 修改点：添加模型和维度参数
    """
    处理JSONL文件，为所有embedding_texts生成向量
    """
    all_texts = []
    text_metadata = []
    documents = []

    print("正在读取JSONL文件...")
    with open(input_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            try:
                data = json.loads(line.strip())
                documents.append(data)

                doc_id = data['doc_id']
                doc_text = data['embedding_texts']['document_embedding_text']
                all_texts.append(doc_text)
                text_metadata.append({
                    'doc_id': doc_id,
                    'type': 'document',
                    'index': len(all_texts) - 1,
                    'text': doc_text
                })

                for i, entity_text in enumerate(data['embedding_texts']['entity_embedding_texts']):
                    all_texts.append(entity_text)
                    text_metadata.append({
                        'doc_id': doc_id,
                        'type': 'entity',
                        'entity_index': i,
                        'index': len(all_texts) - 1,
                        'text': entity_text
                    })

                for i, relation_text in enumerate(data['embedding_texts']['relation_embedding_texts']):
                    all_texts.append(relation_text)
                    text_metadata.append({
                        'doc_id': doc_id,
                        'type': 'relation',
                        'relation_index': i,
                        'index': len(all_texts) - 1,
                        'text': relation_text
                    })

            except json.JSONDecodeError as e:
                print(f"解析第 {line_num} 行时出错: {e}")

    print(f"总共收集了 {len(all_texts)} 个文本需要转换为向量")
    print(f"正在使用模型 {embedding_model} (维度: {dimensions}) 获取文本向量...")  # 修改点：提示信息

    # 修改点：调用时传入指定的模型和维度
    embeddings = batch_get_embeddings(all_texts, model=embedding_model, dimensions=dimensions)

    print("正在构建结果...")
    results = []

    for doc in documents:
        doc_id = doc['doc_id']
        doc_result = {
            'doc_id': doc_id,
            'title': doc['title'],
            'embeddings': {
                'document_embedding': None,
                'entity_embeddings': [],
                'relation_embeddings': []
            },
            'embedding_texts': doc['embedding_texts'],
            'model_used': embedding_model,  # 修改点：记录使用的模型
            'dimensions': dimensions  # 修改点：记录向量维度
        }

        doc_metadata = [m for m in text_metadata if m['doc_id'] == doc_id]

        doc_meta = next((m for m in doc_metadata if m['type'] == 'document'), None)
        if doc_meta and embeddings[doc_meta['index']]:
            doc_result['embeddings']['document_embedding'] = embeddings[doc_meta['index']]

        entity_metas = sorted([m for m in doc_metadata if m['type'] == 'entity'],
                              key=lambda x: x['entity_index'])
        for meta in entity_metas:
            if embeddings[meta['index']]:
                doc_result['embeddings']['entity_embeddings'].append({
                    'text': meta['text'],
                    'embedding': embeddings[meta['index']]
                })

        relation_metas = sorted([m for m in doc_metadata if m['type'] == 'relation'],
                                key=lambda x: x['relation_index'])
        for meta in relation_metas:
            if embeddings[meta['index']]:
                doc_result['embeddings']['relation_embeddings'].append({
                    'text': meta['text'],
                    'embedding': embeddings[meta['index']]
                })

        results.append(doc_result)

    print(f"正在保存结果到 {output_path}...")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    output_jsonl_path = output_path.replace('.json', '.jsonl')
    with open(output_jsonl_path, 'w', encoding='utf-8') as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')

    vectors_path = output_path.replace('.json', '_vectors.npz')
    vectors_data = {}
    for i, (text, embedding) in enumerate(zip(all_texts, embeddings)):
        vectors_data[f'text_{i}'] = text
        vectors_data[f'vector_{i}'] = np.array(embedding)

    np.savez(vectors_path, **vectors_data)

    print(f"处理完成！")
    print(f"JSON结果保存到: {output_path}")
    print(f"JSONL结果保存到: {output_jsonl_path}")
    print(f"纯向量数据保存到: {vectors_path}")
    print(f"\n统计信息:")
    print(f"- 处理文档数: {len(documents)}")
    print(f"- 总文本数: {len(all_texts)}")
    print(f"- 使用模型: {embedding_model}")
    print(f"- 向量维度: {dimensions}")
    print(f"- 成功获取向量数: {sum(1 for e in embeddings if e)}")

    return results


if __name__ == "__main__":
    input_file = settings.EMBED_INPUT_FILE
    output_file = settings.EMBED_OUTPUT_FILE

    # 修改点：在这里指定模型和维度，非常灵活
    results = process_jsonl_file(
        input_file,
        output_file,
        embedding_model=settings.EMBED_MODEL,  # 可轻松切换模型
        dimensions=settings.VECTOR_DIMENSIONS  # 可轻松调整维度
    )