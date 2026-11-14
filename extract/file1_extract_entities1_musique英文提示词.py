# file1_extract_entities_wiki.py
import config as settings
import json
import os
from typing import List, Dict, Any
from openai import OpenAI
import re
from tqdm import tqdm

# OpenAI API配置
api_key = settings.EXTRACT_API_KEY
base_url = settings.EXTRACT_BASE_URL
model = settings.EXTRACT_MODEL

dataset_path = settings.EXTRACT_DATASET_PATH
output_file = settings.EXTRACT_OUTPUT_FILE

# 包含首端和尾端
start_item = settings.EXTRACT_START_ITEM
end_item = settings.EXTRACT_END_ITEM  # 处理到文件末尾


class EntityRelationExtractor:
    """实体关系抽取器"""

    def __init__(self, api_key: str, base_url: str):
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url
        )
        self.entity_type_mapping = {
            "PERSON": "人物",
            "ORG": "组织",
            "LOC": "地点",
            "DATE": "日期",
            "TIME": "时间",
            "WORK_OF_ART": "作品",
            "EVENT": "事件",
            "OTHER": "其他"
        }

    def extract_from_text(self, text: str, title: str = "") -> Dict[str, Any]:
        """使用GPT-4o从文本中抽取实体和关系"""

        system_prompt = """You are a professional information extraction system. Please extract entities and the relationships between entities from the given text.

        Please return the results in the following JSON format:
        {
            "entities": [
                {
                    "name": "Entity name",
                    "type": "Entity type",
                    "normalized_name": "Standardized name (if applicable, such as full name or official name)",
                    "context": "Context in which the entity appears (brief description)"
                }
            ],
            "relations": [
                {
                    "subject": "Subject entity (prefer using normalized_name)",
                    "predicate": "Relation type",
                    "object": "Object entity (prefer using normalized_name)",
                    "confidence": "Confidence level (high / medium / low)",
                    "context": "Context in which the relation appears (1 sentences)"
                }
            ]
        }

        Entity type guidelines (prefer using these types, but you may create new ones if needed):
        - PERSON: Person (human names)
        - ORG: Organizations, institutions, companies, groups
        - LOC: Locations, places, cities, countries
        - DATE: Dates, years, months
        - TIME: Time, moments, periods
        - WORK_OF_ART: Works, artworks, books, albums, movies
        - EVENT: Events, activities, conferences
        - TITLE: Titles, positions, honorifics (e.g., President, CEO, Champion)
        - CONCEPT: Abstract concepts (e.g., theories, systems, laws)

        Relation type guidelines:
        - You may freely create appropriate relation types to describe relationships between entities
        - Relation types should be concise, accurate, and meaningful
        - Example relation types: born_in, created, belongs_to, participated_in, located_in, released_in, contains, member_of, married_on, graduated_from, employed_at, created_in, occurred_in, etc.

        Important notes:
        1. Ensure entity names are standardized (e.g., "Grant Green" instead of "Green")
        2. For entities with the same name, provide contextual information whenever possible
        3. Only extract entities and relationships explicitly mentioned in the text; do not add information based on common knowledge or guess relationships for the sake of a "complete chain"
        4. For people, use full names whenever possible
        5. For abbreviations, expand them to their full forms whenever possible
        6. Provide a brief context description (1 sentences) for each entity and relationship to support downstream vectorization
        7. If existing entity types cannot accurately describe an entity, you may create a new type
        8. Relation types may be freely created based on the text, but must remain concise and accurate
        9. Coreference resolution is mandatory: pronouns such as he / she / it / they / this city / that country / the former / the latter must be resolved to specific entities and use their normalized_name
        10. Definition of confidence:
            - High: explicitly and directly stated in the text
            - Medium: requires slight inference but is unambiguous
            - Low: some uncertainty exists, but there is textual evidence

        Output constraints (engineering-level requirements):
        Please ensure the output is returned strictly in JSON format. Do not include any explanatory text, comments, Markdown, or pre/post explanations."""

        user_message = f"""Please extract entities and relationships from the following text and return the results in JSON format:

        Text title: {title}

        Text content:
        {text}

        Please return the extracted results in the specified JSON format."""

        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message}
                ],
                temperature=settings.EXTRACT_TEMPERATURE,
                response_format={"type": "json_object"}
            )

            result_json = json.loads(response.choices[0].message.content)
            return result_json

        except Exception as e:
            print(f"抽取过程中发生错误: {e}")
            return {"entities": [], "relations": []}

    def clean_entity_name(self, name: str) -> str:
        """清理实体名称"""
        if not name:
            return ""
        name = name.strip()
        name = re.sub(r'\s+', ' ', name)
        name = name.strip('"\'')

        if name and name[0].isalpha():
            words = name.split()
            if len(words) > 1:
                capitalized_words = []
                for word in words:
                    if word and word[0].isalpha():
                        capitalized_words.append(word[0].upper() + word[1:].lower())
                    else:
                        capitalized_words.append(word)
                name = ' '.join(capitalized_words)
            else:
                name = name[0].upper() + name[1:].lower()

        return name

    def normalize_entity_name(self, entity_data: Dict) -> str:
        """标准化实体名称"""
        normalized_name = entity_data.get("normalized_name", "")
        if normalized_name and normalized_name.strip():
            return self.clean_entity_name(normalized_name)
        else:
            return self.clean_entity_name(entity_data.get("name", ""))

    def generate_embedding_texts(self, title: str, text: str, extraction_result: Dict[str, Any]) -> Dict[str, Any]:
        """根据抽取结果生成三种embedding文本"""

        # 1. 文档节点的embedding文本
        document_embedding_text = f"{title}: {text[:5000]}"

        # 2. 实体节点的embedding文本列表
        entity_embedding_texts = []
        for entity in extraction_result.get("entities", []):
            entity_name = entity.get("normalized_name", entity.get("name", ""))
            entity_type = entity.get("type", "OTHER")
            context = entity.get("context", "")

            if context:
                entity_embedding = f"{entity_name} ({entity_type}): {context}"
            else:
                entity_embedding = f"{entity_name} ({entity_type})"

            entity_embedding_texts.append(entity_embedding)

        # 3. 关系节点的embedding文本列表
        relation_embedding_texts = []
        for relation in extraction_result.get("relations", []):
            subject = relation.get("subject", "")
            predicate = relation.get("predicate", "")
            obj = relation.get("object", "")
            context = relation.get("context", "")

            if subject and predicate and obj:
                if context:
                    relation_embedding = f"{subject} {predicate} {obj} ({context})"
                else:
                    relation_embedding = f"{subject} {predicate} {obj}"

                relation_embedding_texts.append(relation_embedding)

        return {
            "document_embedding_text": document_embedding_text,
            "entity_embedding_texts": entity_embedding_texts,
            "relation_embedding_texts": relation_embedding_texts
        }


def extract_entities_to_file(dataset_path: str, output_file: str, start_item: int = settings.EXTRACT_DEFAULT_START_ITEM, end_item: int = settings.EXTRACT_DEFAULT_END_ITEM):
    """抽取实体关系并保存到文件"""

    if not os.path.exists(dataset_path):
        print(f"文件不存在: {dataset_path}")
        return

    extractor = EntityRelationExtractor(api_key, base_url)

    if os.path.exists(output_file):
        print(f"输出文件 {output_file} 已存在，将在末尾追加结果")
    else:
        print(f"输出文件 {output_file} 不存在，将创建新文件")

    with open(dataset_path, 'r', encoding='utf-8') as f, \
            open(output_file, 'a', encoding='utf-8') as out_f:

        lines = f.readlines()
        total_lines = len(lines)

        if end_item is None:
            end_item = total_lines
        if end_item > total_lines:
            end_item = total_lines

        lines_to_process = lines[start_item - 1:end_item]
        total_to_process = len(lines_to_process)

        print(f"总数据: {total_lines} 条")
        print(f"抽取范围: 第 {start_item} 条到第 {end_item} 条（共 {total_to_process} 条）")

        items_processed = 0
        for line_index, line in enumerate(tqdm(lines_to_process, desc="抽取实体关系")):
            try:
                data = json.loads(line.strip())
                # 关键修改：兼容 _id 和 id，统一转为 doc_id
                doc_id = data.get("_id") or data.get("id")
                if doc_id is None:
                    # 如果都没有，尝试使用 idx 或跳过
                    doc_id = data.get("idx", f"line_{start_item + line_index}")
                    print(f"警告: 第 {start_item + line_index} 行没有 id 或 _id，使用 {doc_id}")

                title = data.get("title", "")
                text = data.get("text", "")

                if not text:
                    print(f"警告: 文档 {doc_id} 没有 text 字段，跳过")
                    continue

                original_line_num = start_item + line_index

                print(f"\n处理第 {original_line_num} 行（文档ID: {doc_id}）: {title[:50]}")

                # 抽取实体关系
                extraction_result = extractor.extract_from_text(text, title)

                # 标准化实体名称
                for entity in extraction_result.get("entities", []):
                    entity["normalized_name"] = extractor.normalize_entity_name(entity)

                for relation in extraction_result.get("relations", []):
                    relation["subject"] = extractor.normalize_entity_name({"name": relation.get("subject", "")})
                    relation["object"] = extractor.normalize_entity_name({"name": relation.get("object", "")})

                # 生成embedding文本
                embedding_texts = extractor.generate_embedding_texts(title, text, extraction_result)

                # 保存结果（不包含原始text字段）
                result_data = {
                    "doc_id": doc_id,          # 统一使用 doc_id
                    "title": title,
                    "embedding_texts": embedding_texts,
                    "extraction_result": extraction_result,
                    "original_line": original_line_num
                }

                out_f.write(json.dumps(result_data, ensure_ascii=False) + '\n')
                items_processed += 1

            except json.JSONDecodeError as e:
                print(f"JSON解析错误: {e}")
            except Exception as e:
                print(f"处理文档时发生错误: {e}")

    print(f"\n抽取完成！共处理 {items_processed} 个文档")
    print(f"结果保存到: {output_file}")


if __name__ == "__main__":
    extract_entities_to_file(dataset_path, output_file, start_item=start_item, end_item=end_item)