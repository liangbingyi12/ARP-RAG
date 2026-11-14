# 拿json对象文本.py （适配 HotpotQA）
import config as settings
import json
import os

BASE_DIR = settings.DATA_DIR

def convert_hotpot_to_docs(input_file, output_file):
    """
    读取 HotpotQA 格式的 jsonl，把每个 context.title 和 context.sentences
    拼成完整文档，写成 {"id": "...", "title": "...", "text": "..."} 的 jsonl。
    """
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    doc_count = 0
    with open(input_file, 'r', encoding='utf-8') as fin, \
         open(output_file, 'w', encoding='utf-8') as fout:

        for line_num, line in enumerate(fin, 1):
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"第 {line_num} 行 JSON 解析错误: {e}")
                continue

            qid = obj.get('id') or obj.get('_id')
            if not qid:
                print(f"第 {line_num} 行缺少 id，跳过")
                continue

            context = obj.get('context')
            if not context:
                continue

            titles = context.get('title', [])
            sentences_list = context.get('sentences', [])

            if len(titles) != len(sentences_list):
                print(f"第 {line_num} 行 title 与 sentences 长度不一致，跳过")
                continue

            for idx, (title, sents) in enumerate(zip(titles, sentences_list)):
                full_text = ' '.join(sents).strip()
                if not full_text:
                    continue

                doc = {
                    "id": f"{qid}_{idx}",   # 唯一文档id
                    "title": title,
                    "text": full_text
                }
                fout.write(json.dumps(doc, ensure_ascii=False) + '\n')
                doc_count += 1

    print(f"已生成 {doc_count} 个文档，保存在: {output_file}")
    return doc_count

if __name__ == "__main__":
    # 输入：上一步随机抽样得到的文件（保证还在 抽取 内）
    input_path = os.path.join(BASE_DIR, settings.SAMPLE_FOLDER, settings.SAMPLE_FILENAME)
    # 输出：沿用原来的文件名 combined_contexts.jsonl，放在 random_samples 下
    output_path = os.path.join(BASE_DIR, settings.SAMPLE_FOLDER, settings.CONTEXT_FILENAME)
    convert_hotpot_to_docs(input_path, output_path)