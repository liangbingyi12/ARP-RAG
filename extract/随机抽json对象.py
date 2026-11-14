# 随机抽json对象.py （适配 HotpotQA）
import config as settings
import json
import random
import os

BASE_DIR = settings.DATA_DIR

def process_hotpot_file(file_path, sample_size=settings.SAMPLE_DEFAULT_SAMPLE_SIZE):
    try:
        all_objects = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue
                obj = json.loads(line)
                # 只保留有 context.title 和 context.sentences 的对象
                if 'context' in obj and 'title' in obj['context'] and 'sentences' in obj['context']:
                    all_objects.append(obj)

        total = len(all_objects)
        print(f"共有 {total} 个有效 HotpotQA 对象")

        n = min(sample_size, total)
        selected = random.sample(all_objects, n)
        print(f"随机选择了 {n} 个对象")

        # 输出到 抽取/random_samples/ 下
        out_dir = os.path.join(BASE_DIR, settings.SAMPLE_FOLDER)
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, settings.SAMPLE_FILENAME)
        with open(out_path, 'w', encoding='utf-8') as f:
            for obj in selected:
                f.write(json.dumps(obj, ensure_ascii=False) + '\n')

        print(f"已保存至: {out_path}")
        return selected

    except Exception as e:
        print(f"错误: {e}")
        return None

if __name__ == "__main__":
    # Hotpot 原始验证集路径（请确认该文件已移入 抽取 文件夹内）
    input_file = os.path.join(BASE_DIR, settings.HOTPOT_FOLDER, settings.HOTPOT_FILENAME)
    process_hotpot_file(input_file, sample_size=settings.SAMPLE_SAMPLE_SIZE)