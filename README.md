# ARP-RAG

ARP-RAG 是面向多跳问答的检索增强生成方法。它从候选文本中抽取实体和关系，建立带向量索引的 Neo4j 图谱，再通过关系检索、路径剪枝和查询重规划收集证据并生成答案。

## 仓库目录

| 路径 | 内容 |
| --- | --- |
| `data/` | MuSiQue、2WikiMultiHopQA 和 HotpotQA 的 JSONL 输入文件 |
| `extract/` | 实体与关系抽取、向量编码、Neo4j 批量导入 |
| `retrieve/` | LangGraph 问答流程、图检索和提示词 |
| `requirements.txt` | Python 依赖及版本 |

## 运行准备

原实验使用 Windows 和 Python 3.11.10，软件包版本见 [依赖清单](requirements.txt)。运行前需准备支持向量索引的 Neo4j 数据库，以及生成模型和向量模型的 API 访问凭据。

在仓库根目录打开 PowerShell，使用 Python 3.11 环境安装依赖。

```powershell
python -m pip install -r requirements.txt
```

在 [构图配置](extract/config.py) 中设置数据路径、抽取模型、向量模型和 Neo4j 连接信息。确保抽取输出与向量编码输入对应，导入脚本能读取抽取结果和向量文件。在 [检索配置](retrieve/config.py) 中设置同一数据集对应的数据库、模型接口及输入输出路径。构图和检索应使用相同的向量模型和维度。论文主实验采用 GPT-4o 和 1536 维的 text-embedding-3-large。

## 构图与问答

以下 PowerShell 命令从仓库根目录执行。每个数据集单独构图，同一数据集的问题共享图谱。

```powershell
python "./extract/file1_extract_entities1_musique英文提示词.py"
python "./extract/变向量1.py"
python "./extract/插入图谱4批量处理.py"
```

构图完成后，在检索配置中用 `INPUT_FILE` 指定问题文件，用 `START_LINE` 和 `END_LINE` 选择处理范围，再运行问答入口。

```powershell
python "./retrieve/langgraph_neo4j_匹配所有api.py"
```

结果和错误记录分别写入 `OUTPUT_FILE` 和 `ERROR_FILE` 指定的位置。

## 数据准备

ARP-RAG 使用三个公开的多跳问答数据集。实验输入包含问题、参考答案和候选文本，候选文本用于构图，问题和参考答案用于问答评测。下表列出原始数据的获取入口和本项目使用的 JSONL 文件名。

| 数据集 | 官方获取入口 | 项目输入路径 |
| --- | --- | --- |
| HotpotQA | [项目主页](https://hotpotqa.github.io/)，下载 distractor 设置的开发集 | `data/hotpot_validation.jsonl` |
| 2WikiMultiHopQA | [官方仓库](https://github.com/Alab-NII/2wikimultihop)，下载数据包并取开发集 | `data/2wiki_dev_processed.jsonl` |
| MuSiQue | [官方仓库](https://github.com/StonyBrookNLP/musique)，下载 MuSiQue-Ans 开发集 | `data/musique_dev_processed.jsonl` |

仓库目前提供源码，实验用的三个 JSONL 文件单独准备。官方入口提供原始开发集；复现论文所用子集还需对齐评测问题 ID、候选文本和输入格式。[HopRAG 仓库](https://github.com/LIU-Hao-2002/HopRAG) 提供这三个数据集的示例和预处理代码，可用于了解数据组织方式。

准备好实验输入后，在项目根目录创建 `data/`，按上表放置文件，并在构图与检索配置中指定相应输入路径。

## 凭据

API 密钥和数据库密码从配置文件中 `os.getenv` 指定的同名环境变量读取。
