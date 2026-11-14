"""ARP-RAG 检索配置。相对路径以运行时工作目录为基准。"""
import os

# 主程序显式客户端与 SDK 环境变量原本使用不同配置。
SDK_API_KEY = os.getenv("SDK_API_KEY", "")
SDK_BASE_URL = "https://api.gptsapi.net/v1"
CHAT_API_KEY = os.getenv("CHAT_API_KEY", "")
CHAT_BASE_URL = "https://api.deepseek.com"
CHAT_MODEL = "deepseek-v4-pro"
DEEPSEEK_THINKING = os.getenv("DEEPSEEK_THINKING", "disabled").strip().lower()
DEEPSEEK_REASONING_EFFORT = os.getenv("DEEPSEEK_REASONING_EFFORT", "high")
MAIN_NEO4J_URI = "bolt://localhost:7687"
MAIN_NEO4J_USER = "请填写你自己的账号密码"
MAIN_请填写你自己的账号密码 = os.getenv("MAIN_请填写你自己的账号密码", "")
LANGCHAIN_TRACING_V2 = "true"
LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY", "")
DASHSCOPE_EMBEDDING_MODEL = "text-embedding-v1"
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY", "")
DASHSCOPE_API_BASE = "http://api.wlai.vip/v1"
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")
TAVILY_MAX_RESULTS = 2
START_LINE = 1
END_LINE = 2  # 处理前50条
INPUT_FILE = r'抽取/random_samples/random_50_samples.jsonl'
OUTPUT_FILE = r'图谱实验结果musique_deepseek-v4-pro/输出结果.xlsx'
ERROR_FILE = r'图谱实验结果musique_deepseek-v4-pro/错误输出.xlsx'

# 检索客户端
BFS_NEO4J_URI = "bolt://127.0.0.1:7687"
VECTOR_NEO4J_URI = "neo4j://localhost:7687"
RETRIEVAL_NEO4J_USERNAME = "请填写你自己的账号密码"
RETRIEVAL_请填写你自己的账号密码 = os.getenv("RETRIEVAL_请填写你自己的账号密码", "")
RETRIEVAL_OPENAI_API_KEY = os.getenv("RETRIEVAL_OPENAI_API_KEY", "")
RETRIEVAL_OPENAI_BASE_URL = "https://api.gptsapi.net/v1"
RETRIEVAL_EMBEDDING_MODEL = "text-embedding-3-large"
RETRIEVAL_EMBEDDING_DIMENSIONS = 1536

# 检索和实验参数
TEMPERATURE = 0
GRAPH_QUERY_LIMIT = 5
WEB_QUERY_LIMIT = 6
WEB_SEARCH_RESULTS = 10
WEB_DISPLAY_RESULTS = 5
DDG_RESULTS = 5
SEARCH_WORKERS = 2
BFS_DEPTH = 2
BFS_DEFAULT_MAX_NODES = 2000
BFS_MAX_NODES = 1000
START_NODE_DEFAULT_K = 1
START_NODE_SEARCH_K = 5
PRUNE_DEFAULT_K = 2
PRUNE_RELATION_K = 3
SUBGRAPH_FILE = "graph_rag_subgraph.json"
DEFAULT_SUBGRAPH_FILE = "subgraph.json"
BFS_NEIGHBOR_LIMIT = 1000
ENTITY_INDEX_CANDIDATES = [
        "entity_embeddings",
        "entity_embedding",
        "entities_embeddings",
        "node_embeddings",
    ]
VECTOR_RELATION_K = 5
VECTOR_DOCUMENT_K = 2
MULTIHOP_RELATION_K = 3
RELATION_INDEX = "relationship_embeddings"
DOCUMENT_INDEX = "document_embeddings"

DEFAULT_START_LINE = 1
DEFAULT_END_LINE = 3
EXAMPLE_NEO4J_ALLSEARCH = "START_NODE_NAME:Jim Wilson;DEPTH:3;q1:Which team released Jim Wilson?;q2:What organization includes the team #1?;q3:Who has the most hits in the history of #2?"
EXAMPLE_NEO4J_BFS = "START_NODE_NAME:Jim Wilson;DEPTH:2;q1:Which team released Jim Wilson?;q2:What organization includes the team #1?;q3:Who has the most hits in the history of #2?"
EXAMPLE_NEO4J_VERTOR = "START_NODE_NAME:Jim Wilson;DEPTH:2;q1:Which team released Jim Wilson?;q2:What organization includes the team #1?;q3:Who has the most hits in the history of #2?"
