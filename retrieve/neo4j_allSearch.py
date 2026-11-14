import config
from concurrent.futures import ThreadPoolExecutor
from neo4j_BFS import neo4j_BFS
from neo4j_vertor import neo4j_vertor


def neo4j_allSearch(
        subquestions=config.EXAMPLE_NEO4J_ALLSEARCH):
    with ThreadPoolExecutor(max_workers=config.SEARCH_WORKERS) as executor:
        future_vector = executor.submit(neo4j_vertor, subquestions)
        future_bfs = executor.submit(neo4j_BFS, subquestions)

        # 获取结果（若某个函数异常，可根据需要处理）
        vector_result = future_vector.result()
        bfs_result = future_bfs.result()

    result = f"\nneo4j_vertor向量匹配检索结果：\n{vector_result}\nneo4j_BFS广度优先检索结果：\n{bfs_result}\n"
    print(result)
    return result


if __name__ == "__main__":
    print(neo4j_allSearch())