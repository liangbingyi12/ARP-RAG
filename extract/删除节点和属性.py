# clear_neo4j.py
import config as settings
from neo4j import GraphDatabase

# Neo4j 连接配置（请与你的实际配置保持一致）
uri = settings.NEO4J_URI
username = settings.NEO4J_USERNAME
password = settings.NEO4J_PASSWORD

def clear_database(driver):
    """删除数据库中所有节点和关系"""
    with driver.session() as session:
        # DETACH DELETE 会先删除关系再删除节点
        result = session.run("MATCH (n) DETACH DELETE n")
        # 获取删除的节点数（可选）
        summary = result.consume()
        print("✅ 数据库已清空（所有节点和关系已删除）")

def main():
    try:
        driver = GraphDatabase.driver(uri, auth=(username, password))
        print("🔗 已连接到 Neo4j 数据库")
        clear_database(driver)
    except Exception as e:
        print(f"❌ 清空数据库时出错: {e}")
    finally:
        if 'driver' in locals():
            driver.close()
            print("🔒 数据库连接已关闭")

if __name__ == "__main__":
    main()