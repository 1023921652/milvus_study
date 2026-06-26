from pymilvus import MilvusClient
import sys
import os

# 修复 milvus-lite 在 Windows 平台上 os.rename 无法覆盖已有文件的系统级 Bug
if sys.platform == "win32":
    os.rename = os.replace
# 单数据库局限性：Milvus Lite 专为本地快速原型设计、笔记本开发和 Jupyter Notebook 演示设计，为了保持极致轻量和极简，Milvus Lite 默认且仅支持单一数据库状态
client = MilvusClient(
    uri="http://localhost:19530",
    token= "root:Milvus"
)
client.create_database(
    db_name="my_database_2",
    properties={
        "database.replica.number": 3
    }
)
client.list_databases()


client.describe_database(
    db_name="default"
)

