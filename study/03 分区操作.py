from pymilvus import MilvusClient

import sys
import os

# 修复 milvus-lite 在 Windows 平台上 os.rename 无法覆盖已有文件的系统级 Bug
if sys.platform == "win32":
    os.rename = os.replace

client = MilvusClient("milvus_demo.db")

if client.has_collection(collection_name="demo_collection"):
    client.drop_collection(collection_name="demo_collection")
client.create_collection(
    collection_name="demo_collection",
    dimension=768,  # The vectors we will use in this demo has 768 dimensions
)
client.create_partition(
    collection_name="demo_collection",
    partition_name="partitionA"
)
res = client.list_partitions(
    collection_name="demo_collection"
)

# 检查是否存在特定分区
res = client.has_partition(
    collection_name="demo_collection",
    partition_name= "partitionA"
)
# 加载指定分区
# 如果一个集合中有 5 个分区，你只加载了其中的 2 个。此时，虽然那 2 个分区在后台已经可以正常检索，但“集合”这个整体在系统层面依然会被标记为“未加载状态（NotLoaded）”
client.load_partitions(
    collection_name="demo_collection",
    partition_names=["partitionA"]
)

res = client.get_load_state(
    collection_name="demo_collection",
    partition_name= "partitionA"
)

print(res)
# 释放分区
client.release_partitions(
    collection_name="my_collection",
    partition_names=[ "partitionA"]
)
# 删除分区前先释放分区
client.drop_partition(
    collection_name="my_collection",
    partition_name="partitionA"
)
#
# from pymilvus import model
#
#
# embedding_fn = model.DefaultEmbeddingFunction()
#
# docs = [
#     "Artificial intelligence was founded as an academic discipline in 1956.",
#     "Alan Turing was the first person to conduct substantial research in AI.",
#     "Born in Maida Vale, London, Turing was raised in southern England.",
# ]
#
# vectors = embedding_fn.encode_documents(docs)
# print("Dim:", embedding_fn.dim, vectors[0].shape)  # Dim: 768 (768,)
#
# data = [
#     {"id": i, "vector": vectors[i], "text": docs[i], "subject": "history"}
#     for i in range(len(vectors))
# ]
#
# # print("Data has", len(data), "entities, each with fields: ", data[0].keys())
# # print("Vector dim:", len(data[0]["vector"]))
#
# res = client.insert(collection_name="demo_collection", data=data)
#
# print(res)
#
# query_vectors = embedding_fn.encode_queries(["Who is Alan Turing?"])
#
# res = client.search(
#     collection_name="demo_collection",  # target collection
#     data=query_vectors,  # query vectors
#     limit=2,  # number of returned entities
#     output_fields=["text", "subject"],  # specifies fields to be returned
# )
#
# print(res)