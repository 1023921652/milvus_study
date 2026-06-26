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
# 一个 Collection 可以有多个别名。
# 一个别名一次只能指向一个 Collections。
# 处理请求时，Milvus 会首先检查是否存在提供名称的 Collection。如果不存在，它就会检查该名称是否是某个 Collection 的别名。
client.create_alias(
    collection_name="demo_collection",
    alias="bob"
)
# 列出别名
res = client.list_aliases(
    collection_name="demo_collection"
)

print(res)
# 描述别名
res = client.describe_alias(
    alias="bob"
)

print(res)
# 您可以将已分配给特定集合的别名重新分配给另一个集合。
client.alter_alias(
    collection_name="my_collection_2",
    alias ="alice"
)


# 9.5 Drop aliases
client.drop_alias(
    alias= "bob"
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