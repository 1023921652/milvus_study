# 文本匹配可与向量相似性搜索结合使用，以缩小搜索范围并提高搜索性能。通过在向量相似性搜索前使用文本匹配过滤 Collections，可以减少需要搜索的文档数量，从而加快查询速度。
#
# 在本例中，filter 表达式过滤了搜索结果，使其只包含与指定术语keyword1 或keyword2 匹配的文档。然后在此过滤后的文档子集中执行向量相似性搜索。
import random
from pymilvus import MilvusClient, DataType

# 1. 初始化客户端
# 如果没有运行本地的 Milvus Docker，可以变更为 client = MilvusClient("milvus_demo.db") 运行本地轻量版
client = MilvusClient(
    uri="http://localhost:19530",
    token="root:Milvus"
)

collection_name = "my_collection"

# 为了反复测试，如果已存在该集合则先删除
if client.has_collection(collection_name):
    client.drop_collection(collection_name)

# 2. 定义 Schema 并修复重复定义 'text' 字段的问题
schema = client.create_schema(enable_dynamic_field=False)
schema.add_field(
    field_name="id",
    datatype=DataType.INT64,
    is_primary=True,
    auto_id=True
)

analyzer_params = {
    "type": "english"
}

# 合并后的 text 字段定义
schema.add_field(
    field_name='text',
    datatype=DataType.VARCHAR,
    max_length=1000,
    enable_analyzer=True,
    analyzer_params=analyzer_params,
    enable_match=True
)

schema.add_field(
    field_name="embeddings",
    datatype=DataType.FLOAT_VECTOR,
    dim=5
)

# 3. 创建 Collection
client.create_collection(
    collection_name=collection_name,
    schema=schema
)

# 4. 准备模拟数据并插入
# 准备了4条数据：2条包含 keyword1，1条包含 keyword2，1条无关。因为 auto_id=True，数据中无需包含 "id"
data = [
    {
        "text": "This document introduces the key concepts of keyword1 and data structures.",
        "embeddings": [0.1, 0.2, 0.3, 0.4, 0.5]
    },
    {
        "text": "Another document that contains keyword2 for system verification.",
        "embeddings": [0.15, 0.25, 0.35, 0.45, 0.55]
    },
    {
        "text": "This is a random sentence without any special matching vocabulary.",
        "embeddings": [0.9, 0.8, 0.7, 0.6, 0.5]
    },
    {
        "text": "Let us discuss keyword1 again in the context of retrieval models.",
        "embeddings": [0.12, 0.22, 0.32, 0.42, 0.52]
    }
]

insert_result = client.insert(
    collection_name=collection_name,
    data=data
)
print(f"模拟数据插入成功，生成的主键 ID 列表：{insert_result['ids']}")

# 5. 创建索引 (向量字段和文本匹配字段)
index_params = client.prepare_index_params()

# 为向量字段创建索引 (因为 dim 为 5，用 FLAT 索引最适合演示)
index_params.add_index(
    field_name="embeddings",
    index_type="FLAT",
    metric_type="COSINE"
)

# 为文本字段创建标量倒排索引，加速 TEXT_MATCH
index_params.add_index(
    field_name="text",
    index_type="INVERTED"
)

client.create_index(
    collection_name=collection_name,
    index_params=index_params
)

# 6. 加载 Collection 到内存
client.load_collection(collection_name=collection_name)

# 7. 定义检索所需的查询向量和文本匹配规则
query_vector = [0.1, 0.2, 0.3, 0.4, 0.5]
filter_expr = "TEXT_MATCH(text, 'keyword1 keyword2')"

# 8. 执行混合检索：文本过滤 + 向量相似度搜索
result = client.search(
    collection_name=collection_name,
    anns_field="embeddings",
    data=[query_vector],
    filter=filter_expr,
    search_params={"params": {"nprobe": 10}},
    limit=10,
    output_fields=["id", "text"]
)

# 9. 打印检索结果
print("\n=== 检索结果 (应该只包含出现 keyword1 或 keyword2 的文档) ===")
for hits in result:
    for hit in hits:
        print(f"ID: {hit['id']}\t评分 (Similarity): {hit['distance']:.4f}\t文本内容: {hit['entity']['text']}")