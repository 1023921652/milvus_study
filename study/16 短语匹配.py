import time
from pymilvus import MilvusClient, DataType

# 1. 初始化客户端
client = MilvusClient(
    uri="http://localhost:19530",
    token="root:Milvus"
)

collection_name = "tech_articles"

# 重置环境（确保测试无脏数据）
if client.has_collection(collection_name=collection_name):
    client.drop_collection(collection_name=collection_name)

# 2. 创建 Schema
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

# 3. 准备索引参数
index_params = client.prepare_index_params()
index_params.add_index(
    field_name="embeddings",
    index_type="FLAT",
    metric_type="COSINE"
)
# 为文本字段创建倒排索引
index_params.add_index(
    field_name="text",
    index_name="text_inverted",
    index_type="INVERTED"
)

# 4. 创建 Collection 并绑定索引 (MilvusClient 在此时会自动在后台完成建索引和 Load 载入内存)
client.create_collection(
    collection_name=collection_name,
    schema=schema,
    index_params=index_params
)

# 5. 插入专门用于测试 slop 的模拟数据
data = [
    # doc_1: 完全紧邻且顺序一致（slop=0，应被 slop=1 和 slop=2 匹配）
    {
        "text": "We love learning machine deeply.",
        "embeddings": [0.1, 0.2, 0.3, 0.4, 0.5]
    },
    # doc_2: 包含 1 个词的间隔（slop=1，应被 slop=1 和 slop=2 匹配）
    {
        "text": "This is a learning to machine system.",
        "embeddings": [0.15, 0.25, 0.35, 0.45, 0.55]
    },
    # doc_3: 顺序相反的紧邻（计算编辑位移需要 2 步，应被 slop=2 匹配，而被 slop=1 排除）
    {
        "text": "This article is about machine learning.",
        "embeddings": [0.2, 0.3, 0.4, 0.5, 0.6]
    },
    # doc_4: 无关词 -> 两个测试中均不应匹配
    {
        "text": "Deep learning is a subset of artificial intelligence.",
        "embeddings": [0.9, 0.8, 0.7, 0.6, 0.5]
    }
]

client.insert(collection_name=collection_name, data=data)

# 6. 定义查询向量
query_vector = [0.1, 0.2, 0.3, 0.4, 0.5]


# ===================== 测试 1: SLOP = 1 =====================
filter_slop1 = "PHRASE_MATCH(text, 'learning machine', 1)"

result_slop1 = client.search(
    collection_name=collection_name,
    anns_field="embeddings",
    data=[query_vector],
    filter=filter_slop1,
    consistency_level="Strong",  # 关键：通过强一致性保证刚插入的数据在查询时立即可见，完全替代 flush()
    search_params={"params": {"nprobe": 10}},
    limit=10,
    output_fields=["id", "text"]
)

print("\n=== 【测试一】SLOP = 1 检索结果（预期匹配 2 条） ===")
for hits in result_slop1:
    for hit in hits:
        print(f"ID: {hit['id']}\t检索得分: {hit['distance']:.4f}\t文本: {hit['entity']['text']}")


# ===================== 测试 2: SLOP = 2 =====================
filter_slop2 = "PHRASE_MATCH(text, 'learning machine', 2)"

result_slop2 = client.search(
    collection_name=collection_name,
    anns_field="embeddings",
    data=[query_vector],
    filter=filter_slop2,
    consistency_level="Strong",  # 同样采用强一致性级别检索
    search_params={"params": {"nprobe": 10}},
    limit=10,
    output_fields=["id", "text"]
)

print("\n=== 【测试二】SLOP = 2 检索结果（预期匹配 3 条，包含颠倒的文档） ===")
for hits in result_slop2:
    for hit in hits:
        print(f"ID: {hit['id']}\t检索得分: {hit['distance']:.4f}\t文本: {hit['entity']['text']}")