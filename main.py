
# 在启用分区关键字功能的 Collections 中进行 ANN 搜索时，需要在搜索请求中包含涉及分区关键字的过滤表达式。在过滤表达式中，你可以将 Partition Key 的值限制在特定范围内，这样 Milvus 就会将搜索范围限制在相应的分区内。
# 启用 "分区密钥隔离 "后，必须在基于分区密钥的过滤条件中只包含一个特定值，这样 Milvus 才能在匹配的索引所包含的实体内限制搜索范围。


import time
from pymilvus import MilvusClient, DataType

# 1. 初始化客户端
client = MilvusClient(
    uri="http://localhost:19530",
    token="root:Milvus"
)

collection_name = "my_collection"

# 重置测试环境
if client.has_collection(collection_name):
    client.drop_collection(collection_name)

# 2. 定义 Schema
schema = client.create_schema(enable_dynamic_field=False)

# 修正 1：设置 auto_id=True 以配合模拟数据自动生成 ID
schema.add_field(
    field_name="id",
    datatype=DataType.INT64,
    is_primary=True,
    auto_id=True
)

schema.add_field(
    field_name="vector",
    datatype=DataType.FLOAT_VECTOR,
    dim=5
)

schema.add_field(
    field_name="my_varchar",
    datatype=DataType.VARCHAR,
    max_length=512,
    is_partition_key=True, # 声明为分区键
)
# 3. 准备索引参数 (提前定义)
index_params = client.prepare_index_params()
index_params.add_index(
    field_name="vector",
    index_type="FLAT",
    metric_type="COSINE"
)
# 3. 创建 Collection 开启分区键隔离属性
client.create_collection(
    collection_name=collection_name,
    schema=schema,
    index_params=index_params,  # 修复：建表时直接绑定，避免后续调用 create_index 同步卡死
    num_partitions=1024,
    properties={"partitionkey.isolation": True} # 开启分区隔离
)

# 4. 插入代表性的多租户模拟数据
# 包含不同的 my_varchar (分区键) 属性，代表不同的租户
data = [
    {"vector": [0.1, 0.2, 0.3, 0.4, 0.5], "my_varchar": "tenant_a"},
    {"vector": [0.15, 0.25, 0.35, 0.45, 0.55], "my_varchar": "tenant_a"},
    {"vector": [0.2, 0.3, 0.4, 0.5, 0.6], "my_varchar": "tenant_b"},
    {"vector": [0.9, 0.8, 0.7, 0.6, 0.5], "my_varchar": "tenant_c"}
]

insert_res = client.insert(collection_name=collection_name, data=data)
print(f"数据插入成功，生成的主键列表：{insert_res['ids']}")

# 5. 补充索引构建（向量检索的前置条件）


# 6. 载入 Collection 进内存
client.load_collection(collection_name=collection_name)
time.sleep(1) # 稍微等待载入就绪

# 7. 定义测试用的查询向量
query_vector = [0.1, 0.2, 0.3, 0.4, 0.5]

# 8. 执行分区隔离检索
# 核心验证点：过滤表达式中只包含分区键的一个特定值 "tenant_a"
filter_expr = "my_varchar == 'tenant_a'"

result = client.search(
    collection_name=collection_name,
    anns_field="vector",
    data=[query_vector],
    filter=filter_expr,  # 传入隔离过滤规则
    limit=10,
    consistency_level="Bounded", # 强一致性保障刚插入的数据可见
    output_fields=["id", "my_varchar"]
)

# 9. 打印结果
# 预期表现：检索会瞬间路由至 tenant_a 专属的索引分段上，完全绕过 tenant_b 和 tenant_c 的索引和数据，且结果中只包含 tenant_a。
print("\n=== 【测试】分区键隔离检索结果（预期仅召回属于 tenant_a 的数据） ===")
for hits in result:
    for hit in hits:
        print(f"ID: {hit['id']}\t检索得分: {hit['distance']:.4f}\t租户标识 (Partition Key): {hit['entity']['my_varchar']}")