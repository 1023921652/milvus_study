# 搜索迭代器（Search Iterator）的设计初衷
# 普通检索（collection.search）：设计目的是为了快速获取前几条（如 Top-5、Top-10）最精准的相似结果。
# 搜索迭代器（search_iterator）：设计目的是为了在大规模、宽范围的相似结果集下进行安全的分页拉取（例如，当您需要拉取前 10,000 个相似点进行离线重排或进一步的业务过滤，直接一次性拉取会挤爆内存。迭代器就像一个游标，通过 next() 每次只给您返回 50 条）。
#
# 是的，搜索迭代器（Search Iterator）返回的结果是严格按照相似性排名全局有序返回的。
# 无论是单个批次（Batch）内部，还是不同批次（Page）之间，都严格遵循**相似度由高到低（如果是 L2 距离，则是距离由近到远）**的顺序。
# 具体表现在以下几个方面：
# 1. 跨批次的全局有序性（Batch-to-Batch Ordering）
# 当您连续调用 iterator.next() 分页拉取数据时，其相似度在宏观上是完全连续的：
# 第 1 页（1~50条）：包含的是全库中与查询向量最相似的第 1 到第 50 条数据。
# 第 2 页（51~100条）：包含的是次相似的第 51 到第 100 条数据。
# 第 3 页（101~150条）：则是相似度更低一些的第 101 到第 150 条数据。
# 绝对不会出现“第二页中某条数据的相似度高于第一页”的情况。
import random
import time
from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType, utility

# 1. 连接 Milvus 服务
connections.connect(
    uri="http://localhost:19530",
    token="root:Milvus"
)

collection_name = "iterator_collection"

# 为了保证测试环境干净，如果集合已存在则先删除
if utility.has_collection(collection_name):
    utility.drop_collection(collection_name)

# 2. 定义 Schema 并创建 Collection
fields = [
    # 显式定义主键 id，5维浮点向量字段 vector，以及输出的颜色标识字段 color
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=False),
    FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=5),
    FieldSchema(name="color", dtype=DataType.VARCHAR, max_length=100)
]
schema = CollectionSchema(fields, description="Search iterator test collection")
collection = Collection(name=collection_name, schema=schema)

# 3. 准备并插入 150 条模拟数据
# 这样在您设置 batch_size=50 时，迭代器会自动运行 3 次分页拉取，方便您观察
ids = [i for i in range(150)]
# 随机生成 5 维特征向量
vectors = [[random.uniform(-1.0, 1.0) for _ in range(5)] for _ in range(150)]
# 随机赋予一些颜色属性
colors = [random.choice(["red", "blue", "green", "yellow", "orange"]) for _ in range(150)]

data = [ids, vectors, colors]
collection.insert(data)

# 4. 创建 IVF_FLAT 索引 (因为 nprobe=16 强依赖 IVF 系列索引，我们在此设置 nlist=128)
index_params = {
    "metric_type": "L2",
    "index_type": "IVF_FLAT",
    "params": {"nlist": 128}  # 确保聚类数 nlist > nprobe
}
collection.create_index(field_name="vector", index_params=index_params)

# 5. 加载 Collection 到内存中（检索前必须加载）
collection.load()

# 6. 定义搜索迭代器
query_vectors = [
    [0.3580376395471989, -0.6023495712049978, 0.18414012509913835, -0.26286205330961354, 0.9029438446296592]
]

iterator = collection.search_iterator(
    data=query_vectors,
    anns_field="vector",
    param={"metric_type": "L2", "params": {
        "nprobe": 16,
        "radius": 1.0  # 只有 L2 距离小于 1.0 的向量才会被召回，其余的直接过滤(此处使用的是返回搜索)
    }
           },
    batch_size=50,  # 每次调用 next() 预计返回 50 条数据
    output_fields=["color"],
    limit=20000     # 最多返回这么多
)

# 7. 开始循环迭代拉取数据
results = []
page_count = 0

print("\n=== 开始使用迭代器拉取数据 ===")
while True:
    result = iterator.next()
    # 如果没有更多符合条件的数据返回，退出循环并关闭迭代器
    if not result:
        iterator.close()
        break

    page_count += 1
    print(f"第 {page_count} 页数据拉取成功，本批次获取到 {len(result)} 条数据。")

    # 逐条解析 hit 实体并加入结果列表
    for hit in result:
        results.append(hit.to_dict())

# 8. 打印最终统计和样例展示
print(f"\n=== 测试结束 ===")
print(f"共计分页迭代次数: {page_count} 次")
print(f"成功拉取数据总量: {len(results)} 条")
if results:
    print(f"前 3 条检索结果样例: {results[:3]}")