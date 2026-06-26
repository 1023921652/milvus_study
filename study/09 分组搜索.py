#  Collections 执行近似近邻 (ANN) 搜索时，搜索结果可能包括同一文档中的多个段落，有可能导致其他文档被忽略，这可能与预期用例不符。
# 为了提高搜索结果的多样性，可以在搜索请求中添加group_by_field 参数来启用分组搜索。如图所示，您可以将group_by_field 设置为docId 。收到此请求后，Milvus 将
#
# 根据提供的查询向量执行 ANN 搜索，找到与查询最相似的所有实体。
#
# 按指定的group_by_field 对搜索结果进行分组，如docId 。
#
# 根据limit 参数的定义，返回每个组的顶部结果，并从每个组中选出最相似的实体。

from pymilvus import MilvusClient

client = MilvusClient(
    uri="http://localhost:19530",
    token="root:Milvus"
)

query_vectors = [
    [0.14529211512077012, 0.9147257273453546, 0.7965055218724449, 0.7009258593102812, 0.5605206522382088]]
# 此时limit=3,表示返回三组，没有一个实体
res = client.search(
    collection_name="my_collection",
    data=query_vectors,
    limit=3,
    group_by_field="docId",
    output_fields=["docId"]
)

doc_ids = [result['entity']['docId'] for result in res[0]]

# 此时limit=5,表示返回5组，没有2个实体
# 默认情况下，分组搜索每个组只返回一个实体。如果希望每组有多个结果，请调整group_size 和strict_group_size 参数。
res = client.search(
    collection_name="my_collection",
    data=query_vectors, # query vector
    limit=5, # number of groups to return
    group_by_field="docId", # grouping field
    group_size=2, # p to 2 entities to return from each group
    strict_group_size=True, # return exact 2 entities from each group
    output_fields=["docId"]
)
