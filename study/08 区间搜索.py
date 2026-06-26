from pymilvus import MilvusClient

client = MilvusClient(
    uri="http://localhost:19530",
    token="root:Milvus"
)

query_vector = [0.3580376395471989, -0.6023495712049978, 0.18414012509913835, -0.26286205330961354, 0.9029438446296592]
# 按照度量区间搜索
res = client.search(
    collection_name="my_collection",
    data=[query_vector],
    limit=3,
    search_params={
        # highlight-start
        "params": {
            "radius": 0.4,
            "range_filter": 1
        }
        # highlight-end
    }
)

for hits in res:
    print("TopK results:")
    for hit in hits:
        print(hit)
