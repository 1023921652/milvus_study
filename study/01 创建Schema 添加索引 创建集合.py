
from pymilvus import MilvusClient, DataType
client = MilvusClient(uri="http://172.29.97.126:32297")


schema = client.create_schema(
    auto_id=False,
    enable_dynamic_fields=True,
)

schema.add_field(field_name="varchar_field1", datatype=DataType.VARCHAR, max_length=100, nullable=True, default_value="Unknown")
schema.add_field(field_name="varchar_field2", datatype=DataType.VARCHAR, max_length=200, nullable=True)
schema.add_field(field_name="pk", datatype=DataType.INT64, is_primary=True)
schema.add_field(field_name="embedding", datatype=DataType.FLOAT_VECTOR, dim=3)


index_params = client.prepare_index_params()

index_params.add_index(
    field_name="varchar_field1",
    index_type="AUTOINDEX",
    index_name="varchar_index"
)

index_params.add_index(
    field_name="embedding",
    index_type="AUTOINDEX",  # Use automatic indexing to simplify complex index settings
    metric_type="COSINE"  # Specify similarity metric type, options include L2, COSINE, or IP
)

if client.has_collection(collection_name="my_collection"):
    client.drop_collection(collection_name="my_collection")
client.create_collection(
    collection_name="my_collection",
    schema=schema,
    index_params=index_params
)

data = [
    {"varchar_field1": "Product A", "varchar_field2": "High quality product", "pk": 1, "embedding": [0.1, 0.2, 0.3]},
    {"varchar_field1": "Product B", "pk": 2, "embedding": [0.4, 0.5, 0.6]}, # varchar_field2 field is missing, which should be NULL
    {"varchar_field1": None, "varchar_field2": None, "pk": 3, "embedding": [0.2, 0.3, 0.1]},  # `varchar_field1` should default to `Unknown`, `varchar_field2` is NULL
    {"varchar_field1": "Product C", "varchar_field2": None, "pk": 4, "embedding": [0.5, 0.7, 0.2]},  # `varchar_field2` is NULL
    {"varchar_field1": None, "varchar_field2": "Exclusive deal", "pk": 5, "embedding": [0.6, 0.4, 0.8]},  # `varchar_field1` should default to `Unknown`
    {"varchar_field1": "Unknown", "varchar_field2": None, "pk": 6, "embedding": [0.8, 0.5, 0.3]},  # `varchar_field2` is NULL
    {"varchar_field1": "", "varchar_field2": "Best seller", "pk": 7, "embedding": [0.8, 0.5, 0.3]}, # Empty string is not treated as NULL
]

client.insert(
    collection_name="my_collection",
    data=data
)

filter = 'varchar_field1 == "Product A"'

res = client.query(
    collection_name="my_collection",
    filter=filter,
    output_fields=["varchar_field1", "varchar_field2"],
    consistency_level="Strong"
)

print(res)

filter = 'varchar_field2 is null'

res = client.query(
    collection_name="my_collection",
    filter=filter,
    output_fields=["varchar_field1", "varchar_field2"]
)

print(res)

filter = 'varchar_field1 == "Unknown"'

res = client.query(
    collection_name="my_collection",
    filter=filter,
    output_fields=["varchar_field1", "varchar_field2"]
)

print(res)

# 使用过滤表达式进行向量搜索
filter = 'varchar_field2 == "Best seller"'

res = client.search(
    collection_name="my_collection",
    data=[[0.3, -0.6, 0.1]],
    limit=5,
    search_params={"params": {"nprobe": 10}},
    output_fields=["varchar_field1", "varchar_field2"],
    filter=filter
)

print(res)





