
from pymilvus import MilvusClient, DataType
client = MilvusClient(uri="http://127.0.0.1:19530")


analyzer_params_built_in = {
    "type": "english"
}

analyzer_params_custom = {
    "tokenizer": "standard",
    "filter": [
        "lowercase",  # Built-in filter: convert tokens to lowercase
        {
            "type": "length",  # Custom filter: restrict token length
            "max": 40
        },
        {
            "type": "stop",  # Custom filter: remove specified stop words
            "stop_words": ["of", "for"]
        }
    ]
}


schema = client.create_schema(
    auto_id=False,
    enable_dynamic_fields=True,
)
schema.add_field(
    field_name='title_en',
    datatype=DataType.VARCHAR,
    max_length=1000,
    enable_analyzer=True,
    analyzer_params=analyzer_params_built_in,
    enable_match=True,
)

schema.add_field(
    field_name='title',
    datatype=DataType.VARCHAR,
    max_length=1000,
    enable_analyzer=True,
    analyzer_params=analyzer_params_custom,
    enable_match=True,
)

schema.add_field(field_name="embedding", datatype=DataType.FLOAT_VECTOR, dim=3)

schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True)
index_params = client.prepare_index_params()
index_params.add_index(field_name="embedding", metric_type="COSINE", index_type="AUTOINDEX")

if client.has_collection(collection_name="my_collection"):
    client.drop_collection(collection_name="my_collection")
client.create_collection(
    collection_name="my_collection",
    schema=schema,
    index_params=index_params
)




# 使用中文标记器
analyzer_params = {
    "type": "chinese", # Uses the standard built-in analyzer
    # "stop_words": ["a", "an", "for"] # Defines a list of common words (stop words) to exclude from tokenization
}


text = "对于更高级的文本处理，Milvus 中的自定义分析器允许您通过指定标记符号化器和过滤器来建立一个定制的文本处理管道。这种设置非常适合需要精确控制的特殊用例"

result = client.run_analyzer(
    text,
    analyzer_params
)
print(result)


