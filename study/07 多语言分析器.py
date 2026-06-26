


from pymilvus import MilvusClient, DataType, Function, FunctionType

client = MilvusClient(
    uri="http://localhost:19530",
)

schema = client.create_schema()

schema.add_field(
    field_name="id",                  # Field name
    datatype=DataType.INT64,          # Integer data type
    is_primary=True,                  # Designate as primary key
    auto_id=True                      # Auto-generate IDs (recommended)
)

schema.add_field(
    field_name="language",       # Field name
    datatype=DataType.VARCHAR,   # String data type
    max_length=255               # Maximum length (adjust as needed)
)
multi_analyzer_params = {
  # Define language-specific analyzers
  # Each analyzer follows this format: <analyzer_name>: <analyzer_params>
  "analyzers": {
    "english": {"type": "english"},          # English-optimized analyzer
    "chinese": {"type": "chinese"},          # Chinese-optimized analyzer
    "default": {"tokenizer": "icu"}          # Required fallback analyzer
  },
  "by_field": "language",                    # Field determining analyzer selection
  "alias": {
    "cn": "chinese",                         # Use "cn" as shorthand for Chinese
    "en": "english"                          # Use "en" as shorthand for English
  }
}
schema.add_field(
    field_name="text",                           # Field name
    datatype=DataType.VARCHAR,                   # String data type
    max_length=8192,                             # Maximum length (adjust based on expected text size)
    enable_analyzer=True,                        # Enable text analysis
    multi_analyzer_params=multi_analyzer_params  # Connect with our language analyzers
)

schema.add_field(
    field_name="sparse",                   # Field name
    datatype=DataType.SPARSE_FLOAT_VECTOR  # Sparse vector data type
)
# 该函数会根据每个文本条目的语言标识符自动应用相应的分析器。
bm25_function = Function(
    name="text_to_vector",            # Descriptive function name
    function_type=FunctionType.BM25,  # Use BM25 algorithm
    input_field_names=["text"],       # Process text from this field
    output_field_names=["sparse"]     # Store vectors in this field
)

schema.add_function(bm25_function)

index_params = client.prepare_index_params()

index_params.add_index(
    field_name="sparse",        # Field to index (our vector field)
    index_type="AUTOINDEX",     # Let Milvus choose optimal index type
    metric_type="BM25"          # Must be BM25 for this feature
)

COLLECTION_NAME = "multilingual_documents"

if client.has_collection(COLLECTION_NAME):
    client.drop_collection(COLLECTION_NAME)  # Remove it for this example
    print(f"Dropped existing collection: {COLLECTION_NAME}")

client.create_collection(
    collection_name=COLLECTION_NAME,       # Collection name
    schema=schema,                         # Our multilingual schema
    index_params=index_params              # Our search index configuration
)

documents = [
    # English documents
    {
        "text": "Artificial intelligence is transforming technology",
        "language": "english",  # Using full language name
    },
    {
        "text": "Machine learning models require large datasets",
        "language": "en",  # Using our defined alias
    },
    # Chinese documents
    {
        "text": "人工智能正在改变技术领域",
        "language": "chinese",  # Using full language name
    },
    {
        "text": "机器学习模型需要大型数据集",
        "language": "cn",  # Using our defined alias
    },
]

result = client.insert(COLLECTION_NAME, documents)

inserted = result["insert_count"]
print(f"Successfully inserted {inserted} documents")
print("Documents by language: 2 English, 2 Chinese")



search_params = {
    "metric_type": "BM25",            # Must match index configuration
    "analyzer_name": "english",  # Analyzer that matches the query language
    "drop_ratio_search": "0",     # Keep all terms in search (tweak as needed)
}

english_results = client.search(
    collection_name=COLLECTION_NAME,  # Collection to search
    data=["artificial intelligence"],                # Query text
    anns_field="sparse",              # Field to search against
    search_params=search_params,      # Search configuration
    limit=3,                      # Max results to return
    output_fields=["text", "language"],  # Fields to include in the output
    consistency_level="Bounded",       # Data‑consistency guarantee
)

print("\n=== English Search Results ===")
for i, hit in enumerate(english_results[0]):
    print(f"{i+1}. [{hit.score:.4f}] {hit.entity.get('text')} "
          f"(Language: {hit.entity.get('language')})")


search_params["analyzer_name"] = "cn"

chinese_results = client.search(
    collection_name=COLLECTION_NAME,  # Collection to search
    data=["人工智能"],                # Query text
    anns_field="sparse",              # Field to search against
    search_params=search_params,      # Search configuration
    limit=3,                      # Max results to return
    output_fields=["text", "language"],  # Fields to include in the output
    consistency_level="Bounded",       # Data‑consistency guarantee
)

print("\n=== Chinese Search Results ===")
for i, hit in enumerate(chinese_results[0]):
    print(f"{i+1}. [{hit.score:.4f}] {hit.entity.get('text')} "
          f"(Language: {hit.entity.get('language')})")

