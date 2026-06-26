from pymilvus import MilvusClient, DataType, Function, FunctionType

client = MilvusClient(
    uri="http://localhost:19530",
    token="root:Milvus"
)

schema = client.create_schema()

schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True, auto_id=True) # Primary field
analyzer_params = {
    "type": "english"
}

schema.add_field(field_name="text", datatype=DataType.VARCHAR, max_length=1000, enable_analyzer=True,analyzer_params=analyzer_params) # Text field

schema.add_field(field_name="sparse", datatype=DataType.SPARSE_FLOAT_VECTOR) # Sparse vector field; no dim required for sparse vectors

bm25_function = Function(
    name="text_bm25_emb", # Function name
    input_field_names=["text"], # Name of the VARCHAR field containing raw text data
    output_field_names=["sparse"], # Name of the SPARSE_FLOAT_VECTOR field reserved to store generated embeddings
    # highlight-next-line
    function_type=FunctionType.BM25, # Set to `BM25`
)

schema.add_function(bm25_function)

index_params = client.prepare_index_params()

index_params.add_index(
    field_name="sparse",

    index_type="SPARSE_INVERTED_INDEX",
    metric_type="BM25",
    params={
        "inverted_index_algo": "DAAT_MAXSCORE",
        "bm25_k1": 1.2,
        "bm25_b": 0.75
    }

)

if client.has_collection(collection_name="my_collection"):
    client.drop_collection(collection_name="my_collection")
client.create_collection(
    collection_name='my_collection',
    schema=schema,
    index_params=index_params
)
client.insert('my_collection', [
    {'text': 'information retrieval is a field of study.'},
    {'text': 'information retrieval focuses on finding relevant information in large datasets.'},
    {'text': 'data mining and information retrieval overlap in research.'},
])

from pymilvus import LexicalHighlighter

highlighter = LexicalHighlighter(
    pre_tags=["<b>", "<i>"],              # Tag inserted before each highlighted term
    post_tags=["<b>", "<i>"],             # Tag inserted after each highlighted term
    highlight_search_text=True,  # Enable search term highlighting for BM25 full text search
    fragment_offset=5,     # Number of characters to reserve before the first matched term
    fragment_size=60,      # Max. length of each fragment to return
    num_of_fragments=1     # Max. number of fragments to return（默认为 5）
)

res = client.search(
    collection_name='my_collection',
    # highlight-start
    data=['whats the focus of information retrieval?'],
    anns_field='sparse',
    output_fields=['text'], # Fields to return in search results; sparse field cannot be output
    # highlight-end
    limit=3,
    search_params={"metric_type": "BM25", "params": {"drop_ratio_search": 0.0}},
    highlighter=highlighter,
    consistency_level="Strong"
)

print(res)
