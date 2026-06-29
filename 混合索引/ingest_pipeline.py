import sys
import logging
import json
import os
from pymilvus import MilvusClient, DataType, Function, FunctionType

# ================= 1. 从系统环境变量（Environment Variables）读取配置参数 =================
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
RAW_DOCUMENTS_JSON_PATH = os.getenv("RAW_DOCUMENTS_JSON_PATH", "../raw_documents.json")

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("HybridIngestPipeline")

# 阿里云 API 密钥与向量配置
API_KEY = os.getenv("EMBEDDING_API_KEY", "sk-a42163d874e74c41923259805c86a453")
BASE_URL = os.getenv("EMBEDDING_BASE_URL",
                     "https://ws-oi8z1umy0fuyv6if.cn-beijing.maas.aliyuncs.com/compatible-mode/v1")
MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "text-embedding-v4")
VECTOR_DIM = int(os.getenv("EMBEDDING_VECTOR_DIM", "2048"))

# Milvus 数据库连接配置
MILVUS_URI = os.getenv("MILVUS_URI", "http://localhost:19530")
MILVUS_TOKEN = os.getenv("MILVUS_TOKEN", "root:Milvus")
COLLECTION_NAME = os.getenv("MILVUS_COLLECTION_NAME", "wiki_documents_hybrid_flat")

# 文本长度超参数
PARENT_CHUNK_MAX_LENGTH = int(os.getenv("PARENT_CHUNK_MAX_LENGTH", "8192"))
PARENT_CHUNK_SUMMARY_MAX_LENGTH = int(os.getenv("PARENT_CHUNK_SUMMARY_MAX_LENGTH", "2048"))
CHILD_PARAGRAPH_MAX_LENGTH = int(os.getenv("CHILD_PARAGRAPH_MAX_LENGTH", "2048"))
DOCUMENT_TITLE_MAX_LENGTH = int(os.getenv("DOCUMENT_TITLE_MAX_LENGTH", "512"))
GLOBAL_SUMMARY_MAX_LENGTH = int(os.getenv("GLOBAL_SUMMARY_MAX_LENGTH", "16384"))

# 索引构建超参数
INDEX_TYPE = os.getenv("INDEX_TYPE", "HNSW")
INDEX_HNSW_M = int(os.getenv("INDEX_HNSW_M", "16"))
INDEX_HNSW_EF = int(os.getenv("INDEX_HNSW_EF", "500"))

# 句群切分与摘要参数
CHUNK_WINDOW_SIZE = int(os.getenv("CHUNK_WINDOW_SIZE", "3"))
CHUNK_STEP = int(os.getenv("CHUNK_STEP", "1"))
GENERATE_SUMMARY_MAX_LEN = int(os.getenv("GENERATE_SUMMARY_MAX_LEN", "150"))
GLOBAL_SUMMARY_MAX_LEN = int(os.getenv("GLOBAL_SUMMARY_MAX_LEN", "300"))

# ================= 2. 初始化 Qwen 稠密向量客户端 =================
try:
    from langchain_openai import OpenAIEmbeddings

    embeddings = OpenAIEmbeddings(
        api_key=API_KEY,
        base_url=BASE_URL,
        model=MODEL_NAME,
        check_embedding_ctx_length=False,
        dimensions=VECTOR_DIM,
        chunk_size=10
    )
except ImportError:
    logger.error("未找到 langchain_openai 库，请使用 pip install langchain-openai 进行安装。")
    sys.exit(1)


# ================= 3. 辅助处理工具 =================
def generate_summary(text, max_len=GENERATE_SUMMARY_MAX_LEN):
    clean_text = text.replace("\n", " ")
    return f"【摘要】{clean_text[:max_len]}..."


def chunk_by_sentences(paragraphs, window_size=CHUNK_WINDOW_SIZE, step=CHUNK_STEP):
    chunks = []
    num_sentences = len(paragraphs)
    if num_sentences == 0:
        return chunks

    for i in range(0, num_sentences, step):
        window = paragraphs[i: i + window_size]
        chunk_text = "".join([s if s.endswith("。") or s.endswith("！") or s.endswith("？") or s.endswith(
            "‘") or s.endswith("’") else s + "。" for s in window])
        chunks.append(chunk_text)

        if i + window_size >= num_sentences:
            break

    return chunks


# ================= 4. 安全地读取本地 JSON 数据集 =================
if os.path.isabs(RAW_DOCUMENTS_JSON_PATH):
    json_file_path = RAW_DOCUMENTS_JSON_PATH
else:
    json_file_path = os.path.join(os.path.dirname(__file__) if "__file__" in locals() else ".", RAW_DOCUMENTS_JSON_PATH)

try:
    with open(json_file_path, "r", encoding="utf-8") as f:
        raw_documents = json.load(f)
    logger.info(f"成功加载本地数据集 [{json_file_path}]，共读取了 {len(raw_documents)} 个文档节点。")
except FileNotFoundError:
    logger.error(f"错误：在路径 [{json_file_path}] 未找到数据源文件。")
    sys.exit(1)


# ================= 5. 定义扁平 Schema + 中文分析器与内置 BM25 Function =================
def initialize_database():
    logger.info(f"正在连接 Milvus 节点: {MILVUS_URI} ...")
    client = MilvusClient(uri=MILVUS_URI, token=MILVUS_TOKEN)

    if client.has_collection(COLLECTION_NAME):
        logger.warning(f"检测到已存在 Collection: {COLLECTION_NAME}，正在执行物理删除...")
        client.drop_collection(COLLECTION_NAME)

    # 5.1 创建扁平化 Schema 结构
    schema = client.create_schema(auto_id=True)
    schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True, description="子段落主键 ID")
    schema.add_field(field_name="doc_title", datatype=DataType.VARCHAR, max_length=DOCUMENT_TITLE_MAX_LENGTH)
    schema.add_field(field_name="doc_summary", datatype=DataType.VARCHAR, max_length=GLOBAL_SUMMARY_MAX_LENGTH)
    schema.add_field(field_name="parent_idx", datatype=DataType.INT64)
    schema.add_field(field_name="parent_text", datatype=DataType.VARCHAR, max_length=PARENT_CHUNK_MAX_LENGTH)
    schema.add_field(field_name="parent_summary", datatype=DataType.VARCHAR, max_length=PARENT_CHUNK_SUMMARY_MAX_LENGTH)

    # 定义中文分词器参数
    analyzer_params = {
        "type": "chinese",  # 使用内置的中文分词器
    }

    # 子段落原始文本字段，开启自带分词功能并指定中文分析器
    schema.add_field(
        field_name="text",
        datatype=DataType.VARCHAR,
        max_length=CHILD_PARAGRAPH_MAX_LENGTH,
        enable_analyzer=True,
        analyzer_params=analyzer_params
    )
    # 子段落稠密向量 (通过 Qwen 离线生成)
    schema.add_field(field_name="text_dense", datatype=DataType.FLOAT_VECTOR, dim=VECTOR_DIM)
    # 子段落稀疏向量 (通过 BM25 内置 Function 自动生成)
    schema.add_field(field_name="text_sparse", datatype=DataType.SPARSE_FLOAT_VECTOR)

    # 5.2 定义并挂载 BM25 内置计算函数
    bm25_function = Function(
        name="text_bm25_emb",
        input_field_names=["text"],
        output_field_names=["text_sparse"],
        function_type=FunctionType.BM25,
    )
    schema.add_function(bm25_function)

    # 5.3 声明索引配置
    index_params = client.prepare_index_params()

    # 稠密向量索引 (HNSW)
    index_params.add_index(
        field_name="text_dense",
        index_name="text_dense_index",
        index_type=INDEX_TYPE,
        metric_type="COSINE"
    )

    # 稀疏向量索引 (SPARSE_INVERTED_INDEX)
    index_params.add_index(
        field_name="text_sparse",
        index_name="text_sparse_index",
        index_type="SPARSE_INVERTED_INDEX",
        metric_type="BM25",
        params={"inverted_index_algo": "DAAT_MAXSCORE"}
    )

    # 针对标题与文档大小的标量过滤倒排索引
    index_params.add_index(
        field_name="doc_title",
        index_name="doc_title_index",
        index_type="INVERTED"
    )

    logger.info("正在创建 Collection 并自动构建双路混合索引结构...")
    client.create_collection(
        collection_name=COLLECTION_NAME,
        schema=schema,
        index_params=index_params
    )
    logger.info("扁平混合索引集合构建成功。")
    return client


# ================= 6. 数据导入流水线 =================
def run_ingestion_pipeline(client):
    logger.info("启动扁平化聚合映射与真实向量化导入流水线...")
    merged_documents = {}

    for doc in raw_documents:
        base_title = doc["title"].split(" - ")[0]
        if base_title not in merged_documents:
            merged_documents[base_title] = {
                "title": base_title,
                "chapters": [],
                "sentences": []
            }

        chapter_text = "\n".join(doc["paragraphs"])
        char_count = len(chapter_text)
        summary = generate_summary(chapter_text)

        merged_documents[base_title]["chapters"].append({
            "text": chapter_text,
            "char_count": char_count,
            "summary": summary
        })

        current_chapter_idx = len(merged_documents[base_title]["chapters"]) - 1
        grouped_child_chunks = chunk_by_sentences(doc["paragraphs"], window_size=CHUNK_WINDOW_SIZE, step=CHUNK_STEP)

        for chunk_text in grouped_child_chunks:
            merged_documents[base_title]["sentences"].append({
                "text": chunk_text,
                "parent_idx": current_chapter_idx
            })

    flat_data = []
    for base_title, info in merged_documents.items():
        logger.info(f"正在对《{base_title}》的子句调用 Qwen 模型生成稠密向量...")
        sentence_texts = [item["text"] for item in info["sentences"]]

        chunk_embeddings = embeddings.embed_documents(sentence_texts)

        full_article_text_list = [item["text"] for item in info["sentences"]]
        full_article_text = "\n".join(full_article_text_list)
        global_summary = generate_summary(full_article_text, max_len=GLOBAL_SUMMARY_MAX_LEN)

        for item, emb in zip(info["sentences"], chunk_embeddings):
            p_idx = item["parent_idx"]
            parent_chapter = info["chapters"][p_idx]

            flat_data.append({
                "doc_title": base_title,
                "doc_summary": global_summary,
                "parent_idx": p_idx,
                "parent_text": parent_chapter["text"],
                "parent_summary": parent_chapter["summary"],
                "text": item["text"],
                "text_dense": emb
            })

    logger.info(f"正在向 Milvus 批量导入数据 (共包含 {len(flat_data)} 条扁平数据行)...")
    client.insert(
        collection_name=COLLECTION_NAME,
        data=flat_data
    )
    logger.info("数据落盘及索引更新完成。")


if __name__ == "__main__":
    try:
        db_client = initialize_database()
        run_ingestion_pipeline(db_client)
    except Exception as e:
        logger.error(f"导入程序执行异常: {e}", exc_info=True)
        sys.exit(1)