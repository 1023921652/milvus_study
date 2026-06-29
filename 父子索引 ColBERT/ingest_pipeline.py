import sys
import logging
import json
import os
from pymilvus import MilvusClient, DataType
from langchain_openai import OpenAIEmbeddings

# ================= 1. 从系统环境变量（Environment Variables）读取配置参数 =================

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
RAW_DOCUMENTS_JSON_PATH = os.getenv("RAW_DOCUMENTS_JSON_PATH", "../raw_documents.json")

# 配置日志输出
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("IngestPipeline")

# Qwen 向量服务接口配置
API_KEY = os.getenv("EMBEDDING_API_KEY", "sk-a42163d874e74c41923259805c86a453")
BASE_URL = os.getenv("EMBEDDING_BASE_URL", "https://ws-oi8z1umy0fuyv6if.cn-beijing.maas.aliyuncs.com/compatible-mode/v1")
MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "text-embedding-v4")
VECTOR_DIM = int(os.getenv("EMBEDDING_VECTOR_DIM", "2048"))

# Milvus 数据库连接配置
MILVUS_URI = os.getenv("MILVUS_URI", "http://localhost:19530")
MILVUS_TOKEN = os.getenv("MILVUS_TOKEN", "root:Milvus")
COLLECTION_NAME = os.getenv("MILVUS_COLLECTION_NAME", "wiki_documents_unified_parent_child")

# Schema 长度与容量极限超参数
PARENT_CHUNK_MAX_LENGTH = int(os.getenv("PARENT_CHUNK_MAX_LENGTH", "8192"))
PARENT_CHUNK_SUMMARY_MAX_LENGTH = int(os.getenv("PARENT_CHUNK_SUMMARY_MAX_LENGTH", "2048"))
CHILD_PARAGRAPH_MAX_LENGTH = int(os.getenv("CHILD_PARAGRAPH_MAX_LENGTH", "2048"))
DOCUMENT_TITLE_MAX_LENGTH = int(os.getenv("DOCUMENT_TITLE_MAX_LENGTH", "512"))
GLOBAL_SUMMARY_MAX_LENGTH = int(os.getenv("GLOBAL_SUMMARY_MAX_LENGTH", "16384"))
PARENT_CHUNKS_MAX_CAPACITY = int(os.getenv("PARENT_CHUNKS_MAX_CAPACITY", "100"))
PARAGRAPHS_MAX_CAPACITY = int(os.getenv("PARAGRAPHS_MAX_CAPACITY", "500"))

# 数据库索引构建超参数
INDEX_TYPE = os.getenv("INDEX_TYPE", "HNSW")
METRIC_TYPE = os.getenv("METRIC_TYPE", "MAX_SIM_COSINE")
INDEX_HNSW_M = int(os.getenv("INDEX_HNSW_M", "16"))
INDEX_HNSW_EF = int(os.getenv("INDEX_HNSW_EF", "500"))

# 句群切分滑动窗口参数
CHUNK_WINDOW_SIZE = int(os.getenv("CHUNK_WINDOW_SIZE", "3"))
CHUNK_STEP = int(os.getenv("CHUNK_STEP", "1"))

# 摘要提取超参数
GENERATE_SUMMARY_MAX_LEN = int(os.getenv("GENERATE_SUMMARY_MAX_LEN", "150"))
GLOBAL_SUMMARY_MAX_LEN = int(os.getenv("GLOBAL_SUMMARY_MAX_LEN", "300"))


# ================= 2. 初始化 Qwen 向量模型客户端 =================
logger.info(f"正在初始化 Embeddings 客户端: {MODEL_NAME} ...")
embeddings = OpenAIEmbeddings(
    api_key=API_KEY,
    base_url=BASE_URL,
    model=MODEL_NAME,
    check_embedding_ctx_length=False,
    dimensions=VECTOR_DIM,
    chunk_size=10
)


# ================= 3. 辅助文本处理工具 =================
def generate_summary(text, max_len=GENERATE_SUMMARY_MAX_LEN):
    """文本截断，用于生成段落及整篇摘要"""
    clean_text = text.replace("\n", " ")
    return f"【摘要】{clean_text[:max_len]}..."


def chunk_by_sentences(paragraphs, window_size=CHUNK_WINDOW_SIZE, step=CHUNK_STEP):
    """将单句数组按照滑动/固定窗口合成为包含完整语义的句群（Child Chunks）"""
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
    logger.error(f"错误：在路径 [{json_file_path}] 未找到数据源文件。请确保数据集已存储在该目录下。")
    sys.exit(1)


# ================= 5. 数据库初始化及 Schema/多维索引定义 =================
def initialize_database():
    logger.info(f"正在连接 Milvus 节点: {MILVUS_URI} ...")
    client = MilvusClient(uri=MILVUS_URI, token=MILVUS_TOKEN)

    # 清理历史 Collection
    if client.has_collection(COLLECTION_NAME):
        logger.warning(f"检测到已存在 Collection: {COLLECTION_NAME}，正在执行物理删除...")
        client.drop_collection(COLLECTION_NAME)

    # 定义大片段（章节）结构体 Schema
    parent_struct_schema = client.create_struct_field_schema()
    parent_struct_schema.add_field('text', DataType.VARCHAR, max_length=PARENT_CHUNK_MAX_LENGTH)
    parent_struct_schema.add_field('char_count', DataType.INT64)
    parent_struct_schema.add_field('summary', DataType.VARCHAR, max_length=PARENT_CHUNK_SUMMARY_MAX_LENGTH)

    # 定义子句结构体 Schema
    struct_schema = client.create_struct_field_schema()
    struct_schema.add_field('text', DataType.VARCHAR, max_length=CHILD_PARAGRAPH_MAX_LENGTH)
    struct_schema.add_field('emb', DataType.FLOAT_VECTOR, dim=VECTOR_DIM)
    struct_schema.add_field('parent_idx', DataType.INT64)

    # 定义主文章 Schema
    schema = client.create_schema()
    schema.add_field('id', DataType.INT64, is_primary=True, auto_id=True)
    schema.add_field('title', DataType.VARCHAR, max_length=DOCUMENT_TITLE_MAX_LENGTH)
    schema.add_field('total_char_count', DataType.INT64)
    schema.add_field('summary', DataType.VARCHAR, max_length=GLOBAL_SUMMARY_MAX_LENGTH)

    # 父段落结构体数组
    schema.add_field('parent_chunks', DataType.ARRAY,
                     element_type=DataType.STRUCT,
                     struct_schema=parent_struct_schema,
                     max_capacity=PARENT_CHUNKS_MAX_CAPACITY)

    # 小片段结构体数组（含向量）
    schema.add_field('paragraphs', DataType.ARRAY,
                     element_type=DataType.STRUCT,
                     struct_schema=struct_schema,
                     max_capacity=PARAGRAPHS_MAX_CAPACITY)

    # 构建全方位的索引配置 (含向量 Late Interaction 与多标量过滤索引)
    index_params = client.prepare_index_params()

    # (1) 子向量索引：配置 MAX_SIM 后期交互计算
    index_params.add_index(
        field_name="paragraphs[emb]",
        index_type=INDEX_TYPE,
        metric_type=METRIC_TYPE,
        params={"M": INDEX_HNSW_M, "efConstruction": INDEX_HNSW_EF}
    )

    # (2) 标量索引 A：为主标题创建倒转索引
    index_params.add_index(
        field_name="title",
        index_type="INVERTED"
    )

    # (3) 标量索引 B：为主文章字数创建排序索引
    index_params.add_index(
        field_name="total_char_count",
        index_type="STL_SORT"
    )

    logger.info("开始创建 Collection 并自动构建全字段索引结构...")
    client.create_collection(
        collection_name=COLLECTION_NAME,
        schema=schema,
        index_params=index_params
    )
    logger.info("Collection 与多维索引构建成功。")
    return client


# ================= 6. 数据级联聚合与真实向量导入流水线 =================
def run_ingestion_pipeline(client):
    logger.info("启动数据级联聚合与真实向量化流水线...")
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

    formatted_data = []
    for base_title, info in merged_documents.items():
        logger.info(f"正在对《{base_title}》的子句调用本地 Qwen 模型生成向量...")
        sentence_texts = [item["text"] for item in info["sentences"]]

        chunk_embeddings = embeddings.embed_documents(sentence_texts)

        paragraphs_list = []
        full_article_text_list = []
        for item, emb in zip(info["sentences"], chunk_embeddings):
            paragraphs_list.append({
                "text": item["text"],
                "emb": emb,
                "parent_idx": item["parent_idx"]
            })
            full_article_text_list.append(item["text"])

        full_article_text = "\n".join(full_article_text_list)
        total_char_count = len(full_article_text)
        global_summary = generate_summary(full_article_text, max_len=GLOBAL_SUMMARY_MAX_LEN)

        formatted_data.append({
            "title": base_title,
            "total_char_count": total_char_count,
            "summary": global_summary,
            "parent_chunks": info["chapters"],
            "paragraphs": paragraphs_list
        })

    logger.info(f"正在向 Milvus 批量导入数据 (包含 {len(formatted_data)} 篇聚合大文章)...")
    client.insert(
        collection_name=COLLECTION_NAME,
        data=formatted_data
    )
    logger.info("数据落盘及索引更新完成。")


# ================= 7. 主控入口 =================
if __name__ == "__main__":
    try:
        db_client = initialize_database()
        run_ingestion_pipeline(db_client)
    except Exception as e:
        logger.error(f"导入程序执行中发生致命异常: {e}", exc_info=True)
        sys.exit(1)