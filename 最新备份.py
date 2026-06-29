import sys
import logging
import json
import os
import numpy as np
from pymilvus import MilvusClient, DataType
from pymilvus.client.embedding_list import EmbeddingList
from langchain_openai import OpenAIEmbeddings

# ================= 1. 从系统环境变量（Environment Variables）读取所有配置参数 =================

# 1.1 系统日志与文件路径配置
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
RAW_DOCUMENTS_JSON_PATH = os.getenv("RAW_DOCUMENTS_JSON_PATH", "raw_documents.json")

# 配置日志输出（可通过修改环境变量 LOG_LEVEL 控制打印详细度）
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("UnifiedRAGTest")

# 1.2 Qwen 向量服务接口配置
# API_KEY = os.getenv("EMBEDDING_API_KEY", "sk-a42163d874e74c41923259805c86a453")
# BASE_URL = os.getenv("EMBEDDING_BASE_URL", "http://localhost:8001/v1")
# MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "qwen/Qwen3-Embedding-4B")
# VECTOR_DIM = int(os.getenv("EMBEDDING_VECTOR_DIM", "2560"))

# 阿里云 API 密钥 (API Key)
API_KEY = os.getenv("EMBEDDING_API_KEY", "sk-a42163d874e74c41923259805c86a453")
BASE_URL = os.getenv("EMBEDDING_BASE_URL", "https://ws-oi8z1umy0fuyv6if.cn-beijing.maas.aliyuncs.com/compatible-mode/v1")
MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "text-embedding-v4")
VECTOR_DIM = int(os.getenv("EMBEDDING_VECTOR_DIM", "2048"))


# 1.3 检索机制控制参数
TOP_K_CHUNKS_PER_DOC = int(os.getenv("TOP_K_CHUNKS_PER_DOC", "2"))
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.55"))
SEARCH_LIMIT = int(os.getenv("SEARCH_LIMIT", "10"))

# 1.4 Milvus 数据库连接配置
# MILVUS_URI = os.getenv("MILVUS_URI", "http://172.29.97.126:32297")
MILVUS_URI = os.getenv("MILVUS_URI", "http://localhost:19530")
MILVUS_TOKEN = os.getenv("MILVUS_TOKEN", "root:Milvus")
COLLECTION_NAME = os.getenv("MILVUS_COLLECTION_NAME", "wiki_documents_unified_parent_child")

# 1.5 Schema 长度与容量极限超参数
PARENT_CHUNK_MAX_LENGTH = int(os.getenv("PARENT_CHUNK_MAX_LENGTH", "8192"))
PARENT_CHUNK_SUMMARY_MAX_LENGTH = int(os.getenv("PARENT_CHUNK_SUMMARY_MAX_LENGTH", "2048"))
# 配置子chunk的最大字符长度
CHILD_PARAGRAPH_MAX_LENGTH = int(os.getenv("CHILD_PARAGRAPH_MAX_LENGTH", "2048"))
DOCUMENT_TITLE_MAX_LENGTH = int(os.getenv("DOCUMENT_TITLE_MAX_LENGTH", "512"))
GLOBAL_SUMMARY_MAX_LENGTH = int(os.getenv("GLOBAL_SUMMARY_MAX_LENGTH", "16384"))
# 如果超出容量限制，应该将该文章自动切分为“卷1”、“卷2”并作为两个实体（Row）插入。
# 一个文档最大包含多少个段落
PARENT_CHUNKS_MAX_CAPACITY = int(os.getenv("PARENT_CHUNKS_MAX_CAPACITY", "100"))
# 一个文档最大包含多少chunk
PARAGRAPHS_MAX_CAPACITY = int(os.getenv("PARAGRAPHS_MAX_CAPACITY", "500"))

# 1.6 数据库索引构建超参数
INDEX_TYPE = os.getenv("INDEX_TYPE", "HNSW")
METRIC_TYPE = os.getenv("METRIC_TYPE", "MAX_SIM_COSINE")
INDEX_HNSW_M = int(os.getenv("INDEX_HNSW_M", "16"))
INDEX_HNSW_EF = int(os.getenv("INDEX_HNSW_EF", "500"))

# 1.7 句群切分滑动窗口参数
CHUNK_WINDOW_SIZE = int(os.getenv("CHUNK_WINDOW_SIZE", "3"))
CHUNK_STEP = int(os.getenv("CHUNK_STEP", "1"))

# 1.8 摘要提取超参数
GENERATE_SUMMARY_MAX_LEN = int(os.getenv("GENERATE_SUMMARY_MAX_LEN", "150"))
GLOBAL_SUMMARY_MAX_LEN = int(os.getenv("GLOBAL_SUMMARY_MAX_LEN", "300"))

# 1.9 检索测试意图配置 (允许通过 JSON 字符串进行环境覆盖)
DEFAULT_QUERY_CHUNKS = '["Instruct: 查询相关概念\\nQuery: 碳中和目标的出路", "Instruct: 查询相关概念\\nQuery: 什么可以阻挡太阳风"]'
QUERY_CHUNKS_JSON = os.getenv("QUERY_CHUNKS", DEFAULT_QUERY_CHUNKS)
try:
    query_chunks = json.loads(QUERY_CHUNKS_JSON)
except Exception as e:
    logger.error(f"解析 QUERY_CHUNKS 环境变量发生错误，已重置为默认值。错误信息: {e}")
    query_chunks = json.loads(DEFAULT_QUERY_CHUNKS)

# ================= 2. 初始化 Qwen 向量模型客户端 =================
logger.info(f"正在初始化本地 Embeddings 客户端: {MODEL_NAME} ...")
embeddings = OpenAIEmbeddings(
    api_key=API_KEY,
    base_url=BASE_URL,
    model=MODEL_NAME,
    check_embedding_ctx_length=False,  # 关闭本地 tiktoken 校验，避免非 OpenAI 模型冲突
    dimensions=VECTOR_DIM , # 强制约束输出维度与 Milvus 一致
    # text-embedding-v4 模型单次 API 请求支持的文本列表长度（即 Batch Size）最大不能超过 10
    chunk_size=10
)


# ================= 3. 辅助文本处理工具 =================
def generate_summary(text, max_len=GENERATE_SUMMARY_MAX_LEN):
    """文本截断，用于生成段落及整篇摘要"""
    clean_text = text.replace("\n", " ")
    return f"【摘要】{clean_text[:max_len]}..."


def get_cosine_similarity(v1, v2):
    """计算余弦相似度用于客户端命中项过滤"""
    v1, v2 = np.array(v1), np.array(v2)
    dot_product = np.dot(v1, v2)
    norm_v1, norm_v2 = np.linalg.norm(v1), np.linalg.norm(v2)
    return float(dot_product / (norm_v1 * norm_v2)) if norm_v1 > 0 and norm_v2 > 0 else 0.0


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

    # 5.1 清理历史 Collection
    if client.has_collection(COLLECTION_NAME):
        logger.warning(f"检测到已存在 Collection: {COLLECTION_NAME}，正在执行物理删除...")
        client.drop_collection(COLLECTION_NAME)

    # 5.2 定义大片段（章节）结构体 Schema
    parent_struct_schema = client.create_struct_field_schema()
    parent_struct_schema.add_field('text', DataType.VARCHAR, max_length=PARENT_CHUNK_MAX_LENGTH)
    parent_struct_schema.add_field('char_count', DataType.INT64)
    parent_struct_schema.add_field('summary', DataType.VARCHAR, max_length=PARENT_CHUNK_SUMMARY_MAX_LENGTH)

    # 5.3 定义子句结构体 Schema
    struct_schema = client.create_struct_field_schema()
    struct_schema.add_field('text', DataType.VARCHAR, max_length=CHILD_PARAGRAPH_MAX_LENGTH)
    struct_schema.add_field('emb', DataType.FLOAT_VECTOR, dim=VECTOR_DIM)
    struct_schema.add_field('parent_idx', DataType.INT64)

    # 5.4 定义主文章 Schema
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

    # 5.5 核心步骤：构建全方位的索引配置 (含向量 Late Interaction 与多标量过滤索引)
    index_params = client.prepare_index_params()

    # (1) 子向量索引：配置 MAX_SIM 后期交互计算
    index_params.add_index(
        field_name="paragraphs[emb]",
        index_type=INDEX_TYPE,
        metric_type=METRIC_TYPE,
        params={"M": INDEX_HNSW_M, "efConstruction": INDEX_HNSW_EF}
    )

    # (2) 标量索引 A：为主标题创建倒排索引，加速过滤
    index_params.add_index(
        field_name="title",
        index_type="INVERTED"
    )

    # (3) 标量索引 B：为文章字数创建排序索引，加速过滤与统计
    index_params.add_index(
        field_name="total_char_count",
        index_type="STL_SORT"
    )

    # 5.6 创建集合并完成全字段索引构建
    logger.info("开始创建 Collection 并自动构建全字段索引结构...")
    client.create_collection(
        collection_name=COLLECTION_NAME,
        schema=schema,
        index_params=index_params
    )
    logger.info("Collection 与多维索引构建成功。")
    return client


def chunk_by_sentences(paragraphs, window_size=CHUNK_WINDOW_SIZE, step=CHUNK_STEP):
    """
    将单句数组按照滑动/固定窗口合成为包含完整语义的句群（Child Chunks）
    """
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


# ================= 6. 数据级联聚合与真实向量导入流水线 =================
def run_ingestion_pipeline(client):
    logger.info("启动数据级联聚合与真实向量化流水线...")
    merged_documents = {}

    for doc in raw_documents:
        base_title = doc["title"].split(" - ")[0]
        if base_title not in merged_documents:
            merged_documents[base_title] = {
                "title": base_title,
                "chapters": [],  # 存放子篇全文及相关指标
                "sentences": []  # 存放子篇的所有单句
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

        # 按照配置的窗口和移动步长进行句群划分
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
            "parent_chunks": info["chapters"],  # 父段落结构体数组
            "paragraphs": paragraphs_list  # 子文本及向量结构体数组
        })

    logger.info(f"正在向 Milvus 批量导入数据 (包含 {len(formatted_data)} 篇聚合大文章)...")
    client.insert(
        collection_name=COLLECTION_NAME,
        data=formatted_data
    )
    logger.info("数据落盘及索引更新完成。")


# ================= 7. 小到大（Parent-Child）检索与映射还原 =================
def run_retrieval_test(client):
    logger.info("执行 Late Interaction 联合检索测试...")

    # 调用 Qwen 获取查询向量并装入 EmbeddingList
    query_vectors = embeddings.embed_documents(query_chunks)
    query_emb_list = EmbeddingList()
    for vec in query_vectors:
        query_emb_list.add(vec)

    # 检索大文章（由 Milvus 召回最相关的候选大文档）
    results = client.search(
        collection_name=COLLECTION_NAME,
        data=[query_emb_list],
        anns_field="paragraphs[emb]",
        search_params={"metric_type": METRIC_TYPE},
        limit=SEARCH_LIMIT,
        output_fields=["title", "total_char_count", "summary", "parent_chunks", "paragraphs"]
    )
    logger.info("\n" + "=" * 40 + " [检索结果与父子上下文还原] " + "=" * 40)

    # 【优化点 4：归一化查询向量矩阵 (Q, D)，用于后续批量相乘】
    q_matrix = np.array(query_vectors)  # 形状: (Q, D)
    q_norms = np.linalg.norm(q_matrix, axis=1, keepdims=True)
    q_norms[q_norms == 0] = 1.0
    q_matrix_norm = q_matrix / q_norms  # 归一化后的矩阵，形状: (Q, D)

    for rank, hit in enumerate(results[0]):
        doc_title = hit['entity']['title']
        doc_score = hit['distance']
        total_chars = hit['entity']['total_char_count']
        global_summary = hit['entity']['summary']
        parent_chunks = hit['entity']['parent_chunks']
        doc_paragraphs = hit['entity']['paragraphs']

        # 7.1 【优化点 4 实现】：矩阵化批量计算子片段在整个查询多向量下的最大贡献得分并排序
        scored_paragraphs = []
        if doc_paragraphs:
            # 批量提取当前文档内所有子段落的嵌入向量并组装成 NumPy 二维矩阵 (P, D)
            p_embs = np.array([p['emb'] for p in doc_paragraphs])  # 形状: (P, D)
            p_norms = np.linalg.norm(p_embs, axis=1, keepdims=True)
            p_norms[p_norms == 0] = 1.0
            p_embs_norm = p_embs / p_norms  # 归一化子句矩阵，形状: (P, D)

            # 通过矩阵点积计算相似度矩阵: (P, D) x (D, Q) -> (P, Q) 得到所有子句与全部 Query 的相似度
            sim_matrix = np.dot(p_embs_norm, q_matrix_norm.T)

            # 获取每一个子句对应所有查询向量中的最大相似度 (P,)
            max_sims = np.max(sim_matrix, axis=1)

            # 筛选超过或等于阈值的子句
            for idx, max_sim in enumerate(max_sims):
                if max_sim >= SIMILARITY_THRESHOLD:
                    scored_paragraphs.append((idx, float(max_sim)))

        # 按照最大相似度得分，由高到低对子句进行降序排列
        scored_paragraphs.sort(key=lambda x: x[1], reverse=True)

        # ================== 【新增：按父块索引 parent_idx 去重】 ==================
        # 遍历排好序的子片段，如果发现其父级 ID 已被匹配过，则直接跳过（只保留分数最高的那一个）
        seen_parent_indices = set()
        deduplicated_scored_paragraphs = []
        for idx, score in scored_paragraphs:
            p_idx = doc_paragraphs[idx]['parent_idx']
            if p_idx not in seen_parent_indices:
                seen_parent_indices.add(p_idx)  # 记录该父块已占位
                deduplicated_scored_paragraphs.append((idx, score))  # 保留该子句

        # 按照变量 TOP_K_CHUNKS_PER_DOC 截取指定数量的【不同父块】的强关联片段
        selected_paragraphs = deduplicated_scored_paragraphs[:TOP_K_CHUNKS_PER_DOC]

        # 7.2 提取这些强关联片段并映射回父块指针
        parent_chunk_indices = set()
        winning_sentences = []

        for idx, score in selected_paragraphs:
            p_struct = doc_paragraphs[idx]
            winning_sentences.append((p_struct['text'], score))
            parent_chunk_indices.add(p_struct['parent_idx'])

        # 打印匹配结果
        print(f"\n【Rank {rank + 1}】聚合大主题：《{doc_title}》")
        print(f" ├─ 全文规模: 共 {total_chars} 字符")
        print(f" ├─ 宏观大纲摘要: {global_summary}")
        print(f" ├─ Late Interaction 联合得分 (MAX_SIM_COSINE): {doc_score:.4f}")
        print(f" ├─ 🎯 实际命中的微观单句 (Child chunks - 最多保留前 {TOP_K_CHUNKS_PER_DOC} 个):")

        if winning_sentences:
            for sent_text, score in winning_sentences:
                print(f" │   * [相似度: {score:.4f}] {sent_text}")
        else:
            print(" │   * (无任何单句通过余弦相似度阈值)")

        print(" └─ 📄 重构并送入大模型的完整章节上下文 (Parent Chapters):")
        if parent_chunk_indices:
            for p_idx in sorted(list(parent_chunk_indices)):
                parent_node = parent_chunks[p_idx]
                print(f"     --- [章节 {p_idx + 1} (全长 {parent_node['char_count']} 字符)] ---")
                print(f"     - 章节摘要: {parent_node['summary']}")
                print(f"     - 章节真实全文:")
                print(f"       \"{parent_node['text'].replace('\n', '\n       ')}\"")
        else:
            print("     (未命中任何有效的父段落上下文)")


# ================= 8. 主控入口 =================
if __name__ == "__main__":
    try:
        # Step 1: 初始化数据库，建立多维索引
        db_client = initialize_database()

        # Step 2: 启动级联聚合与真实向量化导入管道
        run_ingestion_pipeline(db_client)

        # Step 3: 执行基于 Qwen 向量的 Late Interaction 父块解析还原
        run_retrieval_test(db_client)

    except Exception as e:
        logger.error(f"测试程序在执行中发生致命异常: {e}", exc_info=True)
        sys.exit(1)