import sys
import logging
import json
import os
import numpy as np
from pymilvus import MilvusClient
from pymilvus.client.embedding_list import EmbeddingList
from langchain_openai import OpenAIEmbeddings

# ================= 1. 从系统环境变量（Environment Variables）读取配置参数 =================

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

# 配置日志输出
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("QueryPipeline")

# Qwen 向量服务接口配置
API_KEY = os.getenv("EMBEDDING_API_KEY", "sk-a42163d874e74c41923259805c86a453")
BASE_URL = os.getenv("EMBEDDING_BASE_URL", "https://ws-oi8z1umy0fuyv6if.cn-beijing.maas.aliyuncs.com/compatible-mode/v1")
MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "text-embedding-v4")
VECTOR_DIM = int(os.getenv("EMBEDDING_VECTOR_DIM", "2048"))

# 检索机制控制参数
TOP_K_CHUNKS_PER_DOC = int(os.getenv("TOP_K_CHUNKS_PER_DOC", "2"))
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.55"))
SEARCH_LIMIT = int(os.getenv("SEARCH_LIMIT", "10"))

# Milvus 数据库连接配置
MILVUS_URI = os.getenv("MILVUS_URI", "http://localhost:19530")
MILVUS_TOKEN = os.getenv("MILVUS_TOKEN", "root:Milvus")
COLLECTION_NAME = os.getenv("MILVUS_COLLECTION_NAME", "wiki_documents_unified_parent_child")

METRIC_TYPE = os.getenv("METRIC_TYPE", "MAX_SIM_COSINE")

# 检索测试意图配置
DEFAULT_QUERY_CHUNKS = '["Instruct: 查询相关概念\\nQuery: 碳中和目标的出路"]'
QUERY_CHUNKS_JSON = os.getenv("QUERY_CHUNKS", DEFAULT_QUERY_CHUNKS)
try:
    query_chunks = json.loads(QUERY_CHUNKS_JSON)
except Exception as e:
    logger.error(f"解析 QUERY_CHUNKS 环境变量发生错误，已重置为默认值。错误信息: {e}")
    query_chunks = json.loads(DEFAULT_QUERY_CHUNKS)


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


# ================= 3. 辅助计算工具 =================
def get_cosine_similarity(v1, v2):
    """计算余弦相似度用于客户端命中项过滤"""
    v1, v2 = np.array(v1), np.array(v2)
    dot_product = np.dot(v1, v2)
    norm_v1, norm_v2 = np.linalg.norm(v1), np.linalg.norm(v2)
    return float(dot_product / (norm_v1 * norm_v2)) if norm_v1 > 0 and norm_v2 > 0 else 0.0


# ================= 4. 小到大（Parent-Child）检索与映射还原 =================
def run_retrieval_test(client):
    logger.info("执行 Late Interaction 联合检索测试...")

    # 调用 Qwen 获取查询向量并装入 EmbeddingList
    query_vectors = embeddings.embed_documents(query_chunks)
    query_emb_list = EmbeddingList()
    for vec in query_vectors:
        query_emb_list.add(vec)

    # 检索大文章
    results = client.search(
        collection_name=COLLECTION_NAME,
        data=[query_emb_list],
        anns_field="paragraphs[emb]",
        search_params={"metric_type": METRIC_TYPE},
        limit=SEARCH_LIMIT,
        output_fields=["title", "total_char_count", "summary", "parent_chunks", "paragraphs"]
    )
    logger.info("\n" + "=" * 40 + " [检索结果与父子上下文还原] " + "=" * 40)

    # 归一化查询向量矩阵 (Q, D)，用于后续批量相乘
    q_matrix = np.array(query_vectors)
    q_norms = np.linalg.norm(q_matrix, axis=1, keepdims=True)
    q_norms[q_norms == 0] = 1.0
    q_matrix_norm = q_matrix / q_norms

    for rank, hit in enumerate(results[0]):
        doc_title = hit['entity']['title']
        doc_score = hit['distance']
        total_chars = hit['entity']['total_char_count']
        global_summary = hit['entity']['summary']
        parent_chunks = hit['entity']['parent_chunks']
        doc_paragraphs = hit['entity']['paragraphs']

        # 矩阵化批量计算子片段在整个查询多向量下的最大贡献得分并排序
        scored_paragraphs = []
        if doc_paragraphs:
            p_embs = np.array([p['emb'] for p in doc_paragraphs])
            p_norms = np.linalg.norm(p_embs, axis=1, keepdims=True)
            p_norms[p_norms == 0] = 1.0
            p_embs_norm = p_embs / p_norms

            # 通过矩阵点积计算相似度矩阵: (P, D) x (D, Q) -> (P, Q)
            sim_matrix = np.dot(p_embs_norm, q_matrix_norm.T)

            # 获取每一个子句对应所有查询向量中的最大相似度 (P,)
            max_sims = np.max(sim_matrix, axis=1)

            for idx, max_sim in enumerate(max_sims):
                if max_sim >= SIMILARITY_THRESHOLD:
                    scored_paragraphs.append((idx, float(max_sim)))

        # 降序排列
        scored_paragraphs.sort(key=lambda x: x[1], reverse=True)

        # 按父块索引 parent_idx 去重
        seen_parent_indices = set()
        deduplicated_scored_paragraphs = []
        for idx, score in scored_paragraphs:
            p_idx = doc_paragraphs[idx]['parent_idx']
            if p_idx not in seen_parent_indices:
                seen_parent_indices.add(p_idx)
                deduplicated_scored_paragraphs.append((idx, score))

        # 按照 TOP_K_CHUNKS_PER_DOC 截取
        selected_paragraphs = deduplicated_scored_paragraphs[:TOP_K_CHUNKS_PER_DOC]

        # 提取强关联片段并映射回父块指针
        parent_chunk_indices = set()
        winning_sentences = []

        for idx, score in selected_paragraphs:
            p_struct = doc_paragraphs[idx]
            winning_sentences.append((p_struct['text'], score))
            parent_chunk_indices.add(p_struct['parent_idx'])

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


# ================= 5. 主控入口 =================
if __name__ == "__main__":
    try:
        logger.info(f"正在建立 Milvus 连接: {MILVUS_URI} ...")
        db_client = MilvusClient(uri=MILVUS_URI, token=MILVUS_TOKEN)

        if not db_client.has_collection(COLLECTION_NAME):
            logger.error(f"指定的集合 [{COLLECTION_NAME}] 在数据库中不存在，请先运行 ingest_pipeline.py 导入数据。")
            sys.exit(1)

        run_retrieval_test(db_client)
    except Exception as e:
        logger.error(f"查询程序在执行中发生致命异常: {e}", exc_info=True)
        sys.exit(1)