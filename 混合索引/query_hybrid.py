import sys
import logging
import json
import os
from pymilvus import MilvusClient, AnnSearchRequest, RRFRanker

# ================= 1. 读取混合检索配置 =================
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("HybridQueryPipeline")

# Qwen 模型接口配置
API_KEY = os.getenv("EMBEDDING_API_KEY", "sk-a42163d874e74c41923259805c86a453")
BASE_URL = os.getenv("EMBEDDING_BASE_URL",
                     "https://ws-oi8z1umy0fuyv6if.cn-beijing.maas.aliyuncs.com/compatible-mode/v1")
MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "text-embedding-v4")
VECTOR_DIM = int(os.getenv("EMBEDDING_VECTOR_DIM", "2048"))

# 检索机制控制参数
TOP_K_CHUNKS_PER_DOC = int(os.getenv("TOP_K_CHUNKS_PER_DOC", "2"))
SEARCH_LIMIT = int(os.getenv("SEARCH_LIMIT", "15"))

# Milvus 数据库连接配置
MILVUS_URI = os.getenv("MILVUS_URI", "http://localhost:19530")
MILVUS_TOKEN = os.getenv("MILVUS_TOKEN", "root:Milvus")
COLLECTION_NAME = os.getenv("MILVUS_COLLECTION_NAME", "wiki_documents_hybrid_flat")

# 检索测试意图配置
DEFAULT_QUERY_CHUNKS = '["碳中和目标的出路是什么", "什么可以阻挡太阳风袭击地球"]'
QUERY_CHUNKS_JSON = os.getenv("QUERY_CHUNKS", DEFAULT_QUERY_CHUNKS)
try:
    query_chunks = json.loads(QUERY_CHUNKS_JSON)
except Exception as e:
    logger.error(f"解析 QUERY_CHUNKS 发生错误，重置为默认值。错误信息: {e}")
    query_chunks = json.loads(DEFAULT_QUERY_CHUNKS)

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


# ================= 3. 双路混合检索与去重级联映射 =================
def run_hybrid_search_test(client):
    logger.info("执行 Dense + Sparse 双路混合检索（基于中文分词与 RRF 合并）...")

    # 对查询词分别生成用于第一路检索的 Dense 向量
    query_dense_vectors = embeddings.embed_documents(query_chunks)

    for q_idx, (query_text, query_dense) in enumerate(zip(query_chunks, query_dense_vectors)):
        print("\n" + "=" * 45 + f" [查询 {q_idx + 1}: {query_text}] " + "=" * 45)

        # 3.1 第一路：稠密语义密集搜索
        search_param_dense = {
            "data": [query_dense],
            "anns_field": "text_dense",
            "param": {"metric_type": "COSINE", "nprobe": 10},
            "limit": SEARCH_LIMIT
        }
        request_dense = AnnSearchRequest(**search_param_dense)

        # 3.2 第二路：稀疏关键字词法搜索（自动匹配内置的中文 chinese 分词器参数）
        search_param_sparse = {
            "data": [query_text],
            "anns_field": "text_sparse",
            "param": {
                "metric_type": "BM25"  # 检索端无需传 analyzer_name，系统会自动对齐 schema 定义的 chinese 分词
            },
            "limit": SEARCH_LIMIT
        }
        request_sparse = AnnSearchRequest(**search_param_sparse)

        # 3.3 构建 RRF 融合重排器
        ranker = RRFRanker(k=60)

        # 3.4 发送多路混合搜寻请求
        results = client.hybrid_search(
            collection_name=COLLECTION_NAME,
            reqs=[request_dense, request_sparse],
            ranker=ranker,
            limit=SEARCH_LIMIT,
            output_fields=["doc_title", "doc_summary", "parent_idx", "parent_text", "parent_summary", "text"]
        )

        hits = results[0]

        # ================== 核心还原与级联去重：从子句重新聚合到大文章 ==================
        doc_groups = {}
        doc_order = []

        for hit in hits:
            entity = hit.get('entity', {})
            title = entity.get('doc_title', 'Unknown')
            if not title:
                continue

            if title not in doc_groups:
                doc_groups[title] = {
                    "doc_summary": entity.get('doc_summary', ''),
                    "max_rrf_score": hit.score,
                    "paragraphs": []
                }
                doc_order.append(title)

            doc_groups[title]["paragraphs"].append({
                "text": entity.get('text', ''),
                "score": hit.score,
                "parent_idx": entity.get('parent_idx', -1),
                "parent_text": entity.get('parent_text', ''),
                "parent_summary": entity.get('parent_summary', '')
            })

        # ================== 格式化输出：复用同一形式展现 ==================
        # 可能筛选出多个文档  此处没有对分数进行限制
        for doc_rank, title in enumerate(doc_order):
            doc_data = doc_groups[title]

            seen_parent_indices = set()
            deduplicated_paragraphs = []

            for p_dict in doc_data["paragraphs"]:
                p_idx = p_dict["parent_idx"]
                if p_idx not in seen_parent_indices:
                    seen_parent_indices.add(p_idx)
                    deduplicated_paragraphs.append(p_dict)

            selected_paragraphs = deduplicated_paragraphs[:TOP_K_CHUNKS_PER_DOC]

            print(f"\n【Rank {doc_rank + 1}】聚合大主题：《{title}》")
            print(f" ├─ 宏观大纲摘要: {doc_data['doc_summary']}")
            print(f" ├─ Hybrid + RRF 融合最大得分: {doc_data['max_rrf_score']:.4f}")
            print(f" ├─ 🎯 实际命中的微观单句 (Child chunks - 最多保留前 {TOP_K_CHUNKS_PER_DOC} 个):")

            if selected_paragraphs:
                for p_dict in selected_paragraphs:
                    print(f" │   * [融合 RRF 权重: {p_dict['score']:.4f}] {p_dict['text']}")
            else:
                print(" │   * (无任何单句返回)")

            print(" └─ 📄 重构并送入大模型的完整章节上下文 (Parent Chapters):")
            if selected_paragraphs:
                sorted_by_idx = sorted(selected_paragraphs, key=lambda x: x["parent_idx"])
                for p_dict in sorted_by_idx:
                    p_idx = p_dict["parent_idx"]
                    p_text = p_dict["parent_text"]
                    print(f"     --- [章节 {p_idx + 1} (全长 {len(p_text)} 字符)] ---")
                    print(f"     - 章节摘要: {p_dict['parent_summary']}")
                    print(f"     - 章节真实全文:")
                    print(f"       \"{p_text.replace('\n', '\n       ')}\"")
            else:
                print("     (未命中任何有效的父段落上下文)")


if __name__ == "__main__":
    try:
        logger.info(f"正在建立 Milvus 数据库连接: {MILVUS_URI} ...")
        db_client = MilvusClient(uri=MILVUS_URI, token=MILVUS_TOKEN)

        if not db_client.has_collection(COLLECTION_NAME):
            logger.error(
                f"指定的扁平索引集合 [{COLLECTION_NAME}] 在数据库中不存在，请先运行 ingest_pipeline.py 写入数据。")
            sys.exit(1)

        run_hybrid_search_test(db_client)
    except Exception as e:
        logger.error(f"检索测试程序执行异常: {e}", exc_info=True)
        sys.exit(1)