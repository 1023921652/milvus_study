import time
from pymilvus import MilvusClient, DataType
from pymilvus.client.embedding_list import EmbeddingList
from langchain_openai import OpenAIEmbeddings

# ================= 1. 配置并初始化向量模型 =================
# 阿里云 API 密钥 (API Key)
API_KEY = "sk-a42163d874e74c41923259805c86a453"
BASE_URL = "https://ws-oi8z1umy0fuyv6if.cn-beijing.maas.aliyuncs.com/compatible-mode/v1"
MODEL_NAME = "text-embedding-v4"
VECTOR_DIM = 2048  # 统一定义向量维度为 512

embeddings = OpenAIEmbeddings(
    api_key=API_KEY,
    base_url=BASE_URL,
    model=MODEL_NAME,
    check_embedding_ctx_length=False,  # 关闭本地 tiktoken 校验，避免非 OpenAI 模型冲突
    dimensions=VECTOR_DIM  # 强制约束输出维度与 Milvus 一致
)

# ================= 2. 准备 10 个多片段（Chunk）测试文档 =================
# 每一个文档都是一个 Entity，包含多个在语义上独立（通过断句或换行）的 Paragraph (Chunk)
raw_documents = [
    {
        "title": "大语言模型与搜索引擎的未来",
        "paragraphs": [
            "人工智能与大语言模型正在重塑搜索引擎的未来。传统的关键词匹配正在被深度的语义理解所取代。",
            "通过生成式回答，搜索引擎不再仅仅给出网页链接，而是直接将多源信息整合并呈现给用户。"
        ]
    },
    {
        "title": "向量数据库的核心原理",
        "paragraphs": [
            "向量检索通过将文本、图像等高维数据转化为密集向量，并在高维空间中计算彼此的距离来衡量相似度。",
            "Milvus 等现代向量数据库通过 HNSW、IVF 等高效索引算法，支持在海量数据中实现毫秒级的近似最近邻搜索。"
        ]
    },
    {
        "title": "缓解大模型幻觉的 RAG 技术",
        "paragraphs": [
            "检索增强生成（RAG）技术通过在生成答案前，先从外部知识库中检索出相关的真实事实，以此提供可信上下文。",
            "这种机制可以显著缓解大语言模型由于缺乏最新知识或训练数据限制而产生的“幻觉”现象。"
        ]
    },
    {
        "title": "北京炸酱面的传统风味",
        "paragraphs": [
            "今天中午吃了一碗味道很棒的北京炸酱面，酱香浓郁，菜码丰富。",
            "正宗的北京炸酱面讲究“小碗干炸”，肉丁肥瘦相间，搭配黄瓜丝、豆芽等时令配菜，口感丰富。"
        ]
    },
    {
        "title": "气候变化与全球变暖",
        "paragraphs": [
            "气候变化已成为全球面临的最严峻挑战之一。温室气体排放导致全球气温逐年上升，冰川加速融化。",
            "各国政府正在积极推动碳中和政策，通过发展风能、太阳能等清洁能源来减少化石燃料的使用。"
        ]
    },
    {
        "title": "自动驾驶技术的发展现状",
        "paragraphs": [
            "自动驾驶汽车利用激光雷达、摄像头以及先进的感知算法，能够实时重构周围的交通场景并做出决策。",
            "目前许多城市已经开展了自动驾驶出租车（Robotaxi）的商业化运营试点，但在应对极端天气和复杂路况方面仍面临挑战。"
        ]
    },
    {
        "title": "太空探索与火星移民计划",
        "paragraphs": [
            "人类对宇宙的探索从未停止。火星作为距离地球较近的行星，成为了科学家们研究行星演化和寻找生命迹象的重点对象。",
            "商业航天公司正致力于开发重型运载火箭，旨在降低发射成本，在未来几十年内实现人类在火星的常态化居住。"
        ]
    },
    {
        "title": "咖啡的起源与冲煮文化",
        "paragraphs": [
            "咖啡起源于埃塞俄比亚，经过数百年的传播，如今已成为全球最受欢迎的饮品之一。",
            "从经典的意式浓缩，到讲究风味层次的手冲咖啡，不同的冲煮器具和水温调控展现了丰富的咖啡美学。"
        ]
    },
    {
        "title": "量子计算与密码学安全",
        "paragraphs": [
            "量子计算利用量子叠加和纠缠状态，能够在特定计算任务上实现超越传统超级计算机的“量子霸权”。",
            "这对基于大数分解的现代加密算法（如RSA）构成了潜在威胁，因此后量子密码学的研发迫在眉睫。"
        ]
    },
    {
        "title": "运动对身心健康的影响",
        "paragraphs": [
            "规律的有氧运动，如慢跑或游泳，不仅能够增强心肺功能、改善身体代谢，还能促进大脑释放多巴胺。",
            "这有助于缓解日常工作和学习带来的焦虑与压力，提升睡眠质量，维持长期的心理健康。"
        ]
    }
]

# 调用阿里云 API，批量为这 10 个文档的各个段落（Chunks）生成向量并组装格式
print("正在通过阿里云接口为 10 个文档的片段生成向量...")
formatted_data = []
for doc in raw_documents:
    # 批量将当前文档的段落文本转化为 512 维向量
    chunk_embeddings = embeddings.embed_documents(doc["paragraphs"])

    # 构建嵌入结构体列表
    paragraphs_list = []
    for text, emb in zip(doc["paragraphs"], chunk_embeddings):
        paragraphs_list.append({
            "text": text,
            "emb": emb
        })

    # 组装符合 Milvus 格式的一行 Entity
    formatted_data.append({
        "title": doc["title"],
        "paragraphs": paragraphs_list
    })
print("所有文档向量化处理完成。")

# ================= 3. 初始化 Milvus 数据库与 Schema =================
client = MilvusClient(
    uri="http://localhost:19530",
    token="root:Milvus"
)

collection_name = 'wiki_documents'

# 如果集合已存在，先将其删除（以便多次测试）
if client.has_collection(collection_name):
    client.drop_collection(collection_name)

schema = client.create_schema()
schema.add_field('id', DataType.INT64, is_primary=True, auto_id=True)
schema.add_field('title', DataType.VARCHAR, max_length=512)

# 定义嵌套结构体：包含分段文本和该片段单独对应的向量
struct_schema = client.create_struct_field_schema()
struct_schema.add_field('text', DataType.VARCHAR, max_length=65535)
struct_schema.add_field('emb', DataType.FLOAT_VECTOR, dim=VECTOR_DIM)  # 维度 512

# 声明一段文档由多个结构体（Paragraph Structs）构成，上限 200 个 Chunk
schema.add_field('paragraphs', DataType.ARRAY,
                 element_type=DataType.STRUCT,
                 struct_schema=struct_schema, max_capacity=200)

index_params = client.prepare_index_params()
index_params.add_index(
    field_name="paragraphs[emb]",  # 针对结构体数组内的 emb 向量子字段建索引
    index_type="AUTOINDEX",
    metric_type="MAX_SIM_COSINE"  # 使用 MAX_SIM 进行后期交互计算
)

client.create_collection(
    collection_name=collection_name,
    schema=schema,
    index_params=index_params
)

# ================= 4. 插入多段落（Chunk级）文档 =================
print("正在将结构化文档和片段向量写入 Milvus...")
client.insert(
    collection_name=collection_name,
    data=formatted_data
)
print("写入成功。")

# ================= 5. 执行后期交互（Late Interaction）检索测试 =================
# 我们模拟一个带有双重检索意图的复杂查询："如何用 RAG 缓解幻觉，以及吃碗北京炸酱面"
# 该查询并不直接合成一个向量，而是拆分为 2 个语义子意图：
query_chunks = [
    "如何用RAG技术缓解大语言模型的幻觉问题",
    "今天中午吃北京炸酱面"
]

print(f"\n正在将查询拆分为语义子意图: {query_chunks}")
# 分别对这两个子意图生成各自的 512 维向量
query_vectors = embeddings.embed_documents(query_chunks)

# 使用 Milvus 的 EmbeddingList 组织该查询的多向量组合
query_emb_list = EmbeddingList()
for vec in query_vectors:
    query_emb_list.add(vec)
#
# print("正在执行 Late Interaction (MAX_SIM_COSINE) 联合检索...")
# results = client.search(
#     collection_name=collection_name,
#     data=[query_emb_list],  # 传入包装好的一组查询向量
#     anns_field="paragraphs[emb]",  # 锁定检索字段
#     search_params={
#         "metric_type": "MAX_SIM_COSINE"
#     },
#     limit=5,
#     output_fields=["title"]
# )
#
# # ================= 6. 打印检索结果 =================
# print("\n=== 检索结果 (基于文档内各 Chunk 与 Query 各分量的 MAX_SIM 评分排名) ===")
# for hit in results[0]:
#     print(f"文档标题: 【{hit['entity']['title']}】 -> 最终得分: {hit['distance']:.4f}")

import numpy as np


# ================= 辅助函数：计算余弦相似度并提取匹配段落 =================
def get_cosine_similarity(v1, v2):
    """计算两个向量的余弦相似度"""
    dot_product = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    if norm_v1 == 0 or norm_v2 == 0:
        return 0.0
    return dot_product / (norm_v1 * norm_v2)


def extract_winning_chunks(query_vectors, paragraphs, threshold=0.3, window_size=1):
    """
    从文档实体中提取对 MAX_SIM 贡献最大的段落，并支持滑窗上下文
    :param query_vectors: 查询的多个向量 (list of lists)
    :param paragraphs: 从 Milvus 查出来的该文档的段落数组 (list of dicts)
    :param threshold: 相似度阈值，低于该阈值的段落即使匹配也不提取
    :param window_size: 邻近段落窗口大小。如果匹配到第 k 段，则同时提取 [k-window_size, k+window_size] 范围的文本作为上下文
    """
    matched_indices = set()
    num_paragraphs = len(paragraphs)

    # 1. 遍历每一个子意图查询向量，找出与其最相似的那个段落（MAX_SIM 逻辑）
    for q_vec in query_vectors:
        best_idx = -1
        best_score = -1.0

        for idx, p in enumerate(paragraphs):
            sim = get_cosine_similarity(q_vec, p['emb'])
            if sim > best_score:
                best_score = sim
                best_idx = idx

        # 2. 如果最高相似度大于设定的阈值，则将该段落记录为“获胜段落”
        if best_score >= threshold:
            matched_indices.add(best_idx)

    # 3. 进阶：滑窗上下文（防止“信息断片”）
    # 如果找到了最匹配的第 k 段，把它的前一段和后一段也带上，确保 LLM 获得连贯的语义
    extended_indices = set()
    for idx in matched_indices:
        start = max(0, idx - window_size)
        end = min(num_paragraphs - 1, idx + window_size)
        for i in range(start, end + 1):
            extended_indices.add(i)

    # 4. 按文章的原始物理顺序将筛选后的段落拼接
    winning_texts = []
    for idx in sorted(list(extended_indices)):
        winning_texts.append(paragraphs[idx]['text'])

    return winning_texts


# ================= 5. 执行检索（修改 output_fields） =================
print("正在执行 Late Interaction (MAX_SIM_COSINE) 联合检索...")
results = client.search(
    collection_name=collection_name,
    data=[query_emb_list],
    anns_field="paragraphs[emb]",
    search_params={
        "metric_type": "MAX_SIM_COSINE"
    },
    limit=3,  # 召回 Top-3 文档实体
    output_fields=["title", "paragraphs"]  # 【核心修改】不仅返回 title，同时要求返回 paragraphs
)

# ================= 6. 打印精简后的检索结果 =================
print("\n=== 检索结果 (已在客户端通过 MAX_SIM 贡献度，精简段落上下文) ===")
for hit in results[0]:
    doc_title = hit['entity']['title']
    doc_score = hit['distance']
    doc_paragraphs = hit['entity']['paragraphs']  # 包含文本 'text' 和向量 'emb' 的结构体列表

    # 调用提取函数，获取真正相关的段落（包含前后 1 段的滑窗上下文）
    reduced_chunks = extract_winning_chunks(
        query_vectors=query_vectors,
        paragraphs=doc_paragraphs,
        threshold=0.3,
        window_size=0  # 设为 0 表示只取最匹配的那一段；设为 1 表示取该段及其前后相邻段落
    )

    print(f"\n文档标题: 【{doc_title}】 -> 最终得分: {doc_score:.4f}")
    print(" └── 提取出的核心上下文:")
    for i, chunk_text in enumerate(reduced_chunks):
        print(f"     [段落 {i + 1}]: {chunk_text}")