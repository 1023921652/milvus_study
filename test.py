import time
from langchain_openai import OpenAIEmbeddings

# ================= 配置区 =================
# 1. 请在此处填入您的阿里云 API 密钥 (API Key)
API_KEY = "sk-a42163d874e74c41923259805c86a453"

# 2. 目标 Endpoint 与模型名称
BASE_URL = "https://ws-oi8z1umy0fuyv6if.cn-beijing.maas.aliyuncs.com/compatible-mode/v1"
MODEL_NAME = "text-embedding-v4"


# ================= 纯 Python 相似度计算函数 =================
def cosine_similarity(v1, v2):
    """计算两个向量的余弦相似度，用于语义测试"""
    dot_product = sum(a * b for a, b in zip(v1, v2))
    norm_a = sum(a * a for a in v1) ** 0.5
    norm_b = sum(b * b for b in v2) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot_product / (norm_a * norm_b)


# ================= 主测试流程 =================
def run_test():
    print(f"正在通过 LangChain 初始化 {MODEL_NAME} 客户端...")
    print(f"网关地址: {BASE_URL}\n")

    try:
        # 初始化客户端
        embeddings = OpenAIEmbeddings(
            api_key=API_KEY,
            base_url=BASE_URL,
            model=MODEL_NAME,
            check_embedding_ctx_length=False,  # 关键配置：关闭本地tiktoken校验，避免非OpenAI模型冲突
            # dimensions=1024 # 可选：v4支持指定维度 (2048, 1536, 1024(默认), 768, 512等)
        )

        # 准备测试语料
        text_a = "人工智能与大语言模型正在重塑搜索引擎的未来。"
        text_b = "向量检索和RAG技术可以有效缓解大模型的幻觉问题。"
        text_c = "今天中午吃了一碗味道很棒的北京炸酱面。"

        # 测试一：单文本生成 (embed_query)
        print("【测试一：正在请求单文本向量化（embed_query）】...")
        start_time = time.time()
        vec_a = embeddings.embed_query(text_a)
        vec_b = embeddings.embed_query(text_b)
        vec_c = embeddings.embed_query(text_c)
        print(f"-> 耗时: {time.time() - start_time:.2f} 秒")
        print(f"-> 向量生成成功！维度大小为: {len(vec_a)}")
        print(f"-> 向量数据前5位样例: {vec_a[:5]}")

        # 测试二：批量文本生成 (embed_documents)
        print("\n【测试二：正在请求批量文本向量化（embed_documents）】...")
        start_time = time.time()
        batch_vecs = embeddings.embed_documents([text_a, text_b, text_c])
        print(f"-> 耗时: {time.time() - start_time:.2f} 秒")
        print(f"-> 批量生成成功！共获取到 {len(batch_vecs)} 个文档向量。")

        # 测试三：语义逻辑验证 (通过余弦相似度检查数据是否具有语义特异性)
        sim_a_b = cosine_similarity(vec_a, vec_b)  # 都是技术相关
        sim_a_c = cosine_similarity(vec_a, vec_c)  # 技术 vs 餐饮（无关）

        print("\n【测试三：语义关联验证】")
        print(f"文本 A: \"{text_a}\"")
        print(f"文本 B: \"{text_b}\"")
        print(f"文本 C: \"{text_c}\"")
        print(f"-> 文本 A 与 文本 B (技术相关) 语义相似度: {sim_a_b:.4f}")
        print(f"-> 文本 A 与 文本 C (食物无关) 语义相似度: {sim_a_c:.4f}")

        if sim_a_b > sim_a_c:
            print("\n>>> [测试结果]: 通过。语义相近的文本具有更高的相似度评分。")
        else:
            print(
                "\n>>> [测试异常]: 尽管接口成功返回，但相似度分布异常，请确保输入没有被截断。"
            )

    except Exception as e:
        print("\n>>> [测试失败]: 发生未预期错误，详细异常信息如下：")
        print(e)
        print("\n排查建议：")
        print(
            "1. 确认您的 API Key 填写无误，且对应的百炼账户在此专用空间拥有 text-embedding-v4 的调用权限。"
        )
        print(
            "2. 确保您的当前网络环境可以访问该阿里云专属 Endpoint 地址（若为阿里云专有网络 VPC 内部网关，本地公网测试可能会由于路由不通而超时）。"
        )


if __name__ == "__main__":
    run_test()