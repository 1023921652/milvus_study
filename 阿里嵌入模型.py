import time
from langchain_openai import OpenAIEmbeddings

# ================= 配置区 =================
# 1. 请在此处填入您的阿里云 API 密钥 (API Key)
API_KEY = "sk-a42163d874e74c41923259805c86a453"

# 2. 目标 Endpoint 与模型名称
BASE_URL = "https://ws-oi8z1umy0fuyv6if.cn-beijing.maas.aliyuncs.com/compatible-mode/v1"
MODEL_NAME = "text-embedding-v4"

embeddings = OpenAIEmbeddings(
    api_key=API_KEY,
    base_url=BASE_URL,
    model=MODEL_NAME,
    check_embedding_ctx_length=False,  # 关键配置：关闭本地tiktoken校验，避免非OpenAI模型冲突
    dimensions=3072 # 可选：v4支持指定维度 (2048, 1536, 1024(默认), 768, 512等)
)

# 准备测试语料
text_a = "人工智能与大语言模型正在重塑搜索引擎的未来。"
text_b = "向量检索和RAG技术可以有效缓解大模型的幻觉问题。"
text_c = "今天中午吃了一碗味道很棒的北京炸酱面。"

# 测试一：单文本生成 (embed_query)
print("【测试一：正在请求单文本向量化（embed_query）】...")
vec_a = embeddings.embed_query(text_a)
print("\n【测试二：正在请求批量文本向量化（embed_documents）】...")
batch_vecs = embeddings.embed_documents([text_a, text_b, text_c])
pass