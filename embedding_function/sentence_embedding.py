import os
from dotenv import load_dotenv
from openai import OpenAI


def get_embeddings(text: str):
    """从环境变量读取配置，调用 DashScope/OpenAI 接口生成文本向量。"""
    # 加载环境变量
    load_dotenv()

    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_API_BASE", "https://dashscope.aliyuncs.com/compatible-mode/v1")

    if not api_key:
        raise RuntimeError("未检测到 OPENAI_API_KEY，请在 .env 或系统环境中设置。")

    client = OpenAI(api_key=api_key, base_url=base_url)

    completion = client.embeddings.create(
        model="text-embedding-v3",
        input=text,
        encoding_format="float"
    ).model_dump()

    return completion["data"][0]["embedding"]
