import os
import re
from openai import OpenAI

# ===== 从环境变量中读取配置 =====
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
BASE_URL = os.getenv("BASE_URL")  # 例如：https://ark.cn-beijing.volces.com/api/v3

if not OPENAI_API_KEY:
    raise ValueError("❌ 未找到 OPENAI_API_KEY 环境变量，请先在系统中设置。")

client_kwargs = {"api_key": OPENAI_API_KEY}
if BASE_URL:
    client_kwargs["base_url"] = BASE_URL

client = OpenAI(**client_kwargs)


# ===== 功能函数 =====
def token_counter(usage):
    file_path_overall = 'make_task/token_count/token_overall.txt'
    file_path_stream = 'make_task/token_count/token_stream.txt'

    with open(file_path_stream, 'r', encoding='utf-8') as file:
        token_str_stream = file.read()
    with open(file_path_overall, 'r', encoding='utf-8') as file:
        token_str_overall = file.read()

    with open(file_path_stream, 'w', encoding='utf-8') as tok:
        for key, value in usage.items():
            token_str_stream += f"{key}: **{value}**\n"
        tok.write(token_str_stream)

    if not token_str_overall.strip():
        with open(file_path_overall, 'w', encoding='utf-8') as tok:
            tok.write(token_str_stream)
    else:
        numbers = [int(num) for num in re.findall(r'\d+', token_str_overall)]
        token_str_new = ''
        with open(file_path_overall, 'w', encoding='utf-8') as tok:
            for i, (key, value) in enumerate(usage.items()):
                token_str_new += f"{key}: **{value + numbers[i]}**\n"
            tok.write(token_str_new)


def llm_api(messages):
    response = client.chat.completions.create(
        messages=messages,
        model='ep-20240814160016-j24nr',
        temperature=0.2,
        top_p=1.0,
        n=1,
        stream=False,
        frequency_penalty=0.0,
        presence_penalty=0.0,
        logit_bias={}
    ).model_dump()
    response_text = response['choices'][0]['message']['content']
    token_counter(response['usage'])
    return response_text


def llm_api_lite(messages):
    response = client.chat.completions.create(
        messages=messages,
        model='ep-20240822150534-5nj65',
        temperature=0.2,
        top_p=1.0,
        n=1,
        stream=False,
        frequency_penalty=0.0,
        presence_penalty=0.0,
        logit_bias={}
    ).model_dump()
    response_text = response['choices'][0]['message']['content']
    token_counter(response['usage'])
    return response_text


def get_text_embedding(text: str):
    text = text or "None"
    return client.embeddings.create(
        input=text,
        model="text-embedding-ada-002"
    ).model_dump()['data'][0]['embedding']


def get_code_embedding(code: str):
    code = code or "#"
    return client.embeddings.create(
        input=code,
        model="text-embedding-ada-002"
    ).model_dump()['data'][0]['embedding']
