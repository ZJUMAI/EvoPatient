from pathlib import Path
import os
import re
import json
import csv
import time

from dotenv import load_dotenv
from Simulated.simulated_patient.vagueness import get_vague_patient_info
from Simulated.simulated_patient.patient_agent import Patient
from Simulated.simulated_patient.api_call import llm_api
from Simulated.simulated_patient.agent_evolve import get_text_embedding


# ============== 环境变量（隐去隐私信息，密钥由 llm_api 内部读取） ==============
load_dotenv()
if not os.getenv("OPENAI_API_KEY"):
    raise RuntimeError("未检测到 OPENAI_API_KEY，请在系统或 .env 中设置。")


# ============== 工具函数 ==============
def match_star(context: str) -> str:
    """匹配 **...** 并返回去星号后的原文；找不到则抛出异常。"""
    m = re.search(r"\*\*(.*?)\*\*", context, flags=re.DOTALL)
    if not m:
        raise ValueError("没有找到匹配项")
    return re.sub(r"\*", "", m.group(0))


def read_prompt() -> dict:
    """读取并拼接 Simulated/Prompt/prompt_data.json 中的各条 prompt。"""
    p = Path("Simulated/Prompt/prompt_data.json")
    with p.open("r", encoding="utf-8") as f:
        data = json.load(f)
    final = {}
    for k, v in data.items():
        # v 可能是列表（每行一段），拼接为完整 prompt
        if isinstance(v, list):
            final[k] = "".join(v)
        else:
            final[k] = str(v)
    return final


def ensure_parent(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)


def clean_token_count():
    """清空 token 统计文件（相对路径）。"""
    tp_overall = Path("./make_task/token_count/token_overall.txt")
    tp_stream = Path("./make_task/token_count/token_stream.txt")
    ensure_parent(tp_overall)
    tp_overall.write_text("", encoding="utf-8")
    tp_stream.write_text("", encoding="utf-8")


def get_token_count() -> str:
    """读取累计 token（若不存在或为空则返回 '0'）。"""
    tp_stream = Path("./make_task/token_count/token_stream.txt")
    if not tp_stream.exists():
        return "0"
    token_str = tp_stream.read_text(encoding="utf-8")
    nums = re.findall(r"\d+", token_str)
    return nums[-1] if nums else "0"


# ============== 主流程 ==============
def cover(sheet_name: str = "病程记录_首次病程", row_number: int = 6, col_number: int = 1):
    clean_token_count()

    test_label = str(time.time())
    parent_folder = Path("pool")
    directory = parent_folder / test_label
    (directory / "doctor_record").mkdir(parents=True, exist_ok=True)
    print(f"文件夹 {test_label} 已创建在 {parent_folder} 中。")

    # 获取病人完整与模糊信息
    resource, vague_info = get_vague_patient_info(sheet_name, row_number, col_number)

    # 保存 resource 与 vague 文本（相对路径）
    (directory / "resource.txt").write_text(resource, encoding="utf-8")
    (directory / "vague.txt").write_text(vague_info, encoding="utf-8")

    prompt_data = read_prompt()
    patient = Patient(vague_info, resource, str(directory), prompt_data)

    # 分配科室
    office = match_star(patient.assign_office())

    # 生成主诉
    def generate_main_complaint() -> str:
        patient_question = patient.generate_patient_question()
        patient_answer = match_star(patient_question)
        print("Patient question:", patient_answer)
        return patient_answer

    main_complaint = generate_main_complaint()

    # 生成封面
    prompt = prompt_data["cover"].format(office, resource)
    messages = [{"role": "user", "content": prompt}]
    response = llm_api(messages)
    print(response)

    # 提取封面中可能的多项匹配
    matched = re.findall(r"\*\*(.*?)\*\*", response, flags=re.DOTALL)

    # 嵌入主诉并写入池表
    emb = get_text_embedding(main_complaint)

    pool_csv = Path("dataset/pool.csv")
    ensure_parent(pool_csv)
    new_file = not pool_csv.exists()
    with pool_csv.open("a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if new_file:
            writer.writerow(["main_complaint", "embedding_main_complaint", "question"])
        writer.writerow([
            main_complaint,
            json.dumps(emb, ensure_ascii=False),         # 防止逗号影响 CSV
            json.dumps(matched, ensure_ascii=False)      # 以 JSON 形式保存列表
        ])


def cache() -> int:
    """读取/初始化 case_cache.txt 内的行号。"""
    cache_path = Path("./make_task/case_cache.txt")
    ensure_parent(cache_path)
    if not cache_path.exists():
        cache_path.write_text("0", encoding="utf-8")
        return 0
    txt = cache_path.read_text(encoding="utf-8").strip()
    return int(txt) if txt.isdigit() else 0


def write_cache(value: int):
    cache_path = Path("./make_task/case_cache.txt")
    ensure_parent(cache_path)
    cache_path.write_text(str(value), encoding="utf-8")


if __name__ == "__main__":
    col_number = 1
    sheet_name = "病程记录_首次病程"

    row_number = cache()
    while row_number <= 1300:
        row_number += 1
        cover(sheet_name, row_number, col_number)
        write_cache(row_number)
