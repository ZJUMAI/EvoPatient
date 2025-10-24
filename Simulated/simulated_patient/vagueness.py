import json
import random
import re
from pathlib import Path

import openpyxl
from Simulated.simulated_patient.api_call import llm_api


# ---------- 工具函数 ----------

def select_random_positions(lst, percentage):
    """从列表中按百分比随机选择索引位置。"""
    if not lst or percentage <= 0:
        return []
    k = int(len(lst) * (percentage / 100.0))
    k = max(0, min(k, len(lst)))
    if k == 0:
        return []
    return random.sample(range(len(lst)), k)


def split_string_by_punctuation(text: str):
    """
    将文本按“标点”和“非标点”片段切开，保持顺序与原始字符。
    例：'a,b' -> ['a', ',', 'b']
    """
    pattern = re.compile(r'[^\w\s]|[\w\s]+', flags=re.UNICODE)
    return [m.group(0) for m in pattern.finditer(text or '')]


def random_dropout(split_tokens):
    """
    随机删除约 30% 位置附近的片段，基于简单启发式：
    - 命中数字：删除该位与后 1~2 位
    - 命中字母：删除该位，若前两位是数字则一并删除
    - 其它符号：尝试删除其相邻位（含数字的前一位等）
    """
    selected = select_random_positions(split_tokens, 30)
    to_delete = set()

    for pos in selected:
        token = split_tokens[pos]
        if token.isdigit():
            to_delete.add(pos)
            if pos + 1 < len(split_tokens):
                to_delete.add(pos + 1)
            if pos + 2 < len(split_tokens):
                to_delete.add(pos + 2)
        elif token.isalpha():
            if pos - 2 >= 0 and split_tokens[pos - 2].isdigit():
                to_delete.add(pos - 2)
            to_delete.add(pos)
        else:
            if pos - 1 >= 0 and split_tokens[pos - 1].isdigit():
                to_delete.add(pos - 1)
            if pos + 1 < len(split_tokens):
                to_delete.add(pos + 1)

    return [t for i, t in enumerate(split_tokens) if i not in to_delete]


def dropout_vague(text: str) -> str:
    tokens = split_string_by_punctuation(text or "")
    kept = random_dropout(tokens)
    return "".join(kept)


# ---------- 业务逻辑 ----------

def get_patient_info(sheet_name: str, row_number: int, col_number: int):
    """
    读取相对路径 dataset/patient_text.xlsx 的指定单元格，
    并将其写入 Simulated/Prompt/prompt_data.json 的 data["resource"][0]。
    """
    xlsx_path = Path("dataset") / "patient_text.xlsx"
    if not xlsx_path.is_file():
        raise FileNotFoundError(f"找不到文件：{xlsx_path}")

    wb = openpyxl.load_workbook(xlsx_path)
    try:
        sheet = wb[sheet_name]
        cell_value = sheet[row_number][col_number].value
    finally:
        wb.close()

    text = str(cell_value or "")

    json_path = Path("Simulated") / "Prompt" / "prompt_data.json"
    if not json_path.is_file():
        raise FileNotFoundError(f"找不到文件：{json_path}")

    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    # 确保结构存在
    if "resource" not in data or not isinstance(data["resource"], list):
        data["resource"] = [""]
    if not data["resource"]:
        data["resource"].append("")
    data["resource"][0] = text

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

    return text


def get_vague_patient_info(sheet_name: str, row_number: int, col_number: int):
    """
    读取病人信息 -> 做“模糊化” -> 调用 LLM 生成含糊表达 -> 写回 prompt_data.json 的 data["vague_resource"][0]。
    返回 (原始文本, 模糊文本)
    """
    patient_info = get_patient_info(sheet_name, row_number, col_number)
    patient_info_drop = dropout_vague(patient_info)

    json_path = Path("Simulated") / "Prompt" / "prompt_data.json"
    if not json_path.is_file():
        raise FileNotFoundError(f"找不到文件：{json_path}")

    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    # 组装 vagueness prompt
    if "vagueness" not in data or not isinstance(data["vagueness"], list):
        raise KeyError("prompt_data.json 缺少 'vagueness' 配置或其类型不是 list")
    vagueness_prompt = "".join(data["vagueness"])
    prompt = vagueness_prompt.format(information=patient_info_drop)

    vague_patient_info = llm_api([{"role": "user", "content": prompt}])

    # 写回 vague_resource
    if "vague_resource" not in data or not isinstance(data["vague_resource"], list):
        data["vague_resource"] = [""]
    if not data["vague_resource"]:
        data["vague_resource"].append("")
    data["vague_resource"][0] = vague_patient_info

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

    return patient_info, vague_patient_info
