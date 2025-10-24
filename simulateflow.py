from pathlib import Path
import re
import json
import csv
import time
import random
import shutil
import os

from dotenv import load_dotenv
from Simulated.simulated_patient.vagueness import get_vague_patient_info
from Simulated.simulated_patient.patient_agent import Patient
from Simulated.simulated_patient.doctor_agent import Doctor
from Simulated.simulated_patient.api_call import llm_api  # 用于 LLM 调用（内部应读取环境变量）
# 若需要：from Simulated.simulated_patient.agent_evolve import get_text_embedding


# ====== 环境变量（密钥不落地，实际读取在 llm_api 内部）======
load_dotenv()
if not os.getenv("OPENAI_API_KEY"):
    raise RuntimeError("未检测到 OPENAI_API_KEY，请在系统或 .env 中设置。")


# ====== 工具函数 ======
def ensure_parent(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)


def match_star(context: str) -> str:
    """匹配 **...** 并返回去星号后的原文；找不到则抛出异常。"""
    m = re.search(r"\*\*(.*?)\*\*", context, flags=re.DOTALL)
    if not m:
        raise ValueError("没有找到匹配项")
    return re.sub(r"\*", "", m.group(0))


def read_prompt() -> dict:
    """读取并拼接 Simulated/Prompt/prompt_data.json 中的各条 prompt。"""
    p = Path("Simulated/Prompt/prompt_data.json")
    data = json.loads(p.read_text(encoding="utf-8"))
    final = {}
    for k, v in data.items():
        final[k] = "".join(v) if isinstance(v, list) else str(v)
    return final


def clean_token_count():
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


def count_chinese_characters(text: str) -> int:
    return len(re.findall(r"[\u4e00-\u9fff]", text))


# ====== 主流程 ======
def flow(sheet_name: str = "病程记录_首次病程", row_number: int = 6, col_number: int = 1):
    clean_token_count()

    test_label = str(time.time())
    parent_folder = Path("exp1")
    directory = parent_folder / test_label
    (directory / "doctor_record").mkdir(parents=True, exist_ok=True)
    print(f"文件夹 {test_label} 已创建在 {parent_folder} 中。")

    # 读取患者“完整信息 + 模糊信息”
    resource, vague_info = get_vague_patient_info(sheet_name, row_number, col_number)

    # 保存原始资源和模糊信息
    (directory / "resource.txt").write_text(resource, encoding="utf-8")
    (directory / "vague.txt").write_text(vague_info, encoding="utf-8")

    prompt_data = read_prompt()
    patient = Patient(vague_info, resource, str(directory), prompt_data)

    # 分配科室（从 **...** 提取科室名）
    office = match_star(patient.assign_office())

    # 生成主诉
    def generate_main_complaint() -> str:
        patient_question = patient.generate_patient_question()
        patient_answer = match_star(patient_question)
        print("Patient question:", patient_answer)
        return patient_answer

    main_complaint = generate_main_complaint()

    # 初始化医生并发起首问
    doctor = Doctor(patient, office, main_complaint, str(directory), prompt_data)
    resp = doctor.doctor_qus(main_complaint, 0, 0, 0, 0)
    try:
        doctor_question = match_star(resp)
    except Exception:
        doctor_question = resp

    # 患者首次回答
    patient_answer, score, rel, faith, human = patient.patient_ans(doctor_question)

    max_turn = 10
    cnt = 0
    auto = True
    random_crisis_num = random.randrange(int(max_turn / 2), max_turn)

    # 轮次记录（实验目录内 + 顶层便捷文件）
    exp_csv = directory / "question_record.csv"
    top_csv = Path("question_record.csv")
    for p in (exp_csv, top_csv):
        if not p.exists():
            ensure_parent(p)
            with p.open("w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "row", "question", "answer", "token_count_doctor", "token_count_patient",
                    "resource", "doctor_time", "patient_time", "question_cnt", "answer_cnt"
                ])

    while cnt < max_turn:
        cnt += 1

        # 随机时刻引入“危机事件”
        if cnt == random_crisis_num:
            patient_crisis = patient.crisis_begin()
            doctor_crisis_ans = doctor.doctor_crisis_answer(office, patient_crisis)
            patient_ans_after_crisis = patient.patient_crisis_ans(doctor_crisis_ans)

            crisis_file = directory / "crisis.txt"
            crisis_txt = (
                f"轮数：**{cnt}**病人紧急情况：**{patient_crisis}**"
                f"医生对紧急情况的回复**{doctor_crisis_ans}**"
                f"病人对医生回复的反应**{patient_ans_after_crisis}**"
            )
            crisis_file.write_text(crisis_txt, encoding="utf-8")

        start_time = time.time()

        if auto:
            # 医生提问
            next_q = doctor.doctor_qus(patient_answer, score, rel, faith, human)
            # 修复：判断应基于本次返回的问题而非首次的 resp
            if next_q == "skip":
                continue
            if next_q == "conclusion":
                print("已获取足够信息，提前停止")
                break
            try:
                doctor_question = match_star(next_q)
            except Exception:
                doctor_question = next_q
        else:
            doctor_question = input("请输入问题：")

        token_count_doctor = get_token_count()
        middle_time = time.time()

        # 患者回答
        patient_answer, score, rel, faith, human = patient.patient_ans(doctor_question)
        token_count_patient = get_token_count()
        end_time = time.time()

        doctor_time = middle_time - start_time
        patient_time = end_time - start_time

        # 写入两份 CSV（实验目录与顶层）
        row = [
            row_number,
            doctor_question.replace("\n", ""),
            patient_answer.replace("\n", ""),
            token_count_doctor,
            token_count_patient,
            resource,
            doctor_time,
            patient_time,
            count_chinese_characters(doctor_question),
            count_chinese_characters(patient_answer),
        ]
        for p in (exp_csv, top_csv):
            with p.open("a", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(row)

    # 结论
    conclusion = doctor.conclusion()
    (directory / "conclusion.txt").write_text(conclusion, encoding="utf-8")

    # 统计耗时（从最后一轮开始计时）
    time_cost = time.time() - start_time
    (directory / "time_cost.txt").write_text(str(time_cost), encoding="utf-8")

    # 迁移 token 统计文件
    source_folder = Path("./make_task/token_count")
    destination_folder = directory / "token_count"
    destination_folder.mkdir(parents=True, exist_ok=True)
    if source_folder.exists():
        for file_path in source_folder.glob("*.txt"):
            shutil.move(str(file_path), str(destination_folder))
            print(f"文件 {file_path.name} 已移动到 {destination_folder}")
