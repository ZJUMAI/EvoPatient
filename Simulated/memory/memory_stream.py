import os
from pathlib import Path
from simulated_patient.api_call import llm_api


# === 相对路径定义 ===
MEMORY_FILE = Path("memory_warehouse.txt")
SUMMARY_PROMPT_FILE = Path("Prompt") / "summary_memory.txt"

# 如果不存在文件或文件夹则自动创建
MEMORY_FILE.touch(exist_ok=True)
SUMMARY_PROMPT_FILE.parent.mkdir(parents=True, exist_ok=True)
if not SUMMARY_PROMPT_FILE.exists():
    SUMMARY_PROMPT_FILE.write_text("请总结以下对话内容：{chatstream}", encoding="utf-8")


def summary(patient_question: str, doctor_answer: str):
    """定期调用，利用 llm_api 对 doctor-patient 对话记忆进行总结。"""
    memory_stream = MEMORY_FILE.read_text(encoding="utf-8") if MEMORY_FILE.exists() else ""
    memory_stream += f"医生的问题：{patient_question}\n患者的回答：{doctor_answer}\n"

    prompt_template = SUMMARY_PROMPT_FILE.read_text(encoding="utf-8")
    prompt = prompt_template.format(chatstream=memory_stream)

    messages = [{"role": "user", "content": prompt}]
    memory_summary = llm_api(messages)

    MEMORY_FILE.write_text(memory_summary, encoding="utf-8")
    print("✅ 已更新记忆摘要。")


def memory_store(patient_question: str, doctor_answer: str, turn: int):
    """
    每轮问答后记录 doctor-patient 对话。
    每 10 轮自动进行一次总结（summary）。
    """
    if turn % 10 == 0:
        summary(patient_question, doctor_answer)
    else:
        chat_pair = f"医生的问题：{patient_question}\n患者的回答：{doctor_answer}\n"
        with MEMORY_FILE.open("a", encoding="utf-8") as f:
            f.write(chat_pair)
        print(f"✅ 已追加到记忆文件（第 {turn} 轮）。")
