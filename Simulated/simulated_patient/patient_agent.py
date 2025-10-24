import os
import random
import json
import re
from pathlib import Path

from Simulated.simulated_patient.api_call import llm_api
from Simulated.simulated_patient.agent_evolve import store_patient_qa, agent_evolving_patient
from RAG.rag import rag_patient
from make_task.overall_assessment_llm import overall_assessment_patient


def question_detect(doctor_question: str) -> bool:
    """返回 True 表示问题过于泛泛；False 表示具体可答。"""
    json_file_path = Path("Simulated/Prompt/prompt_data.json")
    with json_file_path.open("r", encoding="utf-8") as file:
        data = json.load(file)
        prompt = "".join(data["question_general_detect"])
    prompt = prompt.format(question=doctor_question)
    messages = [{"role": "user", "content": prompt}]
    res = llm_api(messages)
    return not (("是" in res) or ("yes" in res.lower()))


def match_star(context: str) -> str:
    """匹配 **...** 并返回去星号后的原文；找不到则抛出异常。"""
    pattern = r"\*\*(.*?)\*\*"
    m = re.search(pattern, context, flags=re.DOTALL)
    if not m:
        raise ValueError("没有找到匹配项")
    return re.sub(r"\*", "", m.group(0))


def match_requirements(context: str) -> str:
    """在整段文本中匹配 **...**（跨行），返回第一次匹配内容；无则返回空串。"""
    pattern = r"\*\*(.*)\*\*"
    matches = re.findall(pattern, context, flags=re.DOTALL)
    return matches[0] if matches else ""


class Patient:
    def __init__(self, vague_info: str = "", resource: str = "", folder_path: str = "", prompt_data=None):
        self.profile = ""
        self.vague_info = vague_info
        self.resource = resource
        self.directory = Path(folder_path) if folder_path else Path(".")
        self.prompt_data = prompt_data or {}
        self.crisis = ""

        # 确保用于记录对话的目录存在
        self.directory.mkdir(parents=True, exist_ok=True)

    def generate_patient_question(self) -> str:
        prompt_tpl = self.prompt_data["patient_question_generator"]
        random_profile_num = random.randint(0, 100)
        profile_path = Path("profile") / "profile_pool" / f"{random_profile_num}.txt"
        self.profile = profile_path.read_text(encoding="utf-8") if profile_path.exists() else ""

        prompt = prompt_tpl.format(profile=self.profile, information=self.vague_info)
        messages = [{"role": "user", "content": prompt}]
        return llm_api(messages)

    def assign_office(self) -> str:
        prompt = self.prompt_data["assign_doctor_office"] + self.prompt_data["vague_resource"]
        messages = [{"role": "user", "content": prompt}]
        office = llm_api(messages)
        print("科室：", office)
        return office

    def patient_ans(self, question: str):
        # 如果需要，可打开对泛化问题的检测：
        # not_general_flag = question_detect(question)
        not_general_flag = True

        # 返回值的安全默认值
        score = rel = faith = human = 0
        ans = ""

        if not_general_flag:
            prompt_tpl = self.prompt_data["patient_answer_generator"]

            useful_info = rag_patient(
                question,
                self.resource,
                size=120,
                overlap=40,
                top_k=2,
            )

            # 进化样例检索（相似问答 few-shot）
            patient_evolve_csv = Path("dataset") / "patient_evolve.csv"
            patient_evolve_csv.parent.mkdir(parents=True, exist_ok=True)

            patient_evolve_info = agent_evolving_patient(str(patient_evolve_csv), question)
            attention_requirements = ""
            few_shot_example = ""

            if patient_evolve_info:
                lines = []
                for idx, info_dic in enumerate(patient_evolve_info, start=1):
                    doctor_qus = info_dic["question"]
                    patient_answer = info_dic["answer"]
                    rag_info = info_dic["rag_info"]
                    attention_requirements = info_dic.get("requirements", attention_requirements)
                    lines.append(f"{idx}\n问题：{doctor_qus}\n病情信息：{rag_info}\n病人回答：{patient_answer}\n")
                few_shot_example = "".join(lines)
            else:
                few_shot_example = "无示例。"

            if attention_requirements:
                prompt_tpl = (
                    "你是一个能够模仿一个没有专业医学知识病人口吻进行回答的回答生成器，\n"
                    "这个病人的角色扮演要求：{profile}\n---------\n"
                    "现在有一位医生向你提问，请你依据以下要求回答："
                    + attention_requirements
                    + "\n问题为：{question}。\n病情信息为：{information}。\n例子：{example}"
                )

            prompt = prompt_tpl.format(
                profile=self.profile,
                example=few_shot_example,
                question=question,
                information=useful_info,
            )

            messages = [{"role": "user", "content": prompt}]
            ans = llm_api(messages)

            # 质量评估（达到阈值则入库并动态抽取“注意事项”）
            score, rel, faith, human = overall_assessment_patient(question, useful_info, ans, self.profile)

            if score >= 3:
                dyn_req_tpl = self.prompt_data["dynamic_requirements"]
                dyn_prompt = dyn_req_tpl.format(question=question)
                dyn_msg = [{"role": "user", "content": dyn_prompt}]

                # 稳健提取一次 **...** 的 requirements
                for _ in range(3):
                    requirements_raw = llm_api(dyn_msg)
                    requirements = match_requirements(requirements_raw)
                    if requirements:
                        store_patient_qa(str(patient_evolve_csv), question, useful_info, ans, requirements)
                        break
        else:
            ans = "医生，这个问题太空泛了，要不就是我有点不太明白，问一些具体的吧，而且我也听不懂医学名词，要我去做检查倒是可以。"

        # 追加对话到相对路径文件
        dq_path = self.directory / "doctor_question.txt"
        with dq_path.open("a", encoding="utf-8") as f:
            f.write("--- dialog ---\n")
            f.write(f"doctor question: {question}\n")
            f.write(f"patient answer: {ans}\n")

        return ans, score, rel, faith, human

    def patient_crisis_ans(self, doctor_ans: str) -> str:
        prompt = self.prompt_data["patient_crisis_answer"].format(
            profile=self.profile, information=self.resource, crisis=self.crisis, doctor_answer=doctor_ans
        )
        messages = [{"role": "user", "content": prompt}]
        return llm_api(messages)

    def crisis_begin(self) -> str:
        prompt_tpl = self.prompt_data["patient_crisis_generator"]
        resource = self.prompt_data["resource"]
        prompt = prompt_tpl.format(information=resource)

        messages = [{"role": "user", "content": prompt}]
        patient_crisis = llm_api(messages)
        self.crisis = patient_crisis
        return patient_crisis
