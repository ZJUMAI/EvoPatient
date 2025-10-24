import re
import os
import csv
import json
from pathlib import Path

from Simulated.simulated_patient.api_call import llm_api
from Simulated.simulated_patient.agent_evolve import store_doctor_qa, agent_evolving_doctor
from make_task.overall_assessment_llm import overall_assessment_doctor
from RAG.rag import rag_patient


def match_star(context, symbol):
    pattern = f'{symbol}{symbol}(.*?){symbol}{symbol}'
    match = re.search(pattern, context)
    if match:
        question_match = re.sub(symbol, "", match[0])
    else:
        question_match = "NO"
        print("no matching!")
        print(context)
    return question_match


def crisis_memory_summary(memory_info, prompt):
    prompt = prompt.format(chatstream=memory_info)
    messages = [{"role": "user", "content": prompt}]
    summarized_memory = llm_api(messages)
    return summarized_memory


def memory_control():
    pass


class Doctor:
    def __init__(self, patient, office=None, main_complaint=None, directory=None, prompt_data=None, level=1):
        self.sub_doctor = []
        self.office = office
        self.main_complaint = main_complaint
        self.dialog_history = ""
        self.summary = ""
        self.dialog_turn = 0
        self.summary_trun = 3
        self.new_patient_answer = ""
        self.directory = Path(directory) if directory else Path(".")
        self.patient = patient
        self.prompt_data = prompt_data
        self.last_qus = ""
        self.last_qus_category = ""
        self.last_score = 0
        self.record = ["", "", ""]
        self.level = level
        self.last_score_patient = 0
        self.new_patient_score = 0
        self.last_rel = 0
        self.last_faith = 0
        self.last_human = 0
        self.new_rel = 0
        self.new_faith = 0
        self.new_human = 0
        self.last_doc_rel = 0
        self.last_doc_faith = 0

    def doctor_qus(self, answer, patient_score, rel, faith, human):
        print(f"{self.office} Doctor question {self.last_qus}")
        print("Patient answer", answer)

        # 相对路径：dataset/doctor_evolve_{office}.csv
        evolve_csv = Path("dataset") / f"doctor_evolve_{self.office}.csv"
        evolve_csv.parent.mkdir(parents=True, exist_ok=True)

        if not evolve_csv.is_file():
            with evolve_csv.open("w", newline="", encoding="gbk") as file:
                writer = csv.writer(file)
                writer.writerow([
                    "question1", "qus_embedding", "rag_info1", "answer1",
                    "qus2_embedding", "question2", "answer2", "rag_info2"
                ])

        doctor_evolve_info = agent_evolving_doctor(
            str(evolve_csv),
            self.record
        )

        few_shot_example = ""
        if doctor_evolve_info:
            for idx, info_dic in enumerate(doctor_evolve_info, start=1):
                doctor_qus = info_dic["question1"]
                patient_answer = info_dic["answer1"]
                new_qus = info_dic["question2"]
                few_shot_example += f"例子{idx}:问题：{doctor_qus}病人回答：{patient_answer}新问题{new_qus}"
        else:
            few_shot_example = "无示例。"

        prompt = self.prompt_data["doctor_question_info"]
        info = self.prompt_data["resource"]
        prompt = prompt.format(self.office, self.main_complaint, self.summary, self.dialog_history, few_shot_example, info)

        if len(self.sub_doctor) > 0:
            for doctor in self.sub_doctor:
                if not doctor.dialog_turn > 5:
                    doctor_question = doctor.doctor_qus(
                        doctor.new_patient_answer, doctor.new_patient_score,
                        doctor.new_rel, doctor.new_faith, doctor.new_human
                    )
                    if doctor_question == "skip":
                        continue
                    patient_answer, pat_score, rel_, faith_, human_ = self.patient.patient_ans(doctor_question)
                    doctor.new_patient_answer = patient_answer
                    doctor.new_patient_score = pat_score
                    doctor.new_rel = rel_
                    doctor.new_faith = faith_
                    doctor.new_human = human_
                    doctor.make_summary()

                prompt += f"*****{doctor.office}*****"
                prompt += doctor.summary

        messages = [{"role": "user", "content": prompt}]
        response = llm_api(messages)

        qus = match_star(response, r"\*")
        if "NO" in qus:
            return "skip"
        if "conclusion" in qus:
            return "conclusion"

        self.dialog_history += f"patient answer: {answer}\n"
        office = "NO"

        self.store(
            self.last_qus, self.last_qus_category, answer, office,
            self.last_score, self.last_score_patient, self.last_rel, self.last_faith,
            self.last_human, self.last_doc_rel, self.last_doc_faith
        )

        category = match_star(response, r"#")

        useful_info = rag_patient(
            self.last_qus,
            self.patient.resource,
            size=120,
            overlap=40,
            top_k=2
        )

        score, doc_rel, doc_faith = 0, 0, 0
        if self.last_qus != "":
            score, doc_rel, doc_faith, _ = overall_assessment_doctor(self.last_qus, useful_info, answer)
            print(f"score: {score}")
            print(self.record)
            print(self.last_qus)

        # 修正：字符串比较使用 !=
        if score >= 3 and self.record[0] != "":
            if self.last_score >= 1:
                store_doctor_qa(
                    str(evolve_csv),
                    self.record + [self.last_qus] + [answer] + [useful_info]
                )

        self.record = [self.last_qus, answer, useful_info]
        self.last_score = score

        self.last_qus = qus
        self.last_score_patient = patient_score
        self.last_qus_category = category
        self.last_rel = rel
        self.last_faith = faith
        self.last_human = human
        self.last_doc_rel = doc_rel
        self.last_doc_faith = doc_faith

        self.dialog_history += "--- dialog ---\n"
        self.dialog_history += f"doctor question: {qus}\n"
        self.dialog_turn += 1
        if self.dialog_turn % self.summary_trun == 0:
            self.make_summary()

        return qus

    def conclusion(self):
        prompt = self.prompt_data["conclusion"]
        prompt = prompt.format(self.office, self.summary, self.dialog_history)

        if len(self.sub_doctor) > 0:
            for doctor in self.sub_doctor:
                prompt += f"*****{doctor.office}*****"
                prompt += doctor.summary

        messages = [{"role": "user", "content": prompt}]
        response = llm_api(messages)
        return response

    def store(self, qus, category, ans, office, doc_score, pat_score,
              last_rel, last_faith, last_human, last_doc_rel, last_doc_faith):
        if qus == "":
            return
        path = self.directory / "doctor_record" / f"{self.office}_{self.level}.csv"
        path.parent.mkdir(parents=True, exist_ok=True)

        if not path.is_file():
            with path.open("w", newline="", encoding="gbk") as file:
                writer = csv.writer(file)
                writer.writerow([
                    "question", "category", "answer", "recruit",
                    "doctor_score", "patient_score",
                    "patient_rel", "patient_faith", "patient_human",
                    "doctor_rel", "doctor_faith"
                ])

        with path.open("a", newline="", encoding="gbk") as file:
            writer = csv.writer(file)
            writer.writerow([
                qus.replace("\n", ""),
                category.replace("\n", ""),
                ans.replace("\n", ""),
                office.replace("\n", ""),
                str(doc_score), str(pat_score),
                str(last_rel), str(last_faith), str(last_human),
                str(last_doc_rel), str(last_doc_faith)
            ])

    def recruit(self):
        prompt = self.prompt_data["recruit"]
        prompt = prompt.format(self.office, self.main_complaint, self.summary, self.dialog_history, self.dialog_turn)
        messages = [{"role": "user", "content": prompt}]
        res = llm_api(messages)

        new_office = match_star(res, r"\#")
        if "NO" not in new_office:
            offices = re.split(r"[，,]", new_office)
            doctor_record_dir = self.directory / "doctor_record"
            doctor_record_dir.mkdir(parents=True, exist_ok=True)

            for office in offices:
                if check_files(doctor_record_dir, office):
                    print(f"存在 {office} 科室医生，不招募")
                    continue

                new_doctor = Doctor(
                    self.patient, office, main_complaint=self.main_complaint,
                    directory=self.directory, prompt_data=self.prompt_data, level=self.level + 1
                )

                doctor_question = new_doctor.doctor_qus(new_doctor.main_complaint, 0, 0, 0, 0)
                patient_answer, score, rel_, faith_, human_ = self.patient.patient_ans(doctor_question)
                new_doctor.new_patient_answer = patient_answer
                new_doctor.new_patient_score = score
                new_doctor.new_rel = rel_
                new_doctor.new_faith = faith_
                new_doctor.new_human = human_
                new_doctor.dialog_history += "--- dialog ---\n"
                new_doctor.dialog_history += f"doctor question: {doctor_question}\n"
                self.sub_doctor.append(new_doctor)
                print(f"招募 {office} 医生")
        return new_office

    def make_summary(self):
        prompt = self.prompt_data["summary"]
        prompt = prompt.format(self.office, self.summary)
        prompt += self.dialog_history
        messages = [{"role": "user", "content": prompt}]
        self.summary = llm_api(messages)
        self.dialog_history = ""

    def doctor_reflect(self):
        pass

    def doctor_chat(self):
        pass

    def doctor_crisis_answer(self, office, patient_crisis):
        auto = True
        if auto:
            resource = self.prompt_data["resource"]
            dq_path = self.directory / "doctor_question.txt"
            memory_stream_chat = dq_path.read_text(encoding="utf-8") if dq_path.exists() else ""

            prompt = self.prompt_data["doctor_crisis_answer"]
            prompt = prompt.format(office=office, chat=memory_stream_chat, crisis=patient_crisis, information=resource)
            messages = [{"role": "user", "content": prompt}]
            doctor_answer = llm_api(messages)
        else:
            doctor_answer = input("请输入回复：")
        return doctor_answer


def check_files(directory: Path, prefix: str) -> bool:
    directory.mkdir(parents=True, exist_ok=True)
    for filename in os.listdir(directory):
        if filename.startswith(prefix):
            return True
    return False
