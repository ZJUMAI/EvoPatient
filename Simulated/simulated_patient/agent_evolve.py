import os
import numpy as np
import csv
import json
from pathlib import Path
from Simulated.simulated_patient.api_call import llm_api
from openai import OpenAI

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
BASE_URL = os.getenv("BASE_URL")

if not OPENAI_API_KEY:
    raise RuntimeError("未设置环境变量 OPENAI_API_KEY")

client_kwargs = {"api_key": OPENAI_API_KEY}
if BASE_URL:
    client_kwargs["base_url"] = BASE_URL
client = OpenAI(**client_kwargs)


def write_to_csv(
    directory,
    embedding_res,
    question,
    rag_info,
    answer,
    requirements,
    write_header=False,
):
    directory = Path(directory)
    embedding_res_str = ",".join(map(str, embedding_res))
    data_to_write = [embedding_res_str, question, rag_info, answer, requirements]

    if directory.is_file() and not write_header:
        with directory.open("r", newline="", encoding="gbk") as file:
            reader = csv.reader(file)
            _ = next(reader, None)
            for row in reader:
                if row and row[0] == embedding_res_str:
                    return
    else:
        with directory.open("w", newline="", encoding="gbk") as file:
            writer = csv.writer(file)
            writer.writerow(["qus_embedding", "question", "rag_info", "answer", "requirements"])

    with directory.open("a", newline="", encoding="gbk") as file:
        writer = csv.writer(file)
        writer.writerow(data_to_write)


def write_csv(directory, qus_1, emb_1, emb_2, ans_1, rag_1, qus_2, ans_2, rag_2, write_header=False):
    directory = Path(directory)
    embedding_res_str1 = ",".join(map(str, emb_1))
    embedding_res_str2 = ",".join(map(str, emb_2))
    data_to_write = [qus_1, embedding_res_str1, rag_1, ans_1, embedding_res_str2, qus_2, ans_2, rag_2]

    if directory.is_file() and not write_header:
        with directory.open("r", newline="", encoding="gbk") as file:
            reader = csv.reader(file)
            _ = next(reader, None)
            for row in reader:
                if row and row[0] == embedding_res_str1:
                    return
    else:
        with directory.open("w", newline="", encoding="gbk") as file:
            writer = csv.writer(file)
            writer.writerow(
                ["question1", "qus_embedding", "rag_info1", "answer1", "qus2_embedding", "question2", "answer2", "rag_info2"]
            )

    with directory.open("a", newline="", encoding="gbk") as file:
        writer = csv.writer(file)
        writer.writerow(data_to_write)


def get_text_embedding(text: str):
    text = text or "None"
    return client.embeddings.create(
        input=text,
        model="text-embedding-ada-002",
    ).model_dump()["data"][0]["embedding"]


def read_qus_embedding_from_csv(directory):
    directory = Path(directory)
    qus_embedding_list = []
    with directory.open(newline="", encoding="gbk") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            v = row.get("qus_embedding")
            if v:
                qus_embedding_list.append(v)
    return qus_embedding_list


def read_qus_embedding_doctor(directory):
    directory = Path(directory)
    qus1_embedding_list, qus2_embedding_list = [], []
    with directory.open(newline="", encoding="gbk") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            v1 = row.get("qus_embedding")
            if v1:
                qus1_embedding_list.append(v1)
            v2 = row.get("qus2_embedding")
            if v2:
                qus2_embedding_list.append(v2)
    return qus1_embedding_list, qus2_embedding_list


def get_cosine_similarity(embeddingi, embeddingj):
    embeddingi = np.array(embeddingi)
    embeddingj = np.array(embeddingj)
    return embeddingi.dot(embeddingj) / (np.linalg.norm(embeddingi) * np.linalg.norm(embeddingj))


def quality_check(question, rag_info, answer):
    json_file_path = Path("Simulated/Prompt/prompt_data.json")  # 相对路径
    with json_file_path.open("r", encoding="utf-8") as file:
        data = json.load(file)
        prompt = "".join(data["quality_check_evolve"])
    prompt.format(question=question, infomation=rag_info, answer=answer)
    messages = [{"role": "user", "content": prompt}]
    return llm_api(messages)


def store_patient_qa(directory, question, rag_info, answer, requirements):
    embedding_res = get_text_embedding(question)
    qus_embedding_list = read_qus_embedding_from_csv(directory)
    evolve_flag = 1
    for stored_qus_embedding in qus_embedding_list:
        stored_qus_embedding = [float(item.strip()) for item in stored_qus_embedding.split(",")]
        if get_cosine_similarity(stored_qus_embedding, embedding_res) > 0.95:
            print("此条问答已有相似例子，取消进化。")
            evolve_flag = 0
            break
    if evolve_flag:
        write_to_csv(directory, embedding_res, question, rag_info, answer, requirements)


def store_doctor_qa(directory, record):
    qus_1, ans_1, rag_1, qus_2, ans_2, rag_2 = record
    qus_1_emb = get_text_embedding(qus_1)
    qus_2_emb = get_text_embedding(qus_2)
    qus1_embedding_list, qus2_embedding_list = read_qus_embedding_doctor(directory)
    evolve_flag = 1
    for stored_qus1_embedding, stored_qus2_embedding in zip(qus1_embedding_list, qus2_embedding_list):
        stored_qus1_embedding = [float(item.strip()) for item in stored_qus1_embedding.split(",")]
        stored_qus2_embedding = [float(item.strip()) for item in stored_qus2_embedding.split(",")]
        if get_cosine_similarity(stored_qus1_embedding, qus_1_emb) > 0.8 and get_cosine_similarity(
            stored_qus2_embedding, qus_2_emb
        ) > 0.8:
            print("此条问答已有相似例子，取消进化。")
            evolve_flag = 0
            break
    if evolve_flag:
        write_csv(directory, qus_1, qus_1_emb, qus_2_emb, ans_1, rag_1, qus_2, ans_2, rag_2)
        print(f"store {qus_1} + {qus_2}")


def get_most_related_qus(rank_dic):
    sorted_items = sorted(rank_dic.items(), key=lambda x: x[1], reverse=True)
    if len(sorted_items) > 2:
        return [item[0] for item in sorted_items[:2]]
    elif 0 < len(sorted_items) <= 2:
        return [item[0] for item in sorted_items[:1]]
    else:
        return []


def get_evolve_info(related_qus_list, directory):
    directory = Path(directory)
    matched_rows_data = []
    with directory.open(newline="", encoding="gbk") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            if row["qus_embedding"] in related_qus_list:
                matched_row = {header: row[header] for header in reader.fieldnames}
                matched_rows_data.append(matched_row)
    return matched_rows_data


def get_consistency(directory, qus_embedding):
    qus_embedding_list = read_qus_embedding_from_csv(directory)
    rank_dic = {}
    for stored_embedding_str in qus_embedding_list:
        stored_embedding_list = [float(item.strip()) for item in stored_embedding_str.split(",")]
        task_question_alignment = get_cosine_similarity(stored_embedding_list, qus_embedding)
        if task_question_alignment > 0.9:
            rank_dic[stored_embedding_str] = task_question_alignment
    return get_most_related_qus(rank_dic)


def get_consistency_doctor(directory, qus_embedding, ans_embedding):
    qus1_embedding_list, qus2_embedding_list = read_qus_embedding_doctor(directory)
    rank_dic = {}
    for qus_emb, ans_emb in zip(qus1_embedding_list, qus2_embedding_list):
        qus1_emb_list = [float(item.strip()) for item in qus_emb.split(",")]
        qus2_emb_list = [float(item.strip()) for item in ans_emb.split(",")]
        task_question_alignment = get_cosine_similarity(qus1_emb_list, qus_embedding)
        if task_question_alignment > 0.25:
            rank_dic[qus_emb] = task_question_alignment
    return get_most_related_qus(rank_dic)


def agent_evolving_patient(directory, question):
    qus_embedding = get_text_embedding(question)
    related_qus_list = get_consistency(directory, qus_embedding)
    if related_qus_list:
        return get_evolve_info(related_qus_list, directory)
    else:
        return {}


def agent_evolving_doctor(directory, record):
    qus_embedding = get_text_embedding(record[0])
    ans_embedding = get_text_embedding(record[1])
    related_qus_list = get_consistency_doctor(directory, qus_embedding, ans_embedding)
    if related_qus_list:
        return get_evolve_info(related_qus_list, directory)
    else:
        return {}
