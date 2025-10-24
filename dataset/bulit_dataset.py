import json
from pathlib import Path

import pandas as pd


def load_sheets(excel_path: Path, sheets):
    """安全读取指定工作表（不存在的会跳过），返回 {sheet_name: DataFrame}。"""
    existing = {}
    for name in sheets:
        try:
            df = pd.read_excel(excel_path, sheet_name=name)
            existing[name] = df
        except Exception as e:
            print(f"⚠️ 跳过工作表 {name}: {e}")
    return existing


def normalize_df(df: pd.DataFrame) -> pd.DataFrame:
    """将 NaN 转换为 None，便于 JSON 序列化。"""
    return df.where(pd.notnull(df), None)


def deduplicate_records(records):
    """根据记录内容去重（无序健壮），返回去重后的列表。"""
    seen = set()
    unique = []
    for rec in records:
        key = json.dumps(rec, ensure_ascii=False, sort_keys=True)
        if key not in seen:
            seen.add(key)
            unique.append(rec)
    return unique


def build_patient_dict(excel_path: Path, sheets):
    """按 Patient-SN 聚合跨表数据，去重并移除 Patient-SN 字段。"""
    sheet_map = load_sheets(excel_path, sheets)
    patient_map = {}

    for sheet_name, df in sheet_map.items():
        df = normalize_df(df)
        if "Patient-SN" not in df.columns:
            print(f"⚠️ 工作表 {sheet_name} 缺少列 'Patient-SN'，已跳过。")
            continue

        for _, row in df.iterrows():
            row_dict = row.to_dict()
            patient_id = row_dict.get("Patient-SN")
            if patient_id is None:
                continue

            # 去掉 Patient-SN 字段，仅保留其它信息
            record = {k: v for k, v in row_dict.items() if k != "Patient-SN"}

            patient_map.setdefault(patient_id, []).append(record)

    # 对每位患者的记录去重
    for pid, records in patient_map.items():
        patient_map[pid] = deduplicate_records(records)

    return patient_map


def main():
    # 相对路径：Excel 输入与 JSON 输出
    excel_file = Path("../dataset/patient_text.xlsx")
    output_json = Path("patient_data.json")

    if not excel_file.exists():
        raise FileNotFoundError(f"未找到 Excel 文件：{excel_file.resolve()}")

    sheets = ["患者基本信息", "检查_MRI检查", "病程记录_首次病程", "病理_全部病理", "专科检查_专科检查"]

    patient_dict = build_patient_dict(excel_file, sheets)

    output_json.write_text(
        json.dumps(patient_dict, ensure_ascii=False, indent=4),
        encoding="utf-8",
    )
    print(f"✅ JSON 文件已生成：{output_json.resolve()}")


if __name__ == "__main__":
    main()