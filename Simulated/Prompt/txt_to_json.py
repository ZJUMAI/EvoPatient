import os
import json
from pathlib import Path

# 设置文件夹路径（当前目录）
folder_path = Path("./")

# 输出文件路径
output_file = folder_path / "prompt_data.json"

combined_data = {}

# 遍历文件夹中的所有 txt 文件
for txt_file in folder_path.glob("*.txt"):
    if txt_file.name.startswith("."):  # 忽略隐藏文件
        continue

    with txt_file.open("r", encoding="utf-8") as f:
        content = f.read()
        combined_data[txt_file.name] = content
        print(f"✅ 已读取: {txt_file.name}")

# 将结果写入 JSON 文件
with output_file.open("w", encoding="utf-8") as f:
    json.dump(combined_data, f, ensure_ascii=False, indent=4)

print(f"\n所有 txt 文件已合并到 {output_file.name}")
