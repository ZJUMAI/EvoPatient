import os
import json

# 设置文件夹路径
folder_path = './'

# 创建一个字典来存储所有文本内容
combined_data = {}

# 遍历文件夹中的所有文件
for filename in os.listdir(folder_path):
    if filename.endswith('.txt'):  # 确保处理的是txt文件
        file_path = os.path.join(folder_path, filename)
        with open(file_path, 'r', encoding='utf-8') as file:
            # 读取文件内容
            content = file.read()
            # 将文件名作为键，内容作为值添加到字典中
            combined_data[filename] = content

# 将字典转换为JSON格式并写入文件
with open('prompt_data.json', 'w', encoding='utf-8') as json_file:
    json.dump(combined_data, json_file, ensure_ascii=False, indent=4)

print("所有txt文件已合并到combined_data.json文件中。")