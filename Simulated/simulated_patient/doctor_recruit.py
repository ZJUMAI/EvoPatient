import os
import re
from typing import List, Dict, Any, Optional

from Simulated.simulated_patient.api_call import llm_api


# =========== 环境变量读取（隐去隐私信息，不在代码中写密钥） ===========
# 要求外部环境已设置：
#   export OPENAI_API_KEY="your_key"
#   export BASE_URL="https://your-base.url"   # 可选
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("未设置环境变量 OPENAI_API_KEY，请在运行前 `export OPENAI_API_KEY=...`")

# 如果你的 llm_api 会读取 BASE_URL，也可以在这里校验（可选）
# BASE_URL = os.getenv("BASE_URL")


def match_star(context: str, symbol: str) -> str:
    """匹配 **...** 或 ##...## 样式的片段，返回去掉符号后的内容；未匹配返回 'NO'。"""
    # symbol 可能传入 "\*" 或 "\#"，需要正则转义
    sym = symbol.replace("\\", "")
    pat = f"{re.escape(sym)}{re.escape(sym)}(.*?){re.escape(sym)}{re.escape(sym)}"
    m = re.search(pat, context, flags=re.DOTALL)
    return re.sub(re.escape(sym), "", m.group(0)) if m else "NO"


def split_string(input_string: str) -> List[str]:
    """按中英文逗号切分；无逗号则返回单元素列表。"""
    if "," in input_string:
        return [s.strip() for s in input_string.split(",")]
    if "，" in input_string:
        return [s.strip() for s in input_string.split("，")]
    return [input_string.strip()]


class Recruit:
    """
    递归招募“医生”的控制器。
    - llm_api 由外部模块提供；本文件不接触任何明文密钥。
    - 默认把层级的 key 规范为 int，避免 str/int 混比对。
    """

    def __init__(
        self,
        parent_patient: Any,
        office: str = "",
        doctor: Optional[Any] = None,
        main_complaint: Optional[str] = None,
        directory: Optional[str] = None,
        prompt_data: Optional[Dict[str, Any]] = None,
    ):
        self.parent = parent_patient           # 病人对象
        self.patient = parent_patient          # 与上面保持一致命名，兼容下游调用
        self.prompt_data = prompt_data or {}
        self.office = office or ""
        self.doctor = doctor                   # 传入的 Doctor 类或工厂
        self.sub_doctor: List[Any] = []

        # 运行态上下文（避免引用未定义）
        self.main_complaint = main_complaint or ""
        self.summary = ""
        self.dialog_history = ""
        self.dialog_turn = 0
        self.directory = directory or "."

        # 层级 -> 科室列表；统一使用 int 作为层级键
        self.doctor_office: Dict[int, List[str]] = {1: [self.office]}
        self.available_office: List[str] = []

    def chat(self):
        pass

    def discussion(self, doctor_list: List[str]):
        pass

    def report(self):
        pass

    def bus(self):
        """按当前最大层级调度讨论（占位函数，逻辑可按需扩展）"""
        if not self.doctor_office:
            return
        max_layer = max(self.doctor_office.keys())
        sub_doctor_list = self.doctor_office.get(max_layer, [])
        self.discussion(sub_doctor_list)

    def star(self):
        pass

    def ring(self):
        pass

    def tree(self):
        pass

    def recruit(self, layer: int, office: str):
        """
        招募逻辑：
        - 通过 prompt 查询推荐的下级科室（以 ##...## 形式返回）。
        - 若返回非 "NO"，加入层级映射并继续递归；
        - 完成后实例化 doctor 并执行第一轮问诊。
        """
        while True:
            prompt_tpl = self.prompt_data.get("recruit", "")
            # 兼容早期使用的 format 顺序：office, main_complaint, summary, dialog_history, dialog_turn
            prompt = prompt_tpl.format(
                office, self.main_complaint, self.summary, self.dialog_history, self.dialog_turn
            )
            messages = [{"role": "user", "content": prompt}]
            res = llm_api(messages)

            # 从 ##...## 中解析科室字符串，可能包含多个，以逗号分隔
            parsed = match_star(res, r"\#")
            new_offices = split_string(parsed) if parsed != "NO" else []

            # 正常情况 new_offices 为 ["NO"] 或 ["内科","消化内科"] 等
            if not new_offices or (len(new_offices) == 1 and new_offices[0] == "NO"):
                break

            # 记录到层级结构
            self.doctor_office.setdefault(layer, [])
            for off in new_offices:
                if off and off not in self.doctor_office[layer]:
                    self.doctor_office[layer].append(off)

            # 递归探索下层
            for sub_off in new_offices:
                self.recruit(layer + 1, sub_off)

            # 跳出当前 while，避免无限循环；如需反复招募，可按需放开
            break

        # 实例化并进行第一轮问诊（需要外部传入 doctor 类/工厂）
        if not self.doctor:
            return

        new_office = office or (new_offices[0] if new_offices else self.office)
        new_doctor = self.doctor(
            self.patient,
            new_office,
            main_complaint=self.main_complaint,
            directory=self.directory,
            prompt_data=self.prompt_data,
        )

        # 第一轮对话：先让医生出题，再让病人回答
        doctor_question = new_doctor.doctor_qus(new_doctor.main_complaint, 0, 0, 0, 0)
        patient_answer = self.patient.patient_ans(doctor_question)
        # patient.patient_ans 可以返回 tuple 或 str，这里做个兼容
        if isinstance(patient_answer, (list, tuple)):
            new_doctor.new_patient_answer = patient_answer[0]
        else:
            new_doctor.new_patient_answer = patient_answer

        self.sub_doctor.append(new_doctor)
        print(f"招募 {new_office} 医生")
