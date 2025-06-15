from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
from Simulated.simulated_patient.patient_agent import Patient
from Simulated.simulated_patient.vagueness import get_vague_patient_info
from simulateflow import read_prompt
import time
import os
import pandas as pd
import json
from datetime import datetime
import uuid
from final_evaluate import evaluate_doctor_performance
from conversation_eval import eval_conversations

app = Flask(__name__, static_url_path='', static_folder='static')
CORS(app)

# 存储会话状态
sessions = {}

# 确保logs目录存在
if not os.path.exists('logs'):
    os.makedirs('logs')


# 添加根路由来渲染index.html
@app.route('/')
def index():
    return render_template('main.html')


@app.route('/api/start_session', methods=['POST'])
def start_session():
    try:
        session_id = str(time.time())
        test_label = session_id

        department = request.json.get('department', '内科')  # 默认内科
        worker_id = request.json.get('workerId')
        case_index = request.json.get('caseIndex')

        # 验证工号
        try:
            worker_df = pd.read_excel('工号.xlsx')
            valid_worker_ids = worker_df['工号'].astype(str).tolist()
            if str(worker_id) not in valid_worker_ids:
                return jsonify({'error': '无效的工号'}), 400
        except Exception as e:
            print(f"工号验证失败: {str(e)}")
            return jsonify({'error': '工号验证失败'}), 500

        parent_folder = 'new_recruit_evolve'
        directory = os.path.join(parent_folder, test_label)
        if not os.path.exists(directory):
            os.makedirs(directory)
            os.makedirs(os.path.join(directory, "doctor_record"))

        # 获取病例
        excel_path = f'cases/departments/{department}.xlsx'
        if os.path.exists(excel_path):
            df = pd.read_excel(excel_path)
            total_cases = len(df)
            if total_cases == 0:
                return jsonify({'error': f'未找到{department}的病例'}), 404

            # 验证病例编号
            if case_index is None or case_index < 0 or case_index >= total_cases:
                return jsonify({'error': f'无效的病例编号，请输入0-{total_cases-1}之间的数字'}), 400
        else:
            return jsonify({'error': f'未找到{department}的病例文件'}), 404

        resource, vague_info = get_vague_patient_info(
            case_type=department,
            index=case_index
        )

        if not resource or not vague_info:
            return jsonify({'error': '无法获取病例信息'}), 400

        prompt_data = read_prompt()
        # init_info = generate_init_info(prompt_data['init_info'], resource)
        patient = Patient(vague_info, resource, directory, prompt_data)

        # 从Excel文件中读取主诉
        main_complaint = df.iloc[case_index].get('主诉', '')
        if not main_complaint:
            return jsonify({'error': '无法获取主诉信息'}), 400

        sessions[session_id] = {
            'patient': patient,
            'directory': directory,
            'turn_count': 0,
            'resource': resource,
            # 'init_info': init_info,
            'department': department,
            'case_index': case_index,
            'worker_id': worker_id,
            'chat_history': [],
            'main_complaint': main_complaint,
            'profile': patient.profile
        }

        return jsonify({
            'session_id': session_id,
            'department': department,
            'main_complaint': main_complaint,
            'resource': resource,
            # 'init_info': init_info,
            'case_index': case_index
        })

    except Exception as e:
        print(f"会话启动失败: {str(e)}")
        import traceback
        print(traceback.format_exc())
        return jsonify({'error': str(e)}), 500


@app.route('/api/chat', methods=['POST'])
def chat():
    session_id = request.json.get('session_id')
    doctor_question = request.json.get('question')
    is_chief_complaint = request.json.get('is_chief_complaint', False)

    if session_id not in sessions:
        return jsonify({'error': 'Invalid session'}), 400

    session = sessions[session_id]

    if is_chief_complaint:
        # 如果是主诉，直接使用初始的主诉内容
        patient_answer = session.get('main_complaint', '')
        # 记录对话历史
        session['chat_history'].append({
            'role': 'patient',
            'content': patient_answer,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'type': 'chief_complaint',
            'feedback': None,  # 为主诉添加feedback字段
            'message_id': str(uuid.uuid4())  # 为主诉也生成一个message_id
        })
        message_id = None
    else:
        session['turn_count'] += 1
        patient = session['patient']
        max_attempts = 3
        attempt = 0
        patient_answer = None

        while attempt < max_attempts:
            raw_answer = patient.patient_ans(doctor_question)
            # 尝试提取 JSON 部分
            start_marker = '{"answer":"'
            end_marker = '"}'
            start_index = raw_answer.find(start_marker)
            end_index = raw_answer.find(
                end_marker, start_index + len(start_marker))

            if start_index != -1 and end_index != -1:
                json_part = raw_answer[start_index:end_index + len(end_marker)]
                try:
                    patient_answer = json.loads(json_part)
                    patient_answer = patient_answer['answer']
                    break
                except (json.JSONDecodeError, KeyError, TypeError):
                    pass  # 如果解析失败，继续尝试

            attempt += 1
            print(f"尝试 {attempt} 次提取 answer 失败，重新生成回答。")

        if attempt == max_attempts:
            patient_answer = "我没听清，你再说一次？"
        # 生成消息ID
        message_id = str(uuid.uuid4())

        # 记录对话历史，将问题和回答组合在一起
        session['chat_history'].append({
            'turn': session['turn_count'],
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'qa_pair': {
                'question': {
                    'role': 'doctor',
                    'content': doctor_question
                },
                'answer': {
                    'role': 'patient',
                    'content': patient_answer,
                    'message_id': message_id,
                    'feedback': None  # 初始化反馈为空
                }
            }
        })
    return jsonify({
        'answer': patient_answer,
        'turn_count': session['turn_count'],
        'message_id': message_id
    })


def get_conversations(history):
    content = ''
    for i, item in enumerate(history):
        if i == 0:
            content += '病人到来时的主诉：' + item['content'] + '\n\n'
        else:
            question = item['qa_pair']['question']['content']
            answer = item['qa_pair']['answer']['content']
            content += '医生提问：' + question + '\n病人回答：' + answer + '\n\n'
    conversations = content
    return conversations


@app.route('/api/feedback', methods=['POST'])
def submit_feedback():
    try:
        data = request.json
        session_id = data.get('session_id')
        message_id = data.get('message_id')
        feedback_type = data.get('feedback_type')

        if session_id not in sessions:
            return jsonify({'error': 'Invalid session'}), 400

        session = sessions[session_id]

        # 更新对话历史中的反馈
        for message in session['chat_history']:
            # 检查是否是主诉消息
            if message.get('message_id') == message_id:
                message['feedback'] = feedback_type
                break
            # 检查是否是问答对中的回答
            if message.get('qa_pair') and message['qa_pair'].get('answer', {}).get('message_id') == message_id:
                message['qa_pair']['answer']['feedback'] = feedback_type
                break

        return jsonify({'status': 'success'})
    except Exception as e:
        print(f"提交反馈失败: {str(e)}")
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/api/cases/<department>')
def get_cases(department):
    try:
        excel_path = f'cases/departments/{department}.xlsx'
        if os.path.exists(excel_path):
            df = pd.read_excel(excel_path)
            # 将NaN值替换为空字符串
            df = df.fillna('')
            # 转换为字典列表，确保所有值都是字符串类型
            records = df.astype(str).to_dict('records')
            return jsonify(records)
        return jsonify([])
    except Exception as e:
        print(f"获取病例失败: {str(e)}")
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/api/case/<department>/<int:index>')
def get_case(department, index):
    try:
        excel_path = f'cases/departments/{department}.xlsx'
        if os.path.exists(excel_path):
            df = pd.read_excel(excel_path)
            if index < len(df):
                case_data = df.iloc[index].to_dict()
                # 确保返回诊断(Diagnosis)：字段
                if '诊断内容' not in case_data:
                    case_data['诊断内容'] = ''
                return jsonify(case_data)
        return jsonify({'status': 'error', 'message': '病例不存在'}), 404
    except Exception as e:
        print(f"获取病例失败: {str(e)}")
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/api/submit_diagnosis', methods=['POST'])
def submit_diagnosis():
    try:
        data = request.json
        case_type = data['caseType']
        case_index = data['caseIndex']
        user_diagnosis = data['diagnosis']
        session_id = data['sessionId']

        excel_path = f'cases/{case_type}/cases_{case_type}.xlsx'
        if os.path.exists(excel_path):
            df = pd.read_excel(excel_path)
            if case_index < len(df):
                case_data = df.iloc[case_index].to_dict()
                standard_diagnosis = case_data.get('诊断内容', '')
                case_number = case_data.get('病案号', '')

                # 获取会话信息
                session = sessions.get(session_id, {})
                chat_history = session.get('chat_history', [])
                resource = session.get('resource', '')

                # 创建日志数据
                log_data = {
                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'case_index': case_index,
                    'case_number': case_number,
                    'patient_info': resource,
                    'chat_history': chat_history,
                    'user_diagnosis': user_diagnosis,
                    'standard_diagnosis': standard_diagnosis,
                    'worker_id': session.get('worker_id', '')
                }

                # 保存日志文件
                timestamp = int(time.time())
                worker_id = session.get('worker_id', '')
                worker_log_dir = f'logs/{worker_id}'
                if not os.path.exists(worker_log_dir):
                    os.makedirs(worker_log_dir)
                log_filename = f'{worker_log_dir}/{timestamp}.json'
                with open(log_filename, 'w', encoding='utf-8') as f:
                    json.dump(log_data, f, ensure_ascii=False, indent=2)

                return jsonify({
                    'status': 'success',
                    'standardDiagnosis': standard_diagnosis,
                    'chat_history': chat_history
                })

        return jsonify({'status': 'error', 'message': '病例不存在'}), 404
    except Exception as e:
        print(f"获取诊断内容失败: {str(e)}")
        return jsonify({'status': 'error', 'message': str(e)}), 500


# 添加静态文件路由
@app.route('/figs/<path:filename>')
def serve_figure(filename):
    return send_from_directory('figs', filename)


@app.route('/api/submit_medical_record', methods=['POST'])
def submit_medical_record():
    try:
        data = request.json
        print("接收到的数据:", data)

        department = data['caseType']
        case_index = data['caseIndex']
        session_id = data['sessionId']

        excel_path = f'cases/departments/{department}.xlsx'
        print(f"尝试读取文件: {excel_path}")

        if os.path.exists(excel_path):
            df = pd.read_excel(excel_path)
            print(f"成功读取Excel文件，行数: {len(df)}")

            if case_index < len(df):
                case_data = df.iloc[case_index].to_dict()
                print("读取到的病例数据:", case_data)

                # 将nan值转换为空字符串
                case_data = {k: ('' if pd.isna(v) else v)
                             for k, v in case_data.items()}

                standard_diagnosis = case_data.get('诊断内容', '')
                case_number = case_data.get('病案号', '')

                # 获取会话信息
                session = sessions.get(session_id, {})
                if not session:
                    raise ValueError(f"未找到会话ID: {session_id}")

                chat_history = session.get('chat_history', [])
                resource = session.get('resource', '')
                conversations = get_conversations(chat_history)
                conversations_eval_report = eval_conversations(
                    conversations, resource)

                # 构建医生表单信息
                doctor_form = f"""
                主诉：{data.get('chiefComplaint', '')}
                现病史（History of Present Illness）：{data.get('presentIllness', '')}
                既往史（Past History）：{data.get('pastHistory', '')}
                成瘾药物（Drug Addiction）：{data.get('addictiveDrugs', '')}
                个人史（Personal History）：{data.get('personalHistory', '')}
                婚育史（Obstetrical History）：{data.get('marriageHistory', '')}
                家族史（Family History）：{data.get('familyHistory', '')}
                体格检查（Physical Examination）：{data.get('physicalExam', '')}
                辅助检查（Diagnostic Examination）：{data.get('auxiliaryExam', '')}
                诊断结果：{data.get('initialDiagnosis', '')}
                """

                # 进行医生表现评估
                evaluation_results = evaluate_doctor_performance(
                    resource, doctor_form)
                total_score = sum(int(item['score'])
                                  for item in evaluation_results)
                max_score = len(evaluation_results) * 3
                percentage = (total_score / max_score) * \
                    100 if max_score > 0 else 0

                evaluation_results.append(conversations_eval_report)

                # 创建评估结果对象
                evaluation = {
                    'results': evaluation_results,
                    'total_score': total_score,
                    'max_score': max_score,
                    'percentage': percentage
                }

                # 创建日志数据
                log_data = {
                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'case_type': department,
                    'case_index': case_index,
                    'case_number': case_number,
                    'patient_info': resource,
                    'chat_history': chat_history,
                    'medical_record': {
                        'chiefComplaint': data.get('chiefComplaint', ''),
                        'presentIllness': data.get('presentIllness', ''),
                        'pastHistory': data.get('pastHistory', ''),
                        'addictiveDrugs': data.get('addictiveDrugs', ''),
                        'personalHistory': data.get('personalHistory', ''),
                        'marriageHistory': data.get('marriageHistory', ''),
                        'familyHistory': data.get('familyHistory', ''),
                        'physicalExam': data.get('physicalExam', ''),
                        'auxiliaryExam': data.get('auxiliaryExam', ''),
                        'initialDiagnosis': data.get('initialDiagnosis', '')
                    },
                    'standard_diagnosis': standard_diagnosis,
                    'worker_id': session.get('worker_id', ''),
                    'profile': session.get('profile', ''),
                    'evaluation': evaluation
                }

                # 保存日志文件
                timestamp = int(time.time())
                worker_id = session.get('worker_id', '')
                worker_log_dir = f'logs/{worker_id}'
                if not os.path.exists(worker_log_dir):
                    os.makedirs(worker_log_dir)
                log_filename = f'{worker_log_dir}/{timestamp}.json'
                print(f"保存日志文件: {log_filename}")

                with open(log_filename, 'w', encoding='utf-8') as f:
                    json.dump(log_data, f, ensure_ascii=False, indent=2)

                # 提取所有标准字段，确保没有nan值
                standard_fields = {
                    'standardChiefComplaint': case_data.get('主诉', ''),
                    'standardPresentIllness': case_data.get('现病史（History of Present Illness）', ''),
                    'standardPastHistory': case_data.get('既往史（Past History）', ''),
                    'standardAddictiveDrugs': case_data.get('成瘾药物（Drug Addiction）', ''),
                    'standardPersonalHistory': case_data.get('个人史（Personal History）', ''),
                    'standardMarriageHistory': case_data.get('婚育史（Obstetrical History）', ''),
                    'standardFamilyHistory': case_data.get('家族史（Family History）', ''),
                    'standardPhysicalExam': case_data.get('体格检查（Physical Examination）', ''),
                    'standardAuxiliaryExam': case_data.get('辅助检查（Diagnostic Examination）', ''),
                    'standardNutritionScreening': case_data.get('营养风险筛查(Nutritional Assessment)', ''),
                    'standardFunctionAssessment': case_data.get('功能评估:(Function  Accessment)', ''),
                    'standardThrombosisAssessment': case_data.get('血栓风险评估（Venous thromboembolism Assessment）', '')
                }

                return jsonify({
                    'status': 'success',
                    'standardDiagnosis': standard_diagnosis,
                    'standardFields': standard_fields,
                    'chat_history': chat_history,
                    'evaluation': evaluation
                })

        return jsonify({'status': 'error', 'message': '病例不存在'}), 404
    except Exception as e:
        print(f"提交病历记录失败: {str(e)}")
        import traceback
        print("详细错误信息:")
        print(traceback.format_exc())
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/api/history/<worker_id>')
def get_history(worker_id):
    try:
        worker_log_dir = f'logs/{worker_id}'
        if not os.path.exists(worker_log_dir):
            return jsonify([])

        history_files = []
        for filename in os.listdir(worker_log_dir):
            if filename.endswith('.json'):
                file_path = os.path.join(worker_log_dir, filename)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        file_data = json.load(f)

                    # 文件名转为时间戳
                    timestamp = int(filename.split('.')[0])

                    history_files.append({
                        'filename': filename,
                        'timestamp': timestamp,
                        'data': {
                            'case_type': file_data.get('case_type'),
                            'case_index': file_data.get('case_index')
                        }
                    })
                except Exception as e:
                    print(f"读取历史记录文件失败: {str(e)}")

        # 按时间戳排序，最新的在前
        history_files.sort(key=lambda x: x['timestamp'], reverse=True)

        return jsonify(history_files)
    except Exception as e:
        print(f"获取历史记录失败: {str(e)}")
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/api/history/detail/<filename>')
def get_history_detail(filename):
    try:
        # 获取工号
        parts = filename.split('/')
        if len(parts) == 2:
            worker_id = parts[0]
            filename = parts[1]
        else:
            # 从查询参数中获取工号
            worker_id = request.args.get('worker_id')
            if not worker_id:
                return jsonify({'status': 'error', 'message': '未提供工号'}), 400

        file_path = f'logs/{worker_id}/{filename}'
        if not os.path.exists(file_path):
            return jsonify({'status': 'error', 'message': '文件不存在'}), 404

        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        return jsonify({
            'case_type': data.get('case_type'),
            'case_index': data.get('case_index'),
            'standard_diagnosis': data.get('standard_diagnosis'),
            'chat_history': data.get('chat_history'),
            'main_complaint': data.get('main_complaint', ''),
            'medical_record': data.get('medical_record', {}),
            'profile': data.get('profile', '')  # 添加profile信息
        })
    except Exception as e:
        print(f"获取历史记录详情失败: {str(e)}")
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/api/departments')
def get_departments():
    folder = 'cases/departments'
    departments = []
    for filename in os.listdir(folder):
        if filename.endswith('.xlsx'):
            departments.append(os.path.splitext(filename)[0])
    return jsonify(departments)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=7513, debug=False)
