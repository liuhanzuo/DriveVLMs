import re
import argparse
import json
import numpy as np
import torch.nn as nn
import language_evaluation
from multiprocessing import Pool

import sys
sys.path.append(".")

from tools.evaluation import evaluation_suit

def preprocess_text(text):
    """
    预处理文本，处理中文和混合文本的问题
    1. 保留所有数字和英文
    2. 将中文字符替换为空格
    3. 确保文本非空
    """
    if not text:
        return "empty_response"
    
    # 保留所有数字、英文字母、标点符号，替换中文字符
    processed_text = re.sub(r'[^\x00-\x7F]+', ' ', text)
    
    # 如果处理后文本为空，返回一个占位符
    if not processed_text.strip():
        return "non_ascii_text"
    
    return processed_text

class CustomEvaluationSuit(evaluation_suit):
    """
    扩展评估套件，重写eval_language方法处理中文文本
    """
    def eval_language(self):
        """
        重写language评估方法，添加文本预处理
        """
        answer = [preprocess_text(a) for a in self.language["answer"]]
        GT = [preprocess_text(g) for g in self.language["GT"]]
        
        try:
            results_gen = self.language_eval.run_evaluation(answer, GT)
            results_gen_dict = {
                f"val/{k}": v for k, v in results_gen.items()
            }
            return results_gen_dict
        except Exception as e:
            print(f"语言评估出错: {e}")
            # 返回默认值
            return {
                "val/Bleu_1": 0.0,
                "val/Bleu_2": 0.0,
                "val/Bleu_3": 0.0,
                "val/Bleu_4": 0.0,
                "val/CIDEr": 0.0,
                "val/ROUGE_L": 0.0
            }

if __name__ == '__main__':
    # 获取参数
    parser = argparse.ArgumentParser(description='Evaluation for SFT')
    parser.add_argument('--src', type=str, default="data/DriveLM_nuScenes/split/combined_data.json", help='path to ground truth file')
    parser.add_argument('--tgt', type=str, default="data/DriveLM_nuScenes/refs/infer_results_qwen_vl_prompted.json", help='path to prediction file')
    args = parser.parse_args()
    
    # 读取预测结果文件
    with open(args.tgt, 'r') as f:
        pred_data = json.load(f)
    
    # 处理预测结果文件，将id和answer按对存储，并过滤无效答案
    pred_file = {}
    for item in pred_data:
        ids = item["id"]
        answers = item["answer"]
        for i, (id_str, answer) in enumerate(zip(ids, answers)):
            # 过滤掉空答案或以"!"开头的答案
            if answer and not answer.startswith("!"):
                pred_file[id_str] = {"id": id_str, "answer": [answer]}
    
    # 读取ground truth文件
    with open(args.src, 'r') as f:
        src_data = json.load(f)
    
    # 创建id到ground truth的映射
    gt_dict = {}
    for item in src_data:
        item_id = item["id"]
        # ground truth answer在conversations的第二个元素的value中
        if len(item["conversations"]) > 1:
            gt_answer = item["conversations"][1]["value"]
            gt_dict[item_id] = gt_answer
    
    # 初始化评估工具，使用自定义版本处理中文
    evaluation = CustomEvaluationSuit()
    
    # 检查匹配的ID数量
    matched_ids = set(pred_file.keys()) & set(gt_dict.keys())
    print(f"找到匹配的ID数量: {len(matched_ids)}/{len(pred_file)}")
    
    # 遍历每个匹配的ID直接进行评估
    for idx in matched_ids:
        predict = pred_file[idx]["answer"][0]
        GT = gt_dict[idx]
        
        # 对于所有问题设置tag为[0,2,3]，代表进行accuracy、language和match评估
        tag = [0, 2, 3]
        
        # 使用try-except处理可能的错误
        try:
            evaluation.forward(tag, predict, GT)
        except Exception as e:
            print(f"处理ID {idx}时出错: {e}")
            continue

    # 输出评估结果
    try:
        output = evaluation.evaluation()
        print("accuracy score: ", output["accuracy"])
        print("match score: ", output["match"])
        print("language score: ", output["language"])
        
        # 归一化到0-1并组合分数
        scores = []

        # language
        score = 0
        for idx, key in enumerate(output["language"].keys()):
            if idx < 4:
                score += output["language"][key] / 4. / 3.
            elif idx == 4:
                score += output["language"][key] / 3. 
            else:
                score += output["language"][key] / 10. / 3.

        scores.append(score)
        
        # match
        score = output["match"] / 100.
        scores.append(score)

        # accuracy
        score = output["accuracy"]
        scores.append(score)
        print(f"总分:{sum(scores)/len(scores)}")
    except Exception as e:
        print(f"评估过程出错: {e}")
