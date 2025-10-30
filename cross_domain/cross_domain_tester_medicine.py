import json
import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse
from sklearn.metrics import roc_auc_score

# 定义 MediumContrastiveEncoder，与训练时使用的结构保持一致
class MediumContrastiveEncoder(nn.Module):
    def __init__(self, input_dim=151936, hidden_dim1=37984, hidden_dim2=9496, hidden_dim3=37984, hidden_dim4=151936, dropout=0.0):
        super(MediumContrastiveEncoder, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 2048),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(2048, 792),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(792, 2048),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(2048, input_dim),
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 将除最后一维以外的维度展平，经过全连接网络，再恢复原始形状
        orig_shape = x.shape  # e.g. (B, T, V)
        x_flat = x.view(-1, orig_shape[-1])   # [B*T, V]
        out_flat = self.network(x_flat)       # [B*T, V]
        return out_flat.view(*orig_shape)     # [..., T, V]

# 计算困惑度（PPL）
def perplexity(encoding, logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    """
    计算每个样本的困惑度（perplexity），通过交叉熵实现:
    输入:
      - encoding.input_ids: [B, T]
      - encoding.attention_mask: [B, T]
      - logits: 模型的输出logits，形状 [B, T, V]
    输出:
      - ppl: [B] 张量
    """
    device = logits.device
    labels = encoding.input_ids.to(device)
    mask   = encoding.attention_mask.to(device).float()
    # 对齐 token (下一个词预测)
    shifted_logits = logits[..., :-1, :] / temperature        # [B, T-1, V]
    shifted_labels = labels[..., 1:]                          # [B, T-1]
    shifted_mask   = mask[..., 1:]                            # [B, T-1]
    # 逐 token 交叉熵损失
    ce = F.cross_entropy(shifted_logits.transpose(1, 2), shifted_labels, reduction="none")  # [B, T-1]
    # 只计算有效token位置的平均交叉熵
    total_ce = (ce * shifted_mask).sum(dim=1)
    valid_tokens = shifted_mask.sum(dim=1).clamp(min=1e-6)
    avg_ce = total_ce / valid_tokens  # 平均交叉熵
    return avg_ce  # 困惑度可用 e^(avg_ce) 表示，但比较时不必指数化

# 计算交叉熵 (Cross-Entropy) 用于 xPPL
def cross_entropy(p_logits: torch.Tensor, q_logits: torch.Tensor, attention_mask: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    """
    计算分布 p 相对于分布 q 的交叉熵，按每个样本求平均:
    - p_logits: 观察者模型经过编码器的 logits [B, T, V]
    - q_logits: 执行者模型经过编码器的 logits [B, T, V]
    - attention_mask: [B, T] 注意力掩码
    输出:
      - xppl: [B] 张量 (p 对 q 的平均交叉熵)
    """
    # 计算概率分布
    p_logit_scaled = p_logits / temperature
    q_logit_scaled = q_logits / temperature
    p_prob = F.softmax(p_logit_scaled, dim=-1)    # [B, T, V] 观察者概率
    q_log_prob = F.log_softmax(q_logit_scaled, dim=-1)  # [B, T, V] 执行者对数概率
    
    # 计算逐位置的交叉熵: H(p, q) = - sum_i p(i) log q(i)
    ce = -(p_prob * q_log_prob).sum(dim=-1)       # [B, T]
    mask = attention_mask.to(q_logits.device).float()
    # 平均交叉熵（忽略填充部分）
    total_ce = (ce * mask).sum(dim=1)
    valid_tokens = mask.sum(dim=1).clamp(min=1e-6)
    avg_ce = total_ce / valid_tokens
    return avg_ce

# 同时获取观察者和执行者模型的logits
@torch.no_grad()
def get_logits_pair(encodings, observer_model, performer_model, obs_device, perf_device):
    """
    将相同的输入 encodings 分别送入观察者模型和执行者模型，返回二者的 logits。
    encodings: BatchEncoding，包含 input_ids 和 attention_mask (在CPU上或单GPU上构建，再分别复制到对应设备)。
    """
    # 将输入复制到对应设备
    input_ids = encodings["input_ids"]
    attention_mask = encodings["attention_mask"]
    enc_obs = {"input_ids": input_ids.to(obs_device), "attention_mask": attention_mask.to(obs_device)}
    enc_perf = {"input_ids": input_ids.to(perf_device), "attention_mask": attention_mask.to(perf_device)}
    # 前向传播得到 logits
    obs_logits = observer_model(**enc_obs).logits  # [B, T, V]
    perf_logits = performer_model(**enc_perf).logits  # [B, T, V]
    return obs_logits, perf_logits

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="二分类测试脚本：判断文本是Human还是ChatGPT")
    parser.add_argument("-j", "--jsonl_path", type=str, required=True, help="输入的JSONL数据集路径（每行一个JSON对象）")
    parser.add_argument("-o", "--output_path", type=str, default="./results/scores_new_medicine.xlsx", help="结果保存的Excel路径（可选）")
    parser.add_argument("--scatter_path", type=str, default="./results/score_scatter_medicine.png", help="散点图保存路径")
    parser.add_argument("--sources", type=str, nargs="+", default=["medicine"], help="筛选的来源域列表，如 law 等。如果不提供则使用全部数据")
    parser.add_argument("--sample_size", type=int, default=500, help="测试样本总数（均分正负类）")
    parser.add_argument("--hf_token", type=str, default=None, help="访问私有模型所需的 HuggingFace token，没有可不填")
    parser.add_argument("--ckpt_path", type=str, default="../checkpoints/shared_encoder_adapter_jsonl_medicine.pth", help="编码器权重文件路径")
    args = parser.parse_args()
    
    data_path = args.jsonl_path
    output_xlsx = args.output_path
    scatter_path = args.scatter_path
    source_filters = args.sources if args.sources else []
    sample_size = args.sample_size
    hf_token = args.hf_token or os.environ.get("HF_TOKEN", None)
    ckpt_path = args.ckpt_path

    # 1. 数据读取与筛选
    data = []
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            # 如果设置了筛选来源，则跳过不符合的记录
            if source_filters:
                if record.get("source") not in source_filters:
                    continue
            # 提取人工答案和ChatGPT答案
            human_answers = record.get("human_answers", [])
            chatgpt_answers = record.get("chatgpt_answers", [])
            for ans in human_answers:
                data.append((ans, 0))  # 标签0表示Human
            for ans in chatgpt_answers:
                data.append((ans, 1))  # 标签1表示ChatGPT（机器）
    # 如果没有筛选source_filters，则data包含整个数据集所有答案
    
    # 打乱并抽样 sample_size 个样本（各类各半）
    if sample_size and sample_size < len(data):
        # 尽量均衡抽样正负类各一半
        human_samples = [x for x in data if x[1] == 0]
        machine_samples = [x for x in data if x[1] == 1]
        # 每类各取一半
        half = sample_size // 2
        if len(human_samples) < half or len(machine_samples) < half:
            # 若某类不足，退而求其次用全部
            half = min(len(human_samples), len(machine_samples))
        # 随机抽样
        random.seed(42)
        human_pick = random.sample(human_samples, half)
        machine_pick = random.sample(machine_samples, half)
        data_samples = human_pick + machine_pick
        random.shuffle(data_samples)
    else:
        data_samples = data  # 数据量小于要求或未指定则用全部
    
    texts = [item[0] for item in data_samples]
    true_labels = [item[1] for item in data_samples]
    # 输出样本统计
    num_human = sum(1 for label in true_labels if label == 0)
    num_machine = sum(1 for label in true_labels if label == 1)
    print(f"Total samples: {len(texts)} (Human: {num_human}, Machine: {num_machine})")
    
    # 2. 加载 tokenizer 和 模型
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen-7B", use_fast=True, trust_remote_code=True, pad_token="<|endoftext|>")
    # 设置设备：如果有2块GPU，分别加载；否则都在同一设备
    obs_device = "cuda:0" if torch.cuda.is_available() else "cpu"
    perf_device = "cuda:1" if (torch.cuda.device_count() > 1) else obs_device
    use_bf16 = torch.cuda.is_available()  # 使用GPU则尝试bfloat16
    dtype = torch.bfloat16 if use_bf16 else torch.float32
    
    # 加载观察者模型 (基础模型) 和 执行者模型 (Chat模型)
    observer_model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen-7B",
        device_map={"": obs_device},
        trust_remote_code=True,
        torch_dtype=dtype,
        token=hf_token
    )
    performer_model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen-7B-Chat",
        device_map={"": perf_device},
        trust_remote_code=True,
        torch_dtype=dtype,
        token=hf_token
    )
    observer_model.eval()
    performer_model.eval()
    for p in observer_model.parameters():
        p.requires_grad = False
    for p in performer_model.parameters():
        p.requires_grad = False
    
    # 加载共享编码器权重，并在各设备初始化
    ob_encoder = MediumContrastiveEncoder().to(obs_device).to(dtype)
    pe_encoder = MediumContrastiveEncoder().to(perf_device).to(dtype)
    # 加载权重（假定同一权重适用于两端）
    ob_state = torch.load(ckpt_path, map_location=obs_device)
    pe_state = torch.load(ckpt_path, map_location=perf_device)
    ob_encoder.load_state_dict(ob_state)
    pe_encoder.load_state_dict(pe_state)
    ob_encoder.eval()
    pe_encoder.eval()
    
    # 3. 遍历样本，计算分数
    scores = []
    ppl_values = []
    xppl_values = []
    for text in texts:
        # 将单个文本编码成 BatchEncoding
        encodings = tokenizer([text], return_tensors="pt", padding=False, truncation=True, max_length=512)
        # 获取两模型的 logits
        obs_logits, perf_logits = get_logits_pair(encodings, observer_model, performer_model, obs_device, perf_device)
        # 将 logits 经过编码器适配
        obs_enc_logits = ob_encoder(obs_logits)  # 在 obs_device 上
        perf_enc_logits = pe_encoder(perf_logits)  # 在 perf_device 上
        # 确保两者在同一设备上计算交叉熵（移动到执行者设备上计算以节省显存传输）
        obs_enc_logits = obs_enc_logits.to(perf_device)
        # 计算困惑度和交叉熵
        ppl = perplexity(encodings, perf_enc_logits.to(perf_device))  # 在执行者设备上计算 PPL
        xppl = cross_entropy(obs_enc_logits, perf_enc_logits, encodings["attention_mask"], temperature=1.0)
        # 提取数值并存储
        ppl_val = ppl.cpu().item()
        xppl_val = xppl.cpu().item()
        score_val = ppl_val / (xppl_val if xppl_val != 0 else 1e-6)
        scores.append(score_val)
        ppl_values.append(ppl_val)
        xppl_values.append(xppl_val)
    # 将原始分数保存到Excel（可选）
    df_scores = pd.DataFrame({"score": scores, "label": true_labels})
    df_scores.to_excel(output_xlsx, index=False)
    
    # 4. 确定最佳阈值并评估性能
    # 二分类阈值搜索（正类设为机器文本 label=1）
    best_thr = None
    best_f1 = -1.0
    best_metrics = {}
    # 考虑阈值在分数值之间的中点
    sorted_scores = sorted(scores)
    thr_candidates = []
    if sorted_scores:
        # 在极值之外加入两个端点阈值
        thr_candidates.append(sorted_scores[0] - 1e-6)
        for i in range(len(sorted_scores) - 1):
            thr_mid = (sorted_scores[i] + sorted_scores[i+1]) / 2.0
            thr_candidates.append(thr_mid)
        thr_candidates.append(sorted_scores[-1] + 1e-6)
    else:
        thr_candidates = [0.5]  # 如果没有样本，默认0.5（理论上不会无样本）
    # 遍历所有候选阈值，计算F1
    for thr in thr_candidates:
        # 预测：分数 <= 阈值 判定为机器(1)，否则人工(0)
        pred_labels = [1 if s >= thr else 0 for s in scores]
        
        # 计算混淆矩阵要素
        TP = sum(1 for p, t in zip(pred_labels, true_labels) if p == 1 and t == 1)  # 预测为机器，实际是机器
        FP = sum(1 for p, t in zip(pred_labels, true_labels) if p == 1 and t == 0)  # 预测为机器，实际是人工
        FN = sum(1 for p, t in zip(pred_labels, true_labels) if p == 0 and t == 1)  # 预测为人工，实际是机器
        TN = sum(1 for p, t in zip(pred_labels, true_labels) if p == 0 and t == 0)  # 预测为人工，实际是人工

        # 精确率
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
        # 召回率
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
        # F1-score
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        # 准确率
        accuracy = (TP + TN) / len(pred_labels) if len(pred_labels) > 0 else 0.0

        if f1 > best_f1:
            best_f1 = f1
            best_thr = thr
            best_metrics = {
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "accuracy": accuracy,
                "TP": TP, "FP": FP, "FN": FN, "TN": TN
            }

    # 打印最佳阈值和对应的性能指标
    print(f"Best threshold = {best_thr:.4f}")
    print(f"Accuracy = {best_metrics['accuracy']*100:.2f}%")
    print(f"Precision (machine) = {best_metrics['precision']*100:.2f}%")
    print(f"Recall (machine) = {best_metrics['recall']*100:.2f}%")
    print(f"F1-score (machine) = {best_metrics['f1']*100:.2f}%")
    
    # 5. 绘制散点图 (x轴为xPPL, y轴为PPL)，不同类别用不同颜色表示
    machine_x = [xp for xp, lbl in zip(xppl_values, true_labels) if lbl == 1]  # ChatGPT文本
    machine_y = [pp for pp, lbl in zip(ppl_values, true_labels) if lbl == 1]
    human_x = [xp for xp, lbl in zip(xppl_values, true_labels) if lbl == 0]    # 人类文本
    human_y = [pp for pp, lbl in zip(ppl_values, true_labels) if lbl == 0]
    plt.figure(figsize=(6,6))
    plt.scatter(range(len(scores)), scores, c=['r' if lbl == 1 else 'b' for lbl in true_labels], alpha=0.7)
    # 绘制阈值直线: ppl = best_thr * xPPL
    if best_thr is not None:
        max_x = max(xppl_values) if xppl_values else 0
        x_line = np.linspace(0, max_x, 100)
        y_line = best_thr * x_line
        plt.plot(x_line, y_line, 'g--', label=f'Threshold = {best_thr:.2f}')
    plt.xlabel("Sample Index")
    plt.ylabel("Score")
    plt.title("Human vs ChatGPT Text Distribution")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(scatter_path)
    print(f"Scatter plot saved to {scatter_path}")





    # 新增功能 1: 计算 AUROC
try:
    roc_auc = roc_auc_score(true_labels, scores)
    print(f"AUROC = {roc_auc:.4f}")
except Exception as e:
    print(f"AUROC calculation failed: {e}")

# 新增功能 2: 固定阈值下的 F1 分数
given_thr = 1.708021  # 你可以修改为其他想要的硬编码阈值
pred_labels_fixed = [1 if s >= given_thr else 0 for s in scores]

TP = sum(1 for p, t in zip(pred_labels_fixed, true_labels) if p == 1 and t == 1)
FP = sum(1 for p, t in zip(pred_labels_fixed, true_labels) if p == 1 and t == 0)
FN = sum(1 for p, t in zip(pred_labels_fixed, true_labels) if p == 0 and t == 1)
TN = sum(1 for p, t in zip(pred_labels_fixed, true_labels) if p == 0 and t == 0)

precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
accuracy = (TP + TN) / len(pred_labels_fixed) if len(pred_labels_fixed) > 0 else 0.0

print(f"Fixed threshold ({given_thr}): Precision = {precision*100:.2f}%, Recall = {recall*100:.2f}%, F1 = {f1*100:.2f}%, Accuracy = {accuracy*100:.2f}%")
