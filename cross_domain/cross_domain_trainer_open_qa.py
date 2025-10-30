from typing import Union, List, Dict, Tuple
import os
import json
import random
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import Dataset
from torch.amp import GradScaler, autocast
from transformers import AutoTokenizer, AutoModelForCausalLM, BatchEncoding
from contextlib import nullcontext
import math

import torch.multiprocessing as mp
mp.set_start_method('spawn', force=True)


# ---------------------------
# 配置
# ---------------------------
# 路径
JSONL_PATH = '../datasets/HC3_zh/all.jsonl'  # 改为你的 jsonl 路径
CHECKPOINT_DIR = '../checkpoints'

# 训练/评估来源可配
ALL_SOURCES = ["open_qa", "baike", "nlpcc_dbqa", "medicine", "finance", "psychology", "law"]
train_sources: List[str] = ["baike", "nlpcc_dbqa", "medicine", "finance", "psychology", "law"]
eval_sources:  List[str] = ["open_qa"]  # 参与测试的 source 子集

PER_SOURCE_LIMIT = 500  # 每个 source 选前 K 条
USE_RANDOM_POS_INSTEAD = False  # True 则将 pos 改为“剩余样本随机 human”

# 其他
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# ---------------------------
# 分词器
# ---------------------------
def get_tokenizer(model_name_or_path: str,
                  use_fast: bool = True,
                  trust_remote_code: bool = True) -> AutoTokenizer:
    tok = AutoTokenizer.from_pretrained(
        model_name_or_path,
        use_fast=use_fast,
        trust_remote_code=trust_remote_code,
        pad_token="<|endoftext|>",
    )
    return tok


# ---------------------------
# JSONL -> 三元组构建
# ---------------------------
DOMAIN_MAP = {
    "open_qa": 0,
    "baike": 1,
    "nlpcc_dbqa": 2,
    "medicine": 3,
    "finance": 4,
    "psychology": 5,
    "law": 6,
}

def _to_list(val):
    if val is None:
        return []
    if isinstance(val, list):
        return val
    # 可能是单字符串
    return [str(val)]

def _clean(s: str) -> str:
    if not isinstance(s, str):
        return ""
    return s.replace('\r', ' ').replace('\n', ' ').strip()

def build_triplets_from_jsonl(
    jsonl_path: str,
    sources: List[str],
    per_source_limit: int = 500,
    use_random_pos_instead: bool = False
) -> Tuple[List[str], List[str], List[str], List[int]]:
    """
    从 JSONL 读取并构建三元组（anchor/pos/neg）与 domain_label 列表。
    """
    # 先收集所有记录，按 source 分桶，并建立 human 池以备随机抽样
    buckets: Dict[str, List[Dict]] = {s: [] for s in sources}
    human_pool: List[str] = []

    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue

            src = obj.get("source", "").strip()
            if src not in sources:
                continue

            q = _clean(str(obj.get("question", "")))
            if not q:
                continue

            humans = [_clean(x) for x in _to_list(obj.get("human_answers")) if str(x).strip()]
            gpts   = [_clean(x) for x in _to_list(obj.get("chatgpt_answers")) if str(x).strip()]

            if not humans or not gpts:
                continue

            rec = {
                "question": q,
                "human": humans[0],
                "gpt": gpts[0],
                "source": src
            }
            buckets[src].append(rec)
            human_pool.append(rec["human"])

    # 组装三元组
    anchors, poss, negs, domain_labels = [], [], [], []
    for src in sources:
        cnt = 0
        for rec in buckets.get(src, []):
            if cnt >= per_source_limit:
                break
            anchor = rec["question"]
            pos_same = rec["human"]
            neg = rec["gpt"]

            if use_random_pos_instead and len(human_pool) > 1:
                # 从其余样本的人类答案中随机取一个作为正例
                # 确保不等于当前 pos_same（尽量）
                while True:
                    pos_rand = random.choice(human_pool)
                    if pos_rand != pos_same or len(human_pool) == 1:
                        break
                pos = pos_rand
            else:
                pos = pos_same

            anchors.append(anchor)
            poss.append(pos)
            negs.append(neg)
            domain_labels.append(DOMAIN_MAP.get(src, 0))
            cnt += 1

    return anchors, poss, negs, domain_labels


# ---------------------------
# 数据集：三元组 + 域标签
# ---------------------------
class TokenizedTripletDataset(Dataset):
    """
    读取三元组：anchor/positive/negative，并包含 domain_label。
    """
    def __init__(self,
                 anchors: List[str],
                 positives: List[str],
                 negatives: List[str],
                 domain_labels: List[int],
                 model_name_or_path: str,
                 max_length: int = 512,
                 device: Union[str, torch.device] = None):
        assert len(anchors) == len(positives) == len(negatives) == len(domain_labels)
        self.anchor_texts   = anchors
        self.positive_texts = positives
        self.negative_texts = negatives
        self.domain_label   = domain_labels

        self.tokenizer = get_tokenizer(model_name_or_path)
        self.max_length = max_length
        self.device = torch.device(device) if device is not None else None

    def __len__(self):
        return len(self.anchor_texts)

    def _tokenize(self, texts: List[str]) -> Dict[str, torch.Tensor]:
        enc = self.tokenizer(
            texts,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
            return_attention_mask=True,
            return_token_type_ids=False
        )
        if self.device is not None:
            enc = {k: v.to(self.device) for k, v in enc.items()}
        return enc

    def __getitem__(self, idx):
        anchor   = self.anchor_texts[idx]
        positive = self.positive_texts[idx]
        negative = self.negative_texts[idx]
        domain_label = self.domain_label[idx]

        enc = self._tokenize([anchor, positive, negative])

        return {
            'anchor_input_ids':        enc['input_ids'][0],
            'anchor_attention_mask':   enc['attention_mask'][0],
            'positive_input_ids':      enc['input_ids'][1],
            'positive_attention_mask': enc['attention_mask'][1],
            'negative_input_ids':      enc['input_ids'][2],
            'negative_attention_mask': enc['attention_mask'][2],
            'domain_label': torch.tensor(domain_label, dtype=torch.long)
        }


# ---------------------------
# Qwen logits 计算
# ---------------------------
@torch.inference_mode()
def get_logits_pair(encodings: BatchEncoding,
                    observer_model,
                    performer_model,
                    observer_device: Union[str, torch.device],
                    performer_device: Union[str, torch.device]) -> Tuple[torch.Tensor, torch.Tensor]:
    if isinstance(encodings, dict):
        encodings = BatchEncoding(encodings)
    ids_cpu  = encodings.input_ids.cpu()
    mask_cpu = encodings.attention_mask.cpu()

    enc_obs = BatchEncoding({
        'input_ids': ids_cpu.to(observer_device),
        'attention_mask': mask_cpu.to(observer_device)
    })
    enc_perf = BatchEncoding({
        'input_ids': ids_cpu.to(performer_device),
        'attention_mask': mask_cpu.to(performer_device)
    })

    obs_logits  = observer_model(**enc_obs).logits
    perf_logits = performer_model(**enc_perf).logits
    return obs_logits, perf_logits


@torch.inference_mode()
def compute_full_logits(dataset: TokenizedTripletDataset,
                        observer_model,
                        performer_model,
                        observer_device,
                        performer_device,
                        batch_size: int = 1) -> Dict[str, torch.Tensor]:
    """
    已移除：不再批量预计算所有样本的logits以节省内存。
    """
    raise NotImplementedError("Full logits precomputation removed in online computation update.")


# ---------------------------
# 评分函数（沿用 trainer.py）
# ---------------------------
def perplexity(encoding: BatchEncoding,
               logits: torch.Tensor,
               median: bool = False,
               temperature: float = 1.0) -> torch.Tensor:
    shifted_logits = logits[..., :-1, :] / temperature
    shifted_labels = encoding.input_ids[..., 1:]
    shifted_mask   = encoding.attention_mask[..., 1:].float()

    ce = F.cross_entropy(
        shifted_logits.transpose(1, 2),
        shifted_labels,
        reduction='none'
    )
    ce = ce * shifted_mask
    denom = shifted_mask.sum(dim=1).clamp(min=1e-6)
    val = ce.sum(dim=1) / denom
    if median:
        ce_masked = ce.masked_fill(shifted_mask == 0, float('nan'))
        val = torch.nanmedian(ce_masked, dim=1).values
    return val

def entropy(p_logits: torch.Tensor,
            q_logits: torch.Tensor,
            encoding: BatchEncoding,
            pad_token_id: int,
            median: bool = False,
            sample_p: bool = False,
            temperature: float = 1.0) -> torch.Tensor:
    B, T, V = p_logits.shape
    p = (p_logits / temperature).float()
    q = (q_logits / temperature).float()

    p = p - p.logsumexp(dim=-1, keepdim=True)
    q_logp = q - q.logsumexp(dim=-1, keepdim=True)

    p_proba = p.exp()
    mask = encoding.attention_mask.float()

    if sample_p:
        idx = torch.multinomial(p_proba.view(B*T, V), num_samples=1).view(B, T)
        ce = -q_logp.gather(dim=-1, index=idx.unsqueeze(-1)).squeeze(-1)
    else:
        ce = -(p_proba * q_logp).sum(dim=-1)

    if median:
        ce = ce.masked_fill(mask == 0, float('nan'))
        return torch.nanmedian(ce, dim=1).values
    else:
        ce = ce * mask
        return ce.sum(dim=1) / mask.sum(dim=1).clamp(min=1e-6)

# ---------------------------
# GRL + Domain Classifier（沿用 trainer.py）
# ---------------------------
class GradientReverseFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambda_adv):
        ctx.lambda_adv = lambda_adv
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambda_adv * grad_output, None

class GRL(nn.Module):
    def forward(self, x, lambda_adv):
        return GradientReverseFunction.apply(x, lambda_adv)

class DomainClassifier(nn.Module):
    def __init__(self, input_dim=3, hidden_dim1=64, hidden_dim2=32, num_domains=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim1),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim1, hidden_dim2),
            nn.ReLU(),
            nn.Linear(hidden_dim2, num_domains)
        )

    def forward(self, x):
        return self.net(x)


# ---------------------------
# Shared Encoder + Adapter（沿用 trainer.py）
# ---------------------------
class SharedEncoderAdapter(nn.Module):
    def __init__(self, input_dim=151936, dropout=0.0):
        super().__init__()
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
            nn.Linear(2048, input_dim)
        )

    def forward(self, x):
        B, T, V = x.shape
        x_flat   = x.view(B * T, V)
        out_flat = self.network(x_flat)
        return out_flat.view(B, T, V), None


# ---------------------------
# 主流程
# ---------------------------
if __name__ == "__main__":
    # 训练配置
    epochs = 3
    freeze_epochs = 1
    accumulation_steps = 40
    gamma = 10

    # 设备
    train_device    = torch.device("cuda:1" if torch.cuda.is_available() and torch.cuda.device_count() > 1 else "cuda:0" if torch.cuda.is_available() else "cpu")
    observer_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    performer_device= torch.device("cuda:1" if torch.cuda.is_available() and torch.cuda.device_count() > 1 else observer_device)

    # 构建训练/测试三元组
    tr_anchors, tr_poss, tr_negs, tr_domains = build_triplets_from_jsonl(
        JSONL_PATH, train_sources, PER_SOURCE_LIMIT, USE_RANDOM_POS_INSTEAD
    )
    ev_anchors, ev_poss, ev_negs, ev_domains = build_triplets_from_jsonl(
        JSONL_PATH, eval_sources, PER_SOURCE_LIMIT, USE_RANDOM_POS_INSTEAD
    )

    # 动态确定 num_domains
    used_domains = sorted({d for d in tr_domains + ev_domains})
    num_domains = max(used_domains) + 1 if used_domains else 1

    # 数据集
    base_model_name = "Qwen/Qwen-7B"
    train_set = TokenizedTripletDataset(
        anchors=tr_anchors, positives=tr_poss, negatives=tr_negs, domain_labels=tr_domains,
        model_name_or_path=base_model_name, device=None  # 分词结果先放CPU，按需再搬运
    )
    eval_set = TokenizedTripletDataset(
        anchors=ev_anchors, positives=ev_poss, negatives=ev_negs, domain_labels=ev_domains,
        model_name_or_path=base_model_name, device=None
    )
    N = len(train_set)
    N_eval = len(eval_set)
    print(f"Train samples: {N} | Eval samples: {N_eval} | num_domains={num_domains}")

    # 加载 Qwen 模型（冻结）
    use_bfloat16 = True
    hf_token = os.environ.get("HF_TOKEN", None)

    observer_model  = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen-7B",
        device_map={"": observer_device},
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if use_bfloat16 else torch.float32,
        token=hf_token
    )
    performer_model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen-7B-Chat",
        device_map={"": performer_device},
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if use_bfloat16 else torch.float32,
        token=hf_token
    )
    observer_model.eval()
    performer_model.eval()
    for p in observer_model.parameters():
        p.requires_grad = False
    for p in performer_model.parameters():
        p.requires_grad = False

    # 构建可训练模块
    encoder = SharedEncoderAdapter().to(train_device)
    # 如有已训练的 encoder 权重则加载
    encoder_ckpt_path = os.path.join(CHECKPOINT_DIR, "medium_contrastive_encoder.pth")
    if os.path.exists(encoder_ckpt_path):
        encoder.load_state_dict(torch.load(encoder_ckpt_path, map_location=train_device))

    domain_classifier = DomainClassifier(num_domains=num_domains).to(train_device)
    grl = GRL()

    optimizer_domain = torch.optim.Adam(domain_classifier.parameters(), lr=5e-6)
    optimizer_adv = torch.optim.Adam([
        {'params': encoder.parameters(),           'lr': 1e-6},
        {'params': domain_classifier.parameters(), 'lr': 5e-6},
    ])
    scaler = GradScaler()

    triplet_criterion = nn.TripletMarginLoss(margin=0.3)
    domain_criterion  = nn.CrossEntropyLoss()

    total_steps = N * epochs
    g = torch.Generator().manual_seed(SEED)
    sample_num = 0

    tokenizer = train_set.tokenizer
    pad_id = 151643 if tokenizer.pad_token_id is None else tokenizer.pad_token_id

    # 开始训练循环，逐样本在线计算logits
    for epoch in range(epochs):
        print("start training", epoch)
        perm = torch.randperm(N, generator=g).tolist()

        if epoch < freeze_epochs:
            current_optimizer = optimizer_domain
            lambda_adv = 0.0
        else:
            current_optimizer = optimizer_adv

        for i in perm:
            raw_i = train_set[i]
            # 将当前三元组的 token ids 和 attention mask 取出 (在CPU上)
            anchor_ids = raw_i['anchor_input_ids'].unsqueeze(0).cpu()
            anchor_mask = raw_i['anchor_attention_mask'].unsqueeze(0).cpu()
            pos_ids = raw_i['positive_input_ids'].unsqueeze(0).cpu()
            pos_mask = raw_i['positive_attention_mask'].unsqueeze(0).cpu()
            neg_ids = raw_i['negative_input_ids'].unsqueeze(0).cpu()
            neg_mask = raw_i['negative_attention_mask'].unsqueeze(0).cpu()
            domain_label = raw_i['domain_label'].unsqueeze(0).to(train_device)  # [1]

            # 将三个序列拼接，分别发送到 observer_device 和 performer_device 计算logits
            combined_ids = torch.cat([anchor_ids, pos_ids, neg_ids], dim=0)
            combined_mask = torch.cat([anchor_mask, pos_mask, neg_mask], dim=0)
            # 发送到各模型所在设备并计算logits
            with torch.no_grad():
                # Observer 模型 logits
                obs_input_ids = combined_ids.to(observer_device)
                obs_attention_mask = combined_mask.to(observer_device)
                obs_logits_all = observer_model(input_ids=obs_input_ids, attention_mask=obs_attention_mask).logits
                # Performer 模型 logits
                perf_input_ids = combined_ids.to(performer_device)
                perf_attention_mask = combined_mask.to(performer_device)
                perf_logits_all = performer_model(input_ids=perf_input_ids, attention_mask=perf_attention_mask).logits

            # 将logits移动到训练设备，并转换为 float32 以供编码器处理
            obs_logits_all = obs_logits_all.to(device=train_device, dtype=torch.float32)
            perf_logits_all = perf_logits_all.to(device=train_device, dtype=torch.float32)

            # 拆分出 anchor/positive/negative 的 observer 和 performer logits
            a_obs = obs_logits_all[0:1]   # shape [1, T, V]
            p_obs = obs_logits_all[1:2]
            n_obs = obs_logits_all[2:3]
            a_perf = perf_logits_all[0:1]
            p_perf = perf_logits_all[1:2]
            n_perf = perf_logits_all[2:3]

            p_prog = sample_num / max(total_steps, 1)
            if epoch < freeze_epochs:
                lambda_adv = 0.0
            else:
                # 根据进度调整 lambda_adv，实现从0逐渐升至1的权重
                lambda_adv = float(2. / (1. + np.exp(-gamma * p_prog)) - 1.0)

            ctx = autocast(device_type='cuda', dtype=torch.float16) if train_device.type == 'cuda' else nullcontext()
            with ctx:
                # 前向传播通过编码器（对每个 logits 分别编码）
                a_obs_out, _ = encoder(a_obs)
                a_perf_out, _ = encoder(a_perf)
                p_obs_out, _ = encoder(p_obs)
                p_perf_out, _ = encoder(p_perf)
                n_obs_out, _ = encoder(n_obs)
                n_perf_out, _ = encoder(n_perf)

                # 计算 perplexity 和 cross-perplexity
                anchor_inputs = BatchEncoding({
                    'input_ids':      anchor_ids.to(train_device),
                    'attention_mask': anchor_mask.to(train_device),
                })
                positive_inputs = BatchEncoding({
                    'input_ids':      pos_ids.to(train_device),
                    'attention_mask': pos_mask.to(train_device),
                })
                negative_inputs = BatchEncoding({
                    'input_ids':      neg_ids.to(train_device),
                    'attention_mask': neg_mask.to(train_device),
                })
                a_ppl  = perplexity(anchor_inputs, a_perf_out)
                a_xppl = entropy(a_obs_out, a_perf_out, anchor_inputs, pad_id)
                p_ppl  = perplexity(positive_inputs, p_perf_out)
                p_xppl = entropy(p_obs_out, p_perf_out, positive_inputs, pad_id)
                n_ppl  = perplexity(negative_inputs, n_perf_out)
                n_xppl = entropy(n_obs_out, n_perf_out, negative_inputs, pad_id)

                # Triplet 损失（使用 ppl/xppl 比值）
                triplet_loss = triplet_criterion(
                    a_ppl / a_xppl,
                    p_ppl / p_xppl,
                    n_ppl / n_xppl
                )

                # 域分类输入为三个比值拼接
                domain_input = torch.cat([
                    (a_ppl / a_xppl).unsqueeze(-1),
                    (p_ppl / p_xppl).unsqueeze(-1),
                    (n_ppl / n_xppl).unsqueeze(-1),
                ], dim=-1).to(train_device)

                # 域分类损失，前 freeze_epochs 不反转梯度
                if epoch < freeze_epochs:
                    domain_logits = domain_classifier(domain_input)
                else:
                    domain_logits = domain_classifier(grl(domain_input, lambda_adv))
                domain_loss = domain_criterion(domain_logits, domain_label)

                # 累积损失
                total_loss = (domain_loss if epoch < freeze_epochs else (triplet_loss + domain_loss)) / accumulation_steps

            # 反向传播与梯度累积
            sample_num += 1
            if train_device.type == 'cuda':
                scaler.scale(total_loss).backward()
            else:
                total_loss.backward()

            if sample_num % accumulation_steps == 0:
                if train_device.type == 'cuda':
                    scaler.step(current_optimizer)
                    scaler.update()
                else:
                    current_optimizer.step()
                current_optimizer.zero_grad()

                # 打印当前 step 的损失信息
                tl = float(triplet_loss.detach().cpu())
                dl = float(domain_loss.detach().cpu())
                print(f"[Epoch {epoch}] Step {sample_num} | Triplet: {tl:.4f} | Domain: {dl:.4f} | Total: {float((triplet_loss+domain_loss).detach().cpu()):.4f}")

    # 评估（逐样本计算logits）
    # ---------------------------
    # 在训练结束后查找最佳阈值
    # ---------------------------
    if N > 0:
        encoder.eval()
        with torch.no_grad():
            all_scores = []
            all_labels = []
            for i in range(N):
                raw_i = train_set[i]
                # 取出 positive 和 negative 的 token 序列
                pos_ids = raw_i['positive_input_ids'].unsqueeze(0).cpu()
                pos_mask = raw_i['positive_attention_mask'].unsqueeze(0).cpu()
                neg_ids = raw_i['negative_input_ids'].unsqueeze(0).cpu()
                neg_mask = raw_i['negative_attention_mask'].unsqueeze(0).cpu()
                # 拼接正、负样本一起计算
                combined_ids = torch.cat([pos_ids, neg_ids], dim=0)
                combined_mask = torch.cat([pos_mask, neg_mask], dim=0)
                obs_input_ids = combined_ids.to(observer_device)
                obs_attention_mask = combined_mask.to(observer_device)
                perf_input_ids = combined_ids.to(performer_device)
                perf_attention_mask = combined_mask.to(performer_device)
                obs_logits_all = observer_model(input_ids=obs_input_ids, attention_mask=obs_attention_mask).logits
                perf_logits_all = performer_model(input_ids=perf_input_ids, attention_mask=perf_attention_mask).logits
                obs_logits_all = obs_logits_all.to(device=train_device, dtype=torch.float32)
                perf_logits_all = perf_logits_all.to(device=train_device, dtype=torch.float32)
                # 分割出 positive 和 negative 的 logits
                p_obs = obs_logits_all[0:1]
                n_obs = obs_logits_all[1:2]
                p_perf = perf_logits_all[0:1]
                n_perf = perf_logits_all[1:2]
                # 编码
                p_obs_out, _ = encoder(p_obs)
                p_perf_out, _ = encoder(p_perf)
                n_obs_out, _ = encoder(n_obs)
                n_perf_out, _ = encoder(n_perf)
                # 计算 ppl 和 xppl
                positive_inputs = BatchEncoding({
                    'input_ids': pos_ids.to(train_device),
                    'attention_mask': pos_mask.to(train_device),
                })
                negative_inputs = BatchEncoding({
                    'input_ids': neg_ids.to(train_device),
                    'attention_mask': neg_mask.to(train_device),
                })
                p_ppl = perplexity(positive_inputs, p_perf_out)
                p_xppl = entropy(p_obs_out, p_perf_out, positive_inputs, pad_id)
                n_ppl = perplexity(negative_inputs, n_perf_out)
                n_xppl = entropy(n_obs_out, n_perf_out, negative_inputs, pad_id)
                # 计算 score = ppl / xppl
                score_h = (p_ppl / p_xppl).item()  # 人类答案的分数
                score_m = (n_ppl / n_xppl).item()  # 机器答案的分数
                # 记录分数和标签（human=0, machine=1）
                all_scores.append(score_h); all_labels.append(0)
                all_scores.append(score_m); all_labels.append(1)

        # 根据 score 找到最高 F1 的阈值
        total = len(all_scores)
        actual_positive = sum(all_labels)  # 实际机器答案数量 (label=1)
        actual_negative = total - actual_positive  # 实际人类答案数量
        # 将分数和标签按分数升序排序
        sorted_indices = sorted(range(total), key=lambda k: all_scores[k])
        sorted_scores = [all_scores[k] for k in sorted_indices]
        sorted_labels = [all_labels[k] for k in sorted_indices]
        # 前缀和（累计正类数量）
        prefix_pos = []
        running_pos = 0
        for lbl in sorted_labels:
            if lbl == 1:
                running_pos += 1
            prefix_pos.append(running_pos)
        best_f1 = 0.0
        best_thr = None
        best_rule = None

        # 情况1：score <= thr 判定为机器（正类）
        # 考虑阈值低于最小值（无样本被预测为机器）的情况（F1=0，因为召回=0），可跳过因为不会是最佳
        # 遍历每个唯一分数值作为阈值（包含该值）
        i = 0
        n_scores = total
        while i < n_scores:
            thr_val = sorted_scores[i]
            # 找到值为 thr_val 的最后一个索引 j
            j = i
            while j + 1 < n_scores and math.isclose(sorted_scores[j+1], thr_val, rel_tol=1e-12, abs_tol=1e-12):
                j += 1
            # 预测为机器的数量（score <= thr_val 的样本数）
            predicted_pos = j + 1
            TP = prefix_pos[j]  # <= thr 的正类数
            FP = predicted_pos - TP  # <= thr 的负类数
            FN = actual_positive - TP  # 未预测出的正类数
            precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
            recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
            f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
            if f1 > best_f1:
                best_f1 = f1
                best_thr = thr_val
                best_rule = "<="
            # 跳过相同分数的样本，直接到下一个不同分数
            i = j + 1

        # 情况2：score >= thr 判定为机器（正类）
        # 考虑阈值高于最大值（无样本被预测为机器）的情况（F1=0），可跳过
        # 先考虑阈值低于最小值（所有样本均预测为机器）
        if total > 0:
            # 所有样本预测为机器
            TP = actual_positive
            FP = actual_negative
            FN = 0
            precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
            recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
            f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
            if f1 > best_f1:
                best_f1 = f1
                best_thr = sorted_scores[0]  # 阈值可取最小值（含义：score >= min 即全部预测为机器）
                best_rule = ">="
        # 遍历每个唯一分数值，考虑阈值略高于该值的情况（排除该值及以下）
        i = 0
        while i < n_scores:
            thr_val = sorted_scores[i]
            j = i
            while j + 1 < n_scores and math.isclose(sorted_scores[j+1], thr_val, rel_tol=1e-12, abs_tol=1e-12):
                j += 1
            # 阈值设为刚高于 thr_val，则 <= thr_val 的样本都预测为人类， > thr_val 的预测为机器
            predicted_pos = total - (j + 1)
            TP = actual_positive - prefix_pos[j]  # > thr 的正类数 = 总正类 - <= thr 的正类数
            FP = predicted_pos - TP  # > thr 的负类数
            FN = prefix_pos[j]       # 未被预测出的正类 = <= thr 的正类数
            precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
            recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
            f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
            if f1 > best_f1:
                best_f1 = f1
                # 对于 score >= thr 判定机器的情况，如果最佳阈值在某个值之上，
                # 可将阈值设置为该值（分类时应使用 >= 判断，即包含该值以上为机器）。
                best_thr = thr_val
                best_rule = ">="
            i = j + 1

        # 打印训练集上最佳阈值和对应F1
        if best_thr is not None:
            print(f"[Train] Best F1 = {best_f1:.4f} at threshold {best_thr:.6f} (rule: 'score {best_rule} threshold -> machine')")
        else:
            print("[Train] No threshold found (no positive samples)")


    # 保存最终模型
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    torch.save(encoder.state_dict(), os.path.join(CHECKPOINT_DIR, "shared_encoder_adapter_jsonl_" + eval_sources[0] + ".pth"))
    torch.save(domain_classifier.state_dict(), os.path.join(CHECKPOINT_DIR, "domain_classifier_jsonl_" + eval_sources[0] + ".pth"))
    print("Done.")
