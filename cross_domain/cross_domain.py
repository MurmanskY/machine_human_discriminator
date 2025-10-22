
# -*- coding: utf-8 -*-
"""
合并改版：JSONL 数据 + 可选 source 训练/测试 + 一次性前向，无中间落盘
- 输入：单个 JSONL（每行含 question、human_answers、chatgpt_answers、source）
- 每个 source 取前 K 条（默认 500）
- 三元组定义：
    anchor = question
    pos    = 当前条的第一条 human_answers（或可切换为“剩余样本随机 human”）
    neg    = 当前条的第一条 chatgpt_answers
- 训练/测试来源可配置：train_sources / eval_sources
- 保持原 trainer.py 的损失与大多数超参；num_domains 根据选中来源自动确定
- 仍使用 Qwen/Qwen-7B 与 Qwen/Qwen-7B-Chat
"""

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

import torch.multiprocessing as mp
mp.set_start_method('spawn', force=True)


# ---------------------------
# 配置
# ---------------------------
# 路径
JSONL_PATH = '../datasets/HC3_zh/all.jsonl'  # 改为你的 jsonl 路径
CHECKPOINT_DIR = '../checkpoints'

# 训练/评估来源可配
ALL_SOURCES = ["open_qa", "baike", "nlpcc_dbpa", "medicine", "finance", "psychology", "law"]
train_sources: List[str] = ["open_qa", "baike", "nlpcc_dbpa", "medicine", "finance", "psychology"]              # 参与训练的 source 子集
eval_sources:  List[str] = ["law"]  # 参与测试的 source 子集

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
    "nlpcc_dbpa": 2,
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

            humans = [ _clean(x) for x in _to_list(obj.get("human_answers")) if str(x).strip() ]
            gpts   = [ _clean(x) for x in _to_list(obj.get("chatgpt_answers")) if str(x).strip() ]

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
    读取三元组：anchor/positive/abstract，并包含 domain_label。
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
                    performer_device: Union[str, torch.device]) -> (torch.Tensor, torch.Tensor):
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

    obs_logits   = observer_model(**enc_obs).logits
    perf_logits  = performer_model(**enc_perf).logits
    return obs_logits, perf_logits


def compute_full_logits(dataset: TokenizedTripletDataset,
                        observer_model,
                        performer_model,
                        observer_device,
                        performer_device) -> Dict[str, torch.Tensor]:
    anchor_enc   = dataset._tokenize(dataset.anchor_texts)
    positive_enc = dataset._tokenize(dataset.positive_texts)
    negative_enc = dataset._tokenize(dataset.negative_texts)

    a_obs, a_perf = get_logits_pair(anchor_enc,   observer_model, performer_model, observer_device, performer_device)
    p_obs, p_perf = get_logits_pair(positive_enc, observer_model, performer_model, observer_device, performer_device)
    n_obs, n_perf = get_logits_pair(negative_enc, observer_model, performer_model, observer_device, performer_device)

    logits_all = {
        'anchor_observer_logits':   a_obs.cpu(),
        'anchor_performer_logits':  a_perf.cpu(),
        'positive_observer_logits': p_obs.cpu(),
        'positive_performer_logits':p_perf.cpu(),
        'negative_observer_logits': n_obs.cpu(),
        'negative_performer_logits':n_perf.cpu(),
    }
    del a_obs, a_perf, p_obs, p_perf, n_obs, n_perf, anchor_enc, positive_enc, negative_enc
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return logits_all


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
        x_flat   = x.view(B*T, V)
        out_flat = self.network(x_flat)
        return out_flat.view(B, T, V), None


# ---------------------------
# 主流程
# ---------------------------
if __name__ == "__main__":
    # 训练配置（尽量保持不变；num_domains 根据来源动态设定）
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
        JSONL_PATH, eval_sources,  PER_SOURCE_LIMIT, USE_RANDOM_POS_INSTEAD
    )

    # num_domains 动态
    used_domains = sorted({d for d in tr_domains + ev_domains})
    num_domains = max(used_domains) + 1 if used_domains else 1

    # 数据集
    base_model_name = "Qwen/Qwen-7B"
    train_set = TokenizedTripletDataset(
        anchors=tr_anchors, positives=tr_poss, negatives=tr_negs, domain_labels=tr_domains,
        model_name_or_path=base_model_name, device=train_device
    )
    eval_set = TokenizedTripletDataset(
        anchors=ev_anchors, positives=ev_poss, negatives=ev_negs, domain_labels=ev_domains,
        model_name_or_path=base_model_name, device=train_device
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
    for p in observer_model.parameters():  p.requires_grad = False
    for p in performer_model.parameters(): p.requires_grad = False

    # 一次性前向 logits（训练集）
    print(f"Computing train logits for {N} samples ...")
    train_logits = compute_full_logits(
        dataset=train_set,
        observer_model=observer_model,
        performer_model=performer_model,
        observer_device=observer_device,
        performer_device=performer_device
    )

    # 一次性前向 logits（评估集）
    if N_eval > 0:
        print(f"Computing eval logits for {N_eval} samples ...")
        eval_logits = compute_full_logits(
            dataset=eval_set,
            observer_model=observer_model,
            performer_model=performer_model,
            observer_device=observer_device,
            performer_device=performer_device
        )
    else:
        eval_logits = None

    # 释放 Qwen 模型显存
    del observer_model, performer_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # 构建可训练模块
    encoder = SharedEncoderAdapter().to(train_device)
    if os.path.exists(os.path.join(CHECKPOINT_DIR, "medium_contrastive_encoder.pth")):
        encoder.load_state_dict(torch.load(os.path.join(CHECKPOINT_DIR, "medium_contrastive_encoder.pth"), map_location=train_device))

    domain_classifier = DomainClassifier(num_domains=num_domains).to(train_device)
    grl = GRL()

    optimizer_domain = torch.optim.Adam(domain_classifier.parameters(), lr=5e-6)
    optimizer_adv = torch.optim.Adam([
        {'params': encoder.parameters(),          'lr': 1e-6},
        {'params': domain_classifier.parameters(),'lr': 5e-6},
    ])
    scaler = GradScaler()

    triplet_criterion = torch.nn.TripletMarginLoss(margin=0.3)
    domain_criterion  = torch.nn.CrossEntropyLoss()

    # 训练循环
    epochs = epochs
    total_steps = N * epochs
    g = torch.Generator().manual_seed(SEED)
    sample_num = 0

    tokenizer = train_set.tokenizer
    pad_id = 151643 if tokenizer.pad_token_id is None else tokenizer.pad_token_id

    for epoch in range(epochs):
        perm = torch.randperm(N, generator=g).tolist()

        if epoch < freeze_epochs:
            current_optimizer = optimizer_domain
            lambda_adv = 0.0
        else:
            current_optimizer = optimizer_adv

        for i in perm:
            raw_i = train_set[i]
            anchor_inputs = BatchEncoding({
                'input_ids':      raw_i['anchor_input_ids'].unsqueeze(0),
                'attention_mask': raw_i['anchor_attention_mask'].unsqueeze(0),
            })
            positive_inputs = BatchEncoding({
                'input_ids':      raw_i['positive_input_ids'].unsqueeze(0),
                'attention_mask': raw_i['positive_attention_mask'].unsqueeze(0),
            })
            negative_inputs = BatchEncoding({
                'input_ids':      raw_i['negative_input_ids'].unsqueeze(0),
                'attention_mask': raw_i['negative_attention_mask'].unsqueeze(0),
            })
            domain_label = raw_i['domain_label'].unsqueeze(0).to(train_device)   # [1]

            a_obs = train_logits['anchor_observer_logits'][i:i+1].to(train_device)
            a_perf= train_logits['anchor_performer_logits'][i:i+1].to(train_device)
            p_obs = train_logits['positive_observer_logits'][i:i+1].to(train_device)
            p_perf= train_logits['positive_performer_logits'][i:i+1].to(train_device)
            n_obs = train_logits['negative_observer_logits'][i:i+1].to(train_device)
            n_perf= train_logits['negative_performer_logits'][i:i+1].to(train_device)

            p_prog = sample_num / max(total_steps, 1)
            if epoch < freeze_epochs:
                lambda_adv = 0.0
            else:
                lambda_adv = float(2. / (1. + np.exp(-gamma * p_prog)) - 1.0)

            ctx = autocast(device_type='cuda', dtype=torch.float16) if train_device.type == 'cuda' else nullcontext()
            with ctx:
                a_obs_out, _ = encoder(a_obs)
                a_perf_out,_ = encoder(a_perf)

                p_obs_out, _ = encoder(p_obs)
                p_perf_out,_ = encoder(p_perf)

                n_obs_out, _ = encoder(n_obs)
                n_perf_out,_ = encoder(n_perf)

                a_ppl   = perplexity(anchor_inputs,  a_perf_out)
                a_xppl  = entropy(   a_obs_out, a_perf_out, anchor_inputs,  pad_id)

                p_ppl   = perplexity(positive_inputs, p_perf_out)
                p_xppl  = entropy(   p_obs_out, p_perf_out, positive_inputs, pad_id)

                n_ppl   = perplexity(negative_inputs, n_perf_out)
                n_xppl  = entropy(   n_obs_out, n_perf_out, negative_inputs, pad_id)

                triplet_loss = triplet_criterion(
                    a_ppl / a_xppl,
                    p_ppl / p_xppl,
                    n_ppl / n_xppl
                )

                domain_input = torch.cat([
                    (a_ppl / a_xppl).unsqueeze(-1),
                    (p_ppl / p_xppl).unsqueeze(-1),
                    (n_ppl / n_xppl).unsqueeze(-1),
                ], dim=-1)

                if epoch < freeze_epochs:
                    domain_logits = domain_classifier(domain_input)
                else:
                    domain_logits = domain_classifier(GRL().forward(domain_input, lambda_adv))

                domain_loss = domain_criterion(domain_logits, domain_label)

                total_loss = (domain_loss if epoch < freeze_epochs else (triplet_loss + domain_loss)) / accumulation_steps

            sample_num += 1
            if train_device.type == 'cuda':
                scaler.scale(total_loss).backward()
            else:
                total_loss.backward()

            if (sample_num % accumulation_steps == 0):
                if train_device.type == 'cuda':
                    scaler.step(current_optimizer)
                    scaler.update()
                else:
                    current_optimizer.step()
                current_optimizer.zero_grad()

            if sample_num % accumulation_steps == 0:
                tl = float(triplet_loss.detach().cpu())
                dl = float(domain_loss.detach().cpu())
                print(f"[Epoch {epoch}] Step {sample_num} | Triplet: {tl:.4f} | Domain: {dl:.4f} | Total: {float((triplet_loss+domain_loss).detach().cpu()):.4f}")

    # 简单评估（可选）
    if N_eval > 0:
        encoder.eval()
        with torch.no_grad():
            triplet_losses = []
            dom_correct, dom_total = 0, 0
            for i in range(N_eval):
                raw_i = eval_set[i]
                anchor_inputs = BatchEncoding({
                    'input_ids':      raw_i['anchor_input_ids'].unsqueeze(0),
                    'attention_mask': raw_i['anchor_attention_mask'].unsqueeze(0),
                })
                positive_inputs = BatchEncoding({
                    'input_ids':      raw_i['positive_input_ids'].unsqueeze(0),
                    'attention_mask': raw_i['positive_attention_mask'].unsqueeze(0),
                })
                negative_inputs = BatchEncoding({
                    'input_ids':      raw_i['negative_input_ids'].unsqueeze(0),
                    'attention_mask': raw_i['negative_attention_mask'].unsqueeze(0),
                })
                domain_label = raw_i['domain_label'].unsqueeze(0).to(train_device)

                a_obs = eval_logits['anchor_observer_logits'][i:i+1].to(train_device)
                a_perf= eval_logits['anchor_performer_logits'][i:i+1].to(train_device)
                p_obs = eval_logits['positive_observer_logits'][i:i+1].to(train_device)
                p_perf= eval_logits['positive_performer_logits'][i:i+1].to(train_device)
                n_obs = eval_logits['negative_observer_logits'][i:i+1].to(train_device)
                n_perf= eval_logits['negative_performer_logits'][i:i+1].to(train_device)

                a_obs_out, _ = encoder(a_obs)
                a_perf_out,_ = encoder(a_perf)
                p_obs_out, _ = encoder(p_obs)
                p_perf_out,_ = encoder(p_perf)
                n_obs_out, _ = encoder(n_obs)
                n_perf_out,_ = encoder(n_perf)

                a_ppl   = perplexity(anchor_inputs,  a_perf_out)
                a_xppl  = entropy(   a_obs_out, a_perf_out, anchor_inputs,  pad_id)
                p_ppl   = perplexity(positive_inputs, p_perf_out)
                p_xppl  = entropy(   p_obs_out, p_perf_out, positive_inputs, pad_id)
                n_ppl   = perplexity(negative_inputs, n_perf_out)
                n_xppl  = entropy(   n_obs_out, n_perf_out, negative_inputs, pad_id)

                tl = F.triplet_margin_loss(
                    a_ppl / a_xppl, p_ppl / p_xppl, n_ppl / n_xppl, margin=0.3, reduction='none'
                )
                triplet_losses.append(tl.mean().item())

                domain_input = torch.cat([
                    (a_ppl / a_xppl).unsqueeze(-1),
                    (p_ppl / p_xppl).unsqueeze(-1),
                    (n_ppl / n_xppl).unsqueeze(-1),
                ], dim=-1)
                logits = domain_classifier(domain_input)
                pred = logits.argmax(dim=-1)
                dom_correct += int((pred == domain_label).sum().item())
                dom_total += domain_label.numel()

        print(f"[Eval] TripletLoss: {np.mean(triplet_losses):.4f} | Domain Acc: {dom_correct / max(dom_total,1):.4f}")

    # 保存
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    torch.save(encoder.state_dict(), os.path.join(CHECKPOINT_DIR, "shared_encoder_adapter_jsonl.pth"))
    torch.save(domain_classifier.state_dict(), os.path.join(CHECKPOINT_DIR, "domain_classifier_jsonl.pth"))
    print("Done.")
