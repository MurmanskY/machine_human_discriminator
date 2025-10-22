'''
2025-6-22
通过qwen-7B的tokenizer，对文本数据的meta data进行分词，并封装成Dataset类，使用DataLoader加载
'''
from typing import Union
import pandas as pd
import os
import numpy as np
import torch
import transformers
from transformers import BatchEncoding
from torch import nn
# from torch.cuda.amp import autocast
from torch.nn import TripletMarginLoss
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerFast
from torch.utils.data import Dataset, DataLoader
import torch.multiprocessing as mp
mp.set_start_method('spawn', force=True)
import torch.nn.functional as F
from torch.amp import GradScaler, autocast






def get_tokenizer(model_name_or_path: str,
                  use_fast: bool = True,
                  trust_remote_code: bool = True) -> AutoTokenizer:
    """
    单独加载并返回一个 tokenizer，自动补齐 pad_token。
    """
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        use_fast=use_fast,
        trust_remote_code=trust_remote_code,
        pad_token="<|endoftext|>"
    )

    return tokenizer





class TokenizedTripletDataset(Dataset):
    def __init__(self,
                 tsv_path: str,
                 model_name_or_path: str,
                 max_length: int = 512,
                 device: Union[str, torch.device] = None):
        df = pd.read_csv(tsv_path, sep='\t')
        self.anchor_texts = df['anchor'].tolist()
        self.positive_texts = df['positive'].tolist()
        self.negative_texts = df['abstract'].tolist()
        self.domain_label = df['domain_label'].tolist()

        self.tokenizer = get_tokenizer(model_name_or_path)
        self.max_length = max_length
        self.device = torch.device(device) if device is not None else None

    def __len__(self):
        return len(self.anchor_texts)

    def _tokenize(self, texts: list[str]) -> transformers.BatchEncoding:
        """
        与 Binoculars._tokenize 完全一致：
        - batch size 为 1 时不做 padding
        - batch size > 1 时 padding='longest'
        - return_token_type_ids=False
        - truncation
        - max_length 约束
        """
        batch_size = len(texts)
        encodings = self.tokenizer(
            texts,
            return_tensors="pt",
            padding="max_length" if batch_size > 1 else False,
            truncation=True,
            max_length=self.max_length,
            return_token_type_ids=False
        )
        if self.device is not None:
            encodings = {k: v.to(self.device) for k, v in encodings.items()}
        return encodings

    def __getitem__(self, idx):
        anchor = self.anchor_texts[idx]
        positive = self.positive_texts[idx]
        negative = self.negative_texts[idx]
        domain_label = self.domain_label[idx]

        # 直接调用 _tokenize，与 Binoculars 一致
        encodings = self._tokenize([anchor, positive, negative])

        return {
            'anchor_input_ids':        encodings['input_ids'][0],
            'anchor_attention_mask':   encodings['attention_mask'][0],
            'positive_input_ids':      encodings['input_ids'][1],
            'positive_attention_mask': encodings['attention_mask'][1],
            'negative_input_ids':      encodings['input_ids'][2],
            'negative_attention_mask': encodings['attention_mask'][2],
            'domain_label': torch.tensor(domain_label, dtype=torch.long)
        }




class SavedLogitsDataset(Dataset):
    def __init__(self, dir_path: str):
        self.files    = sorted(os.listdir(dir_path))
        self.dir_path = dir_path

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path   = os.path.join(self.dir_path, self.files[idx])
        sample = torch.load(path, map_location="cpu")
        # 这里把最外面的 1 维 squeeze 掉，保证返回 [T,V]
        # for k, v in sample.items():
        #     sample[k] = v.squeeze(0)
        return sample




def perplexity(encoding: BatchEncoding,
               logits: torch.Tensor,
               median: bool = False,
               temperature: float = 1.0) -> torch.Tensor:
    """
    计算每个样本的 perplexity（平均 token CE）。
    输入:
      - encoding.input_ids:      [B, T]
      - encoding.attention_mask: [B, T]
      - logits:                  [B, T, V]
    输出:
      - ppl: [B] 张量，可求导
    """
    # 1) shift
    shifted_logits = logits[..., :-1, :] / temperature              # [B, T-1, V]
    shifted_labels = encoding.input_ids[..., 1:]                     # [B, T-1]
    shifted_mask   = encoding.attention_mask[..., 1:].float()        # [B, T-1]

    # 2) per-token cross-entropy
    #    F.cross_entropy 接受 [B, V, T-1] 和 [B, T-1]
    ce = F.cross_entropy(
        shifted_logits.transpose(1, 2), 
        shifted_labels, 
        reduction="none"
    )                                                                 # [B, T-1]

    if median:
        # mask out padding
        ce = ce.masked_fill(shifted_mask == 0, float("nan"))
        return torch.nanmedian(ce, dim=1).values                         # [B]
    else:
        summed = (ce * shifted_mask).sum(dim=1)                         # [B]
        denom  = shifted_mask.sum(dim=1).clamp(min=1e-6)                # [B]
        return summed / denom                                          # [B]


def entropy(p_logits: torch.Tensor,
            q_logits: torch.Tensor,
            encoding: BatchEncoding,
            pad_token_id: int,
            median: bool = False,
            sample_p: bool = False,
            temperature: float = 1.0) -> torch.Tensor:
    """
    计算 p 对 q 的 cross-entropy (xppl)，可选对 p 采样简化。
    输入:
      - p_logits, q_logits: [B, T, V]
      - encoding.attention_mask: [B, T]
    输出:
      - xppl: [B] 张量，可求导
    """
    B, T, V = q_logits.shape

    # 1) 温度缩放 & 概率分布
    p_scores = p_logits / temperature
    q_scores = q_logits / temperature
    p_proba = F.softmax(p_scores, dim=-1)                              # [B, T, V]
    q_logp  = F.log_softmax(q_scores, dim=-1)                          # [B, T, V]

    if sample_p:
        # 每个位置采一个类别
        p_flat = p_proba.view(B * T, V)
        idx    = torch.multinomial(p_flat, num_samples=1).view(B, T)    # [B, T]
        # 交叉熵简化成 -log q(idx)
        ce = -q_logp.gather(dim=-1, index=idx.unsqueeze(-1)).squeeze(-1)  # [B, T]
    else:
        # 标准 p·(-log q)
        ce = -(p_proba * q_logp).sum(dim=-1)                            # [B, T]

    mask = encoding.attention_mask.float()                             # [B, T]

    if median:
        ce = ce.masked_fill(mask == 0, float("nan"))
        return torch.nanmedian(ce, dim=1).values                        # [B]
    else:
        summed = (ce * mask).sum(dim=1)                                 # [B]
        denom  = mask.sum(dim=1).clamp(min=1e-6)                        # [B]
        return summed / denom


# added 2025 9 26 Gradient Reversal Layer + Domain Classifier
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


# added 2025 9 26 Domain Classifier
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




# Shared Encoder + Adapter
class SharedEncoderAdapter(nn.Module):
    def __init__(self, input_dim=151936, dropout=0.0):
        super(SharedEncoderAdapter, self).__init__()
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
        x_flat = x.view(B*T, V)
        out_flat = self.network(x_flat)
        return out_flat.view(B, T, V), None



def set_requires_grad(model, requires_grad=True):
    """冻结或解冻模型参数"""
    for param in model.parameters():
        param.requires_grad = requires_grad




if __name__ == "__main__":
    epochs = 3
    freeze_epochs = 1
    accumulation_steps = 40  # 模拟逻辑batch size为
    device = torch.device("cuda:1")
    num_domains = 3
    #################### 构建loader类 ####################
    g = torch.Generator().manual_seed(42)
    
    
    old_dataset = TokenizedTripletDataset(
        tsv_path='./mixed_domain_datasets/raw_datasets/mixed_triplet_datasets_mode1_with_label_shuffled.tsv',
        model_name_or_path="Qwen/Qwen-7B",  # 想要获得tokenizer
        device=device  # 如需直接在 GPU 上得到张量，可传入 device
    )
    old_loader = DataLoader(old_dataset, batch_size=1, generator=g, shuffle=True, num_workers=0, pin_memory=False)
    
        
    dataset = SavedLogitsDataset("./mixed_domain_datasets/logits")
    loader  = DataLoader(dataset, batch_size=1, generator=g, shuffle=True, num_workers=0, pin_memory=False)
    


    encoder = SharedEncoderAdapter().to(device)
    encoder.load_state_dict(torch.load("./checkpoints/medium_contrastive_encoder.pth", map_location=device))
    domain_classifier = DomainClassifier(num_domains=num_domains).to(device)
    grl = GRL()
    
    '''开始是训练domain classifier'''
    optimizer_domain = torch.optim.Adam(domain_classifier.parameters(), lr=5e-6)
    '''后续开始对抗训练，使用如下训练参数'''
    optimizer_adv = torch.optim.Adam([
        {'params': encoder.parameters(), 'lr': 1e-6},
        {'params': domain_classifier.parameters(), 'lr': 5e-6}
    ])
    scaler = GradScaler()
    
    # 损失函数
    triplet_criterion = torch.nn.TripletMarginLoss(margin=0.3)
    domain_criterion = torch.nn.CrossEntropyLoss()


    # old_loader为文本三元组数据集
    # loader为存储Logits值的数据集
    sample_num = 0
    total_steps = len(old_loader) * epochs
    gamma = 10
    for epoch in range(epochs):
        
        
        '''先固定sharedencoder，再训练domain classifer'''
        if epoch < freeze_epochs:
            print(f"Epoch {epoch}: 冻结 SharedEncoderAdapter，只训练 DomainClassifier")
            set_requires_grad(encoder, False)   # 冻结
            current_optimizer = optimizer_domain
            lambda_adv = 0.0                     # 关闭 GRL
        else:
            print(f"Epoch {epoch}: 解冻 SharedEncoderAdapter，开始对抗训练")
            set_requires_grad(encoder, True)    # 解冻
            current_optimizer = optimizer_adv
            
            if epoch == freeze_epochs:
                print("清空 optimizer_adv 残留梯度，确保对抗训练从零梯度开始")
                optimizer_adv.zero_grad(set_to_none=True)
        
        
        
        for raw_batch, logits_batch in zip(old_loader, loader):
            current_step = sample_num
            p = current_step / total_steps
            
            '''先训练 domain classifier，然后再对抗训练'''
            if epoch < freeze_epochs:
                lambda_adv = 0
            else:
                lambda_adv = 2. / (1. + np.exp(-gamma * p)) - 1 # 动态调整
            
            
            
            for k, v in logits_batch.items():
                logits_batch[k] = v.squeeze(dim=0).to(device)
                
            
            with autocast(device_type='cuda', dtype=torch.float16):
                
                '''anchor，使用shared encoder和adapter处理logits'''
                anchor_output, anchor_feat = encoder(logits_batch["anchor_observer_logits"])
                anchor_output_perf, _ = encoder(logits_batch["anchor_performer_logits"])
                '''anchor，计算binoculars socre'''
                anchor_inputs = {
                    'input_ids':      raw_batch["anchor_input_ids"],
                    'attention_mask': raw_batch["anchor_attention_mask"]
                }
                anchor_inputs_batch_encoding = BatchEncoding(anchor_inputs)
                anchor_ppl = perplexity(anchor_inputs_batch_encoding, anchor_output_perf)
                anchor_x_ppl = entropy(anchor_output, anchor_output_perf, anchor_inputs_batch_encoding, 151643)
                
                
                
                '''positive，使用shared encoder和adapter处理logits'''
                pos_output, pos_feat = encoder(logits_batch["positive_observer_logits"])
                pos_output_perf, _ = encoder(logits_batch["positive_performer_logits"])
                '''positive，计算binoculars socre'''
                positive_inputs = {
                    'input_ids':      raw_batch["positive_input_ids"],
                    'attention_mask': raw_batch["positive_attention_mask"]
                }
                positive_inputs_batch_encoding = BatchEncoding(positive_inputs)
                pos_ppl = perplexity(positive_inputs_batch_encoding, pos_output_perf)
                pos_x_ppl = entropy(pos_output, pos_output_perf, positive_inputs_batch_encoding, 151643)
                
                
                
                '''negative，使用shared encoder和adapter处理logits'''
                neg_output, neg_feat = encoder(logits_batch["negative_observer_logits"])
                neg_output_perf, _ = encoder(logits_batch["negative_performer_logits"])
                '''negative，计算binoculars socre'''
                negative_inputs = {
                    'input_ids':      raw_batch["negative_input_ids"],
                    'attention_mask': raw_batch["negative_attention_mask"]
                }
                negative_inputs_batch_encoding = BatchEncoding(negative_inputs)
                neg_ppl = perplexity(negative_inputs_batch_encoding, neg_output_perf)
                neg_x_ppl = entropy(neg_output, neg_output_perf, negative_inputs_batch_encoding, 151643)



                '''基础triplet loss'''
                triplet_loss = triplet_criterion(
                    anchor_ppl/anchor_x_ppl,
                    pos_ppl/pos_x_ppl,
                    neg_ppl/neg_x_ppl
                )
                
                
                
                '''Domain Loss GRL'''
                domain_input = torch.cat([
                    (anchor_ppl / anchor_x_ppl).unsqueeze(-1),
                    (pos_ppl / pos_x_ppl).unsqueeze(-1),
                    (neg_ppl / neg_x_ppl).unsqueeze(-1)
                ], dim=-1)  # [B, 5]



                '''先训练domain classifier，再对抗训练'''
                if epoch < freeze_epochs:
                    domain_logits = domain_classifier(domain_input)  # 直接输入
                else:
                    domain_logits = domain_classifier(grl(domain_input, lambda_adv))

                
                
                '''计算 Domain Loss'''
                domain_label = raw_batch['domain_label'].to(device)
                domain_loss = domain_criterion(domain_logits, domain_label)
                
                
                
                
                
                '''总损失'''
                '''先训练domain classifier，再对抗训练'''
                if epoch < freeze_epochs:
                    total_loss = domain_loss / accumulation_steps
                else:
                    total_loss = (triplet_loss + domain_loss) / accumulation_steps

                
                
                
            '''反向传播，梯度累计'''
            sample_num += 1
            scaler.scale(total_loss).backward()
            
            
            '''每 accumulation_steps 次更新一次参数'''
            if (sample_num % accumulation_steps == 0):
                scaler.step(current_optimizer)
                scaler.update()
                current_optimizer.zero_grad()


            '''打印日志'''
            if sample_num % accumulation_steps == 0:
                print(f"[Epoch {epoch}] Step {sample_num} | "
                      f"Triplet: {triplet_loss.item():.4f}, "
                      f"Domain: {domain_loss.item():.4f}, "
                      f"Total: {total_loss.item():.4f}")
            
        
    torch.save(encoder.state_dict(), "./checkpoints/shared_encoder_adapter_device_1.pth")
    torch.save(domain_classifier.state_dict(), "./checkpoints/domain_classifier_device_1.pth")
