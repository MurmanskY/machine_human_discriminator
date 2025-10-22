'''
构建方法中的流程，用于测试
任务；二分类，直接使用 阈值数值 进行分类
输入：文本tsv
输出：准确率，散点图
'''
from typing import Union
import pandas as pd
import os
import numpy as np
import torch
import transformers
from transformers import BatchEncoding
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.utils.data import Dataset
import torch.multiprocessing as mp
mp.set_start_method('spawn', force=True)
import torch.nn.functional as F
import argparse





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

        # 直接调用 _tokenize，与 Binoculars 一致
        encodings = self._tokenize([anchor, positive, negative])

        return {
            'anchor_input_ids':        encodings['input_ids'][0],
            'anchor_attention_mask':   encodings['attention_mask'][0],
            'positive_input_ids':      encodings['input_ids'][1],
            'positive_attention_mask': encodings['attention_mask'][1],
            'negative_input_ids':      encodings['input_ids'][2],
            'negative_attention_mask': encodings['attention_mask'][2],
        }
        
        
        
        
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
    
    device = logits.device
    # 先把 labels 和 mask 转过去
    labels = encoding.input_ids.to(device)
    mask   = encoding.attention_mask.to(device).float()
    
    # 1) shift
    shifted_logits = logits[..., :-1, :] / temperature              # [B, T-1, V]
    shifted_labels = labels[..., 1:]                     # [B, T-1]
    shifted_mask   = mask[..., 1:].float()        # [B, T-1]

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

    mask = encoding.attention_mask.to(q_logits.device).float()                             # [B, T]

    if median:
        ce = ce.masked_fill(mask == 0, float("nan"))
        return torch.nanmedian(ce, dim=1).values                        # [B]
    else:
        summed = (ce * mask).sum(dim=1)                                 # [B]
        denom  = mask.sum(dim=1).clamp(min=1e-6)                        # [B]
        return summed / denom

    
        
        
        
@torch.no_grad()
def get_logits_pair(encodings: transformers.BatchEncoding,
                    observer_model,
                    performer_model,
                    observer_device,
                    performer_device,
                    output_device,
                    ) -> torch.Tensor:
    
    encodings_ids_cpu = encodings.input_ids.cpu()
    encodings_mask_cpu = encodings.attention_mask.cpu()
    
    encodings_observer = BatchEncoding({
        "input_ids":      encodings_ids_cpu.to(observer_device),
        "attention_mask": encodings_mask_cpu.to(observer_device)
    })
    
    encodings_performer = BatchEncoding({
        "input_ids":      encodings_ids_cpu.to(performer_device),
        "attention_mask": encodings_mask_cpu.to(performer_device)
    })
    
    observer_logits = observer_model(**encodings_observer).logits # torch.tensor
    performer_logits = performer_model(**encodings_performer).logits # torch.tensor

    return observer_logits, performer_logits




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
            # nn.Linear(hidden_dim2, hidden_dim3),
            # nn.ReLU(),
            # nn.Dropout(dropout),
            # nn.Linear(hidden_dim3, hidden_dim4),
            # nn.ReLU(),
            # nn.Dropout(dropout),
            # nn.Linear(hidden_dim4, 512),
            nn.Linear(2048, input_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_shape = x.shape                 # e.g. (1, 1, T, V)
        V = orig_shape[-1]
        # flatten 除了最后一维以外所有维度
        x_flat = x.flatten(start_dim=0, end_dim=-2)  # [B*T, V]
        out_flat = self.network(x_flat)              # [B*T, V]
        return out_flat.view(*orig_shape)            # [..., T, V]





if __name__ == "__main__":
    def parse_args():
        parser = argparse.ArgumentParser(
            description="二分类任务脚本：输入TSV，输出准确率与散点图结果"
        )
        parser.add_argument(
            "-i", "--input_path",
            type=str,
            required=True,
            help="输入的文本TSV文件路径"
        )
        parser.add_argument(
            "-o", "--output_path",
            type=str,
            default="./results/scores_new.xlsx",
            help="结果保存的 Excel 路径（含文件名）"
        )
        return parser.parse_args()
    
    args = parse_args()
    input_tsv  = args.input_path
    output_xlsx = args.output_path

    
    
    
    
    
    observer_device = "cuda:0" if torch.cuda.is_available() else "cpu"
    performer_device = "cuda:1" if torch.cuda.device_count() > 1 else observer_device
    
    use_bfloat16 = True
    huggingface_config = {
        # Only required for private models from Huggingface (e.g. LLaMA models)
        "TOKEN": os.environ.get("HF_TOKEN", None)
    }
    
    
    
    
    ch_tokenizer = get_tokenizer("Qwen/Qwen-7B")
    
    
    
    
    
    
    
    
    observer_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-7B",
                                                                device_map={"": observer_device},
                                                                trust_remote_code=True,
                                                                torch_dtype=torch.bfloat16 if use_bfloat16
                                                                else torch.float32,
                                                                token=huggingface_config["TOKEN"]
                                                                )
    performer_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-7B-Chat",
                                                                device_map={"": performer_device},
                                                                trust_remote_code=True,
                                                                torch_dtype=torch.bfloat16 if use_bfloat16
                                                                else torch.float32,
                                                                token=huggingface_config["TOKEN"]
                                                                )
    
    observer_model.eval()
    performer_model.eval()
    for p in observer_model.parameters():   p.requires_grad = False
    for p in performer_model.parameters():  p.requires_grad = False
    
    
    ob_encoder = MediumContrastiveEncoder().to(observer_device).to(torch.bfloat16)
    ob_state_dict = torch.load("./checkpoints/shared_encoder_adapter_device_0.pth", map_location=observer_device)
    ob_encoder.load_state_dict(ob_state_dict)
    ob_encoder.eval()
    
    pe_encoder = MediumContrastiveEncoder().to(performer_device).to(torch.bfloat16)
    pe_state_dict = torch.load("./checkpoints/shared_encoder_adapter_device_0.pth", map_location=performer_device)
    pe_encoder.load_state_dict(pe_state_dict)
    pe_encoder.eval()
    
    
    # df = pd.read_csv('../datasets/CSL/csl_abstract_40k_sample_random_500.tsv', sep='\t', encoding='utf-8')
    # df = pd.read_csv('../datasets/CSL/csl_abstract_to_combine_LLM_ds_db_500.tsv', sep='\t', encoding='utf-8')
    
    
    # df = pd.read_csv('../datasets/CSL/csl_abstract_40k_sample_random_2000.tsv', sep='\t', encoding='utf-8')
    # df = pd.read_csv('../datasets/CSL/csl_abstract_to_combine_LLM_ds_db_2000.tsv', sep='\t', encoding='utf-8')
    # df = pd.read_csv('./datasets/test/chinese_book.tsv', sep='\t', encoding='utf-8')
    df = pd.read_csv(input_tsv, sep='\t', encoding='utf-8')
    
    
    scores = []
    # 按行遍历摘要列
    for index, row in df.iterrows():
        abstract = row['abstract'] # 获得对应的摘要内容
        # abstract = row['machine'] # 获得对应的摘要内容

        batch_size = len(abstract)
        encodings = ch_tokenizer(
            abstract,
            return_tensors="pt",
            padding="max_length" if batch_size > 1 else False,
            truncation=True,
            max_length=512,
            return_token_type_ids=False
        )
        
        
        # inputs = {
        #     'input_ids':      encodings["anchor_input_ids"],
        #     'attention_mask': encodings["anchor_attention_mask"]
        # }
        
        # inputs_encoding = BatchEncoding(inputs)
        
        ob_logits, pe_logits = get_logits_pair(
            encodings=encodings,
            observer_model=observer_model,
            performer_model=performer_model,
            observer_device=observer_device,
            performer_device=performer_device,
            output_device=observer_device
        )
        
        ob_encode_logits = ob_encoder(ob_logits)
        pe_encode_logits = pe_encoder(pe_logits)
        
        ppl = perplexity(encodings, pe_encode_logits) # numpy
        x_ppl = entropy(ob_encode_logits.to(performer_device), pe_encode_logits, encodings, 151643) #numpy
        score = ppl/x_ppl
        scores.append(score.cpu().item())
        df_scores = pd.DataFrame({'human': scores})
        # df_scores.to_excel('./results/chinese_book.xlsx', index=False)
        df_scores.to_excel(output_xlsx, index=False)
        print(ppl/x_ppl)
    
    
    
    