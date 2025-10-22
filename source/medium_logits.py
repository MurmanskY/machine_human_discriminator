'''
通过qwen-7B的tokenizer，对文本数据的meta data进行分词，并封装成Dataset类，使用DataLoader加载
目标：获取中间的logits值，防止直接喂给模型训练，会出现cpu内存不够的这个问题。
'''
from typing import Union
import pandas as pd
import os
import numpy as np
import torch
import transformers
from transformers import BatchEncoding
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.utils.data import Dataset, DataLoader
import torch.multiprocessing as mp
mp.set_start_method('spawn', force=True)
import torch.nn.functional as F




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






@torch.inference_mode()
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








if __name__ == "__main__":
    #################### 构建loader类 ####################
    model_name = "Qwen/Qwen-7B"
    dataset = TokenizedTripletDataset(
        tsv_path='./mixed_domain_datasets/raw_datasets/mixed_triplet_datasets_mode1_with_label_shuffled.tsv',
        model_name_or_path=model_name,
        device="cuda:0"  # 如需直接在 GPU 上得到张量，可传入 device
    )
    
    print(len(dataset))
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    
        
    #################### 构建训练模型数据流 ####################
    observer_device = "cuda:0" if torch.cuda.is_available() else "cpu"
    performer_device = "cuda:1" if torch.cuda.device_count() > 1 else observer_device
    output_device = observer_device
    
    
    use_bfloat16 = True
    huggingface_config = {
        # Only required for private models from Huggingface (e.g. LLaMA models)
        "TOKEN": os.environ.get("HF_TOKEN", None)
    }
    
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
    
    
    
    out_dir = "./mixed_domain_datasets/logits"
    os.makedirs(out_dir, exist_ok=True)
    
    
    # 用 list 收集
    for idx, tuple in enumerate(loader):
        # print(tuple.keys())
        anchor_inputs = {
            'input_ids':      tuple["anchor_input_ids"],
            'attention_mask': tuple["anchor_attention_mask"]
        }
        anchor_inputs_batch_encoding = BatchEncoding(anchor_inputs)
        anchor_observer_logits, anchor_performer_logits = get_logits_pair(encodings=anchor_inputs_batch_encoding,
                                    observer_model=observer_model,
                                    performer_model=performer_model,
                                    observer_device=observer_device,
                                    performer_device=performer_device,
                                    output_device=output_device)

        
        positive_inputs = {
            'input_ids':      tuple["positive_input_ids"],
            'attention_mask': tuple["positive_attention_mask"]
        }
        positive_inputs_batch_encoding = BatchEncoding(positive_inputs)
        positive_observer_logits, positive_performer_logits = get_logits_pair(encodings=positive_inputs_batch_encoding,
                                    observer_model=observer_model,
                                    performer_model=performer_model,
                                    observer_device=observer_device,
                                    performer_device=performer_device,
                                    output_device=output_device)
        
        
        negative_inputs = {
            'input_ids':      tuple["negative_input_ids"],
            'attention_mask': tuple["negative_attention_mask"]
        }
        
        negative_inputs_batch_encoding = BatchEncoding(negative_inputs)
        
        negative_observer_logits, negative_performer_logits = get_logits_pair(encodings=negative_inputs_batch_encoding,
                                    observer_model=observer_model,
                                    performer_model=performer_model,
                                    observer_device=observer_device,
                                    performer_device=performer_device,
                                    output_device=output_device)
        
        sample_dict = {
            "anchor_observer_logits": anchor_observer_logits.cpu(),
            "anchor_performer_logits": anchor_performer_logits.cpu(),
            "positive_observer_logits": positive_observer_logits.cpu(),
            "positive_performer_logits": positive_performer_logits.cpu(),
            "negative_observer_logits": negative_observer_logits.cpu(),
            "negative_performer_logits": negative_performer_logits.cpu(),
        }
        
        torch.save(sample_dict, os.path.join(out_dir, f"{idx:06d}.pt"))
        # 防止内存泄漏
        del anchor_observer_logits, anchor_performer_logits
        del positive_observer_logits, positive_performer_logits
        del negative_observer_logits, negative_performer_logits
        del sample_dict
        
        
        if idx % 50 == 0:
            print(f"Processed {idx+1}/{len(dataset)}")
        