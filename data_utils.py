import torch

from tqdm import tqdm
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader
from transformers.tokenization_utils import PreTrainedTokenizer
from transformers.models.mbart50.tokenization_mbart50 import MBart50Tokenizer
from transformers.data.data_collator import DataCollatorForSeq2Seq


class Seq2SeqDataset(Dataset):
    
    def __init__(self, tokenizer: PreTrainedTokenizer, src_file, tgt_file=None, aux_tgt_file=None, max_src_len=448, max_tgt_len=64, multi_task=False, no_mbart_lang_token=False) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        with open(src_file, "rb") as fin:
            self.srcs = [line.decode("utf-8").strip() for line in tqdm(fin)]
        self.tgts = None
        if tgt_file is not None:
            with open(tgt_file, "rb") as fin:
                self.tgts = [line.decode("utf-8").strip() for line in tqdm(fin)]
            assert len(self.srcs) == len(self.tgts)
        self.aux_tgts = None
        if aux_tgt_file is not None:
            with open(aux_tgt_file, "rb") as fin:
                self.aux_tgts = [line.decode("utf-8").strip() for line in tqdm(fin)]
            assert len(self.srcs) == len(self.aux_tgts)
        self.max_src_len = max_src_len
        self.max_tgt_len = max_tgt_len
        self.multi_task = multi_task
        self.no_mbart_lang_token = no_mbart_lang_token
        
    def __len__(self):
        return len(self.srcs)
    
    def getter_for_normal(self, idx):
        src = self.srcs[idx]
        if self.tgts is None:
            inputs = self.tokenizer(src)
            inputs["input_ids"] = inputs["input_ids"][:self.max_src_len]
            inputs["attention_mask"] = inputs["attention_mask"][:self.max_src_len]
            return inputs
        tgt = self.tgts[idx]
        inputs = self.tokenizer(src, text_target=tgt)
        inputs["input_ids"] = inputs["input_ids"][:self.max_src_len]
        inputs["attention_mask"] = inputs["attention_mask"][:self.max_src_len]
        inputs["labels"] = inputs["labels"][:self.max_tgt_len]
        if self.aux_tgts is None:
            return inputs
        aux_tgt = self.aux_tgts[idx]
        if self.multi_task:
            inputs["ml_labels"] = self.tokenizer(aux_tgt)["input_ids"][:self.max_tgt_len]
        else:
            inputs["ml_labels"] = self.tokenizer(aux_tgt, add_special_tokens=False)["input_ids"][:self.max_tgt_len]
        return inputs
    
    def getter_for_no_mbart_lang_token(self, idx):
        src = self.srcs[idx]
        if self.tgts is None:
            inputs = self.tokenizer(src, add_special_tokens=False)
            inputs["input_ids"] = inputs["input_ids"][:self.max_src_len-1] + [self.tokenizer.eos_token_id]
            inputs["attention_mask"] = inputs["attention_mask"][:self.max_src_len-1] + [1]
            return inputs
        tgt = self.tgts[idx]
        inputs = self.tokenizer(src, text_target=tgt, add_special_tokens=False)
        inputs["input_ids"] = inputs["input_ids"][:self.max_src_len-1] + [self.tokenizer.eos_token_id]
        inputs["attention_mask"] = inputs["attention_mask"][:self.max_src_len-1] + [1]
        inputs["labels"] = inputs["labels"][:self.max_tgt_len-1] + [self.tokenizer.eos_token_id]
        return inputs
    
    def __getitem__(self, idx):
        if self.no_mbart_lang_token:
            return self.getter_for_no_mbart_lang_token(idx)
        return self.getter_for_normal(idx)


class DataCollatorForSeq2SeqCAR(DataCollatorForSeq2Seq):
    
    def __call__(self, features, return_tensors=None):
        ml_labels = []
        for feature in features:
            if "ml_labels" not in feature:
                break
            ml_labels.append(feature["ml_labels"])
            del feature["ml_labels"]
        batch = super().__call__(features, return_tensors)
        if len(ml_labels) > 0:
            max_len = max([len(l) for l in ml_labels])
            ml_labels = [l+[self.tokenizer.pad_token_id]*(max_len-len(l)) for l in ml_labels]
            batch["ml_labels"] = torch.tensor(ml_labels)
        return batch


class DataCollatorForSeq2SeqPara(DataCollatorForSeq2Seq):
    
    def __call__(self, features, return_tensors=None):
        labels_para = []
        for feature in features:
            if "ml_labels" not in feature:
                break
            labels_para.append(feature["ml_labels"])
            del feature["ml_labels"]
        batch = super().__call__(features, return_tensors)
        if len(labels_para) > 0:
            max_len = max([len(l) for l in labels_para])
            labels_para = [l+[self.tokenizer.pad_token_id]*(max_len-len(l)) for l in labels_para]
            batch["labels_para"] = torch.tensor(labels_para)
        return batch


if __name__ == "__main__":
    src_file = "data/WikiLingua_data_splits/turkish/train.src.tr"
    tgt_file = "data/WikiLingua_data_splits/turkish/train.tgt.en"
    aux_tgt_file = "data/WikiLingua_data_splits/turkish/train.tgt.tr"
    tokenizer = MBart50Tokenizer.from_pretrained(
        "facebook/mbart-large-50",
        src_lang="tr_TR",
        tgt_lang="en_XX",
    )
    train_dataset = Seq2SeqDataset(
        tokenizer,
        src_file=src_file,
        tgt_file=tgt_file,
        aux_tgt_file=aux_tgt_file,
        max_src_len=512,
        max_tgt_len=128,
    )
    eval_dataset = Seq2SeqDataset(
        tokenizer,
        src_file=src_file,
        tgt_file=tgt_file,
        max_src_len=512,
        max_tgt_len=128,
    )
    test_dataset = Seq2SeqDataset(
        tokenizer,
        src_file=src_file,
        max_src_len=512,
        max_tgt_len=128,
    )
    print({k: len(v) for k, v in train_dataset[0].items()})
    print({k: len(v) for k, v in eval_dataset[0].items()})
    print({k: len(v) for k, v in test_dataset[0].items()})
    collator = DataCollatorForSeq2SeqCAR(tokenizer)
    train_loader = DataLoader(train_dataset, 2, collate_fn=collator)
    eval_loader = DataLoader(eval_dataset, 2, collate_fn=collator)
    test_loader = DataLoader(test_dataset, 2, collate_fn=collator)
    for batch in train_loader:
        print({k: v.shape for k, v in batch.items()})
        break
    for batch in eval_loader:
        print({k: v.shape for k, v in batch.items()})
        break
    for batch in test_loader:
        print({k: v.shape for k, v in batch.items()})
        break