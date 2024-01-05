import re
import time
import json
from argparse import ArgumentParser
from pathlib import Path
from typing import Dict, Tuple, Type, Callable
from functools import partial

import torch
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
from transformers import EarlyStoppingCallback
from transformers.trainer_utils import EvalPrediction
from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils import PreTrainedTokenizer
from transformers.trainer_seq2seq import Seq2SeqTrainer
from transformers.trainer_utils import IntervalStrategy
from transformers.training_args_seq2seq import Seq2SeqTrainingArguments
from transformers.data.data_collator import DataCollator
from transformers.models.mbart.configuration_mbart import MBartConfig
from transformers.models.mbart.modeling_mbart import MBartForConditionalGeneration
from transformers.models.mbart50.tokenization_mbart50 import MBart50Tokenizer

from data_utils import Seq2SeqDataset, DataCollatorForSeq2SeqCAR, DataCollatorForSeq2SeqPara
from modeling import MBartForConditionalGenerationCAR
from modeling_mt import MBartForParallelGeneration, MBartForParallelGenerationCAR
from rouge_score.rouge_scorer import RougeScorer


# tokenizer, data collator, model, attention type
LANG_MAP = json.load(open("lang_map.json"))
MODELS: Dict[str, Tuple[Type[PreTrainedTokenizer], Type[DataCollator], Type[PreTrainedModel], Type[Seq2SeqDataset], Type[PretrainedConfig]]] = {
    "MBart": (MBart50Tokenizer, DataCollatorForSeq2SeqCAR, MBartForConditionalGeneration, Seq2SeqDataset, MBartConfig),
    "MBart-car": (MBart50Tokenizer, DataCollatorForSeq2SeqCAR, MBartForConditionalGenerationCAR, Seq2SeqDataset, MBartConfig),
    "MBart-mt": (MBart50Tokenizer, DataCollatorForSeq2SeqPara, MBartForParallelGeneration, partial(Seq2SeqDataset, multi_task=True), MBartConfig),
    "MBart-mt-car":  (MBart50Tokenizer, DataCollatorForSeq2SeqPara, MBartForParallelGenerationCAR, partial(Seq2SeqDataset, multi_task=True), MBartConfig),
}

def instantiate(args, train=True):
    tokenizer_cls, collator_cls, model_cls, dataset_cls, config_cls = MODELS[args.model_type]
    dataset_cls = partial(dataset_cls, no_mbart_lang_token=args.no_mbart_lang_token)
    config = config_cls.from_pretrained(args.model_name_or_path)
    tokenizer = tokenizer_cls.from_pretrained(
        args.model_name_or_path,
        src_lang=args.src_lang,
        tgt_lang=args.tgt_lang,
    )
    if args.model_type.endswith("-car"):
        model = model_cls.from_pretrained(
            args.model_name_or_path,
            config=config,
            tokenizer=tokenizer,
            src_lang=LANG_MAP[args.src_lang],
            pg_weight=args.pg_weight if train else 1.0,
            self_critic=(not args.no_self_critic) if train else False,
        )
    else:
        if args.model_type.endswith("-mt"):
            model = model_cls.from_pretrained(args.model_name_or_path, config=config, pg_weight=args.pg_weight if train else 1.0)
        else:
            model = model_cls.from_pretrained(args.model_name_or_path, config=config)
    if model_cls == MBartForParallelGeneration:
        model.model.decoder_para.load_state_dict(model.model.decoder.state_dict())
    if args.model_type.endswith("-car") or args.model_type.endswith("-mt"):
        dataset = dataset_cls(tokenizer, args.src_file, args.tgt_file, args.ml_tgt_file, args.max_src_len, args.max_tgt_len)
    else:
        dataset = dataset_cls(tokenizer, args.src_file, args.tgt_file, None, args.max_src_len, args.max_tgt_len)
    collator = collator_cls(tokenizer)
    
    if args.model_recover_path:
        state_dict = torch.load(args.model_recover_path)
        if args.model_type.startswith("MT5"):
            state_dict["encoder.embed_tokens.weight"] = state_dict["decoder.embed_tokens.weight"]
            model.load_state_dict(state_dict, strict=False)
        else:
            model.load_state_dict(state_dict)
    
    if train:
        eval_dataset = dataset_cls(tokenizer, args.eval_src_file, args.eval_tgt_file, None, args.max_src_len, args.max_tgt_len)
        return tokenizer, collator, model, dataset, eval_dataset
    return tokenizer, collator, model, dataset


def compute_metrics_factory(args, tokenizer: PreTrainedTokenizer) -> Callable[[EvalPrediction], Dict]:
    rouge_scorer = RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True, lang=LANG_MAP[args.tgt_lang])
    def compute_metrics(eval_pred: EvalPrediction):
        output_ids = eval_pred.predictions
        label_ids = eval_pred.label_ids
        output_ids[(output_ids < 0)|(output_ids >= tokenizer.vocab_size)] = tokenizer.pad_token_id
        label_ids[label_ids == -100] = tokenizer.pad_token_id
        hypothesis = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        references = tokenizer.batch_decode(label_ids, skip_special_tokens=True)
        total_score = {"rouge-1": 0, "rouge-2": 0, "rouge-l": 0}
        for hyp, ref in zip(hypothesis, references):
            total_score["rouge-1"] += rouge_scorer.score(ref, hyp)["rouge1"].fmeasure
            total_score["rouge-2"] += rouge_scorer.score(ref, hyp)["rouge2"].fmeasure
            total_score["rouge-l"] += rouge_scorer.score(ref, hyp)["rougeL"].fmeasure
        mean_score = { k: v / len(references) for k, v in total_score.items() }
        mean_score["mean_score"] = (mean_score["rouge-1"] + mean_score["rouge-2"] + mean_score["rouge-l"]) / 3
        return mean_score
    return compute_metrics


def train(args):
    tokenizer, collator, model, dataset, eval_dataset = instantiate(args, train=True)
    n_gpus = torch.cuda.device_count()
    if n_gpus > 0:
        args.batch_size //= n_gpus
    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        do_train=True,
        do_eval=not args.no_eval,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.lr,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        no_cuda=args.no_cuda,
        fp16=args.fp16,
        save_strategy=IntervalStrategy.EPOCH if args.save_strategy == "epoch" else IntervalStrategy.STEPS,
        save_steps=500 if args.save_strategy == "epoch" else int(args.save_strategy),
        save_total_limit=1,
        logging_strategy=IntervalStrategy.STEPS,
        logging_steps=1,
        evaluation_strategy=(IntervalStrategy.EPOCH if args.save_strategy == "epoch" else IntervalStrategy.STEPS) if not args.no_eval else "no",
        eval_steps=(500 if args.save_strategy == "epoch" else int(args.save_strategy)) if not args.no_eval else None,
        metric_for_best_model="eval_mean_score",
        lr_scheduler_type=args.lr_scheduler,
        warmup_ratio=args.warmup_ratio,
        local_rank=args.local_rank,
        remove_unused_columns=False,
        seed=args.seed,
        predict_with_generate=True,
        generation_max_length=args.max_tgt_len,
        load_best_model_at_end=True,
    )
    callbacks = [EarlyStoppingCallback(args.early_stopping_patience)] if args.use_early_stop else []
    # callbacks += [MixerCallback(args.pg_weight)]
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        data_collator=collator,
        train_dataset=dataset,
        eval_dataset=eval_dataset if not args.no_eval else None,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics_factory(args, tokenizer),
        callbacks=callbacks,
    )
    resume = False
    if list(Path(args.output_dir).glob("checkpoint-*")):
        resume = True
    trainer.train(resume_from_checkpoint=resume)


def decode(args):
    latency = 0.
    N = 0
    decode_out_file = args.decode_out_file
    if decode_out_file is None:
        decode_out_file = f"{args.model_recover_path}.decode.txt" if args.model_recover_path else "decode.txt"
    
    device = "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"
    tokenizer, collator, model, dataset = instantiate(args, train=False)
    if args.fp16:
        model.half()
    model.to(device)
    dataloader = DataLoader(dataset, args.batch_size, collate_fn=collator)
    output_texts = []
    for batch in tqdm(dataloader):
        batch = { k: v.to(device) for k, v in batch.items() }
        if "labels" in batch: del batch["labels"]
        if "ml_labels" in batch: del batch["ml_labels"]
        s = time.time()
        output = model.generate(
            **batch,
            max_new_tokens=args.max_tgt_len,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            num_beams=args.num_beams,
        )
        d = time.time() - s
        N += 1
        latency += (d - latency) / N
        for i in range(0, len(output), 1):
            output_buffer = []
            for output_ids in output[i:i+1]:
                output_text = tokenizer.decode(output_ids).strip()
                output_text = output_text.split(args.tgt_lang)[-1].strip()
                output_text = output_text.replace(tokenizer.pad_token, "").strip()
                output_text = re.sub(r"\s+", " ", output_text)
                output_buffer.append(output_text)
            output_texts.append("\t".join(output_buffer))
    with open(decode_out_file, "w") as fout:
        fout.writelines([line + "\n" for line in output_texts])
    print(f"Inference Latency: {int(latency*1000)}ms")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("task", type=str, default="train")
    parser.add_argument("--model_type", type=str, default="bart")
    parser.add_argument("--model_name_or_path", type=str, default="bart-base")
    parser.add_argument("--model_recover_path", type=str, default=None)
    parser.add_argument("--no_cuda", action="store_true")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--eval_batch_size", type=int, default=32)
    parser.add_argument("--src_file", type=str, default="cnndm-10k/train.src")
    parser.add_argument("--tgt_file", type=str, default=None)
    parser.add_argument("--ml_tgt_file", type=str, default=None)
    parser.add_argument("--src_lang", type=str, default="en_XX")
    parser.add_argument("--tgt_lang", type=str, default="en_XX")
    parser.add_argument("--no_mbart_lang_token", action="store_true")
    parser.add_argument("--no_eval", action="store_true")
    parser.add_argument("--max_src_len", type=int, default=768)
    parser.add_argument("--max_tgt_len", type=int, default=256)
    parser.add_argument("--mask_prob", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fp16", action="store_true")
    args, _ = parser.parse_known_args()
    if args.task == "train":
        parser.add_argument("--local_rank", type=int, default=-1)
        parser.add_argument("--eval_src_file", type=str, default=None)
        parser.add_argument("--eval_tgt_file", type=str, default=None)
        parser.add_argument("--output_dir", type=str, default="output_dir")
        parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
        parser.add_argument("--lr", type=float, default=3e-5)
        parser.add_argument("--num_train_epochs", type=int, default=30)
        parser.add_argument("--max_steps", type=int, default=-1)
        parser.add_argument("--save_strategy", type=str, default="epoch")
        parser.add_argument("--pg_weight", type=float, default=0.1)
        parser.add_argument("--no_self_critic", action="store_true")
        parser.add_argument("--use_early_stop", action="store_true")
        parser.add_argument("--early_stopping_patience", type=int, default=3)
        parser.add_argument("--lr_scheduler", type=str, default="linear")
        parser.add_argument("--warmup_ratio", type=float, default=0.0)
        args = parser.parse_args()
        train(args)
    elif args.task == "decode":
        parser.add_argument("--num_beams", type=int, default=1)
        parser.add_argument("--decode_out_file", type=str, default=None)
        args = parser.parse_args()
        decode(args)