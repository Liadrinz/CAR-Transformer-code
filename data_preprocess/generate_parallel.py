import os, sys

import ray
import math

from tqdm import trange
from typing import List
from transformers import MT5ForConditionalGeneration, T5Tokenizer


batch_size = 8
num_gpus = 4
num_workers = 4

ray.init(num_gpus=num_gpus)

model_name = "csebuetnlp/mT5_multilingual_XLSum"

@ray.remote(num_gpus=num_gpus/num_workers)
def summarize(articles: List[str]):
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = MT5ForConditionalGeneration.from_pretrained(model_name)
    model.cuda()
    summaries = []
    for i in trange(0, len(articles), batch_size):
        batch = articles[i:i+batch_size]
        batch = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=1024)
        batch = {k: v.cuda() for k, v in batch.items()}
        output_ids = model.generate(**batch, max_new_tokens=128)
        summaries.extend(tokenizer.batch_decode(output_ids, skip_special_tokens=True))
    return summaries


with open(sys.argv[1], "r") as fin:
    articles = [line.strip() for line in fin]
N = len(articles)
num_per_worker = math.ceil(N / num_workers)
refs = []
for i in range(num_workers):
    ref = summarize.remote(articles[i*num_per_worker:(i+1)*num_per_worker])
    refs.append(ref)
summaries = []
for ref in refs:
    summaries.extend(ray.get(ref))
with open(sys.argv[2], "w") as fout:
    fout.writelines([summary.strip()+"\n" for summary in summaries])
