from argparse import ArgumentParser
from rouge_score.rouge_scorer import RougeScorer


scorer = RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True, lang="chinese")
parser = ArgumentParser()
parser.add_argument("ref_file", type=str)
parser.add_argument("hyp_file", type=str)
args = parser.parse_args()
with open(args.ref_file, "r") as fin:
    refs = [line.strip() for line in fin]
with open(args.hyp_file, "r") as fin:
    hyps = [line.strip() for line in fin]
scores = [scorer.score(ref, hyp) for ref, hyp in zip(refs, hyps)]
N = len(scores)
avg_r1 = sum([score["rouge1"].fmeasure for score in scores]) / N
avg_r2 = sum([score["rouge2"].fmeasure for score in scores]) / N
avg_rl = sum([score["rougeL"].fmeasure for score in scores]) / N
print(f"{avg_r1:.4f}/{avg_r2:.4f}/{avg_rl:.4f}")
