import json
import re
import os


def proc(data, split):
    processed = {}
    for item in data:
        for lang in item["other_languages"]:
            src_file = item["other_languages"][lang] + ".md"
            src_file = f"data/globalvoices_v0/tokenized/{lang}/{src_file}"
            article = open(src_file).read().strip()
            article = re.sub(r"\s+", " ", article)
            summary = item["summary"]
            summary = re.sub(r"\s+", " ", summary)
            if lang not in processed:
                processed[lang] = []
            processed[lang].append((article, summary))
    for lang in processed:
        path = f"data/globalvoices_v0_proc/{lang}/"
        os.makedirs(path, exist_ok=True)
        with open(f"{path}{split}.src", "w") as fsrc, open(f"{path}{split}.tgt", "w") as ftgt:
            for article, summary in processed[lang]:
                fsrc.write(f"{article}\n")
                ftgt.write(f"{summary}\n")


if __name__ == "__main__":
    raw = json.load(open("data/globalvoices_v0/gv_crowd.json")) + json.load(open("data/globalvoices_v0/gv_snippet.json"))
    N = len(raw)
    raw_train = raw[:int(N*0.8)]
    raw_test = raw[int(N*0.8):]
    raw_eval = raw_test[:len(raw_test)//2]
    raw_test = raw_test[len(raw_test)//2:]
    proc(raw_train, "train")
    proc(raw_eval, "valid")
    proc(raw_test, "test")
