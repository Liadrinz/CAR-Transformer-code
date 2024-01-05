import torch
import matplotlib.pyplot as plt

from transformers.models.mbart.modeling_mbart import MBartForConditionalGeneration
from transformers.models.mbart50.tokenization_mbart50 import MBart50Tokenizer


def visualize(source: str, target: str, name: str, tokenizer: MBart50Tokenizer):
    inputs = tokenizer(source, text_target=target, return_tensors="pt")
    input_ids = inputs["input_ids"]
    labels = inputs["labels"]
    outputs = model.forward(**inputs, output_attentions=True)
    cross_attention_weights = outputs.cross_attentions[-1].mean(dim=1)[0]
    plt.figure(dpi=100, figsize=(8, 8))
    plt.imshow(cross_attention_weights[1:-1, 1:-1].detach().numpy(), cmap="Blues")
    plt.axis("off")
    for i, tk in enumerate(input_ids[0, 1:-1]):
        plt.text(i, -0.75, tokenizer.decode(tk), rotation=90, fontsize=12, horizontalalignment="right", verticalalignment="center")
    for i, tk in enumerate(labels[0, 1:-1]):
        plt.text(-0.75, i, tokenizer.decode(tk), rotation=0, fontsize=12, horizontalalignment="right", verticalalignment="center")
    plt.savefig(f"cross-attention-{name}.png")
    plt.cla()
    source_awareness = cross_attention_weights[1:-1, 1:-1].max(dim=0).values
    plt.bar([tokenizer.decode(tk) for tk in input_ids[0, 1:-1]], source_awareness.detach().numpy() * 10)
    plt.xticks(rotation=90)
    plt.savefig(f"source-awareness-{name}.png")

if __name__ == "__main__":
    model = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-50")
    src_lang = "en_XX"
    tgt_langs = ["de_DE", "fr_XX", "es_XX", "it_IT"]
    source = "I went to Japan last week then"
    targets = ["Ich war letzte Woche in Japan", "Je suis allé au Japon la semaine dernière", "fui a japon la semana pasada", "Sono andato in Giappone la scorsa settimana"]
    for tgt_lang, target in zip(tgt_langs, targets):
        tokenizer = MBart50Tokenizer.from_pretrained("facebook/mbart-large-50", src_lang=src_lang, tgt_lang=tgt_lang)
        visualize(source, target, f"{src_lang}-{tgt_lang}", tokenizer)
