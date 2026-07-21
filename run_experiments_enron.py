"""
Experimentos com o segundo dataset: Enron-Spam (e-mails reais, textos bem
mais longos que SMS). Objetivo: verificar se as conclusoes obtidas com o
SMS Spam Collection se sustentam em um dominio com textos de tamanho (n)
muito maior e mais variavel.

Gera:
  results/enron_exp1_full_scan.csv/png       -> tempo/comparacoes por palavra-gatilho (amostra)
  results/enron_exp2_scaling.csv/png         -> tempo x tamanho do e-mail (escala ampla, ate 200k+ chars)
  results/enron_summary.txt
"""

import re
import csv
import time
import random
import statistics as st
from collections import Counter

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from algorithms import brute_force_search, kmp_search

random.seed(42)
OUT_DIR = "results"

TOKEN_RE = re.compile(r"[a-zA-Z]+")
STOPWORDS = {
    "the", "to", "you", "a", "i", "and", "in", "is", "for", "of", "on",
    "your", "have", "it", "that", "my", "me", "are", "be", "will", "with",
    "at", "not", "this", "so", "if", "just", "but", "was", "we", "can",
    "get", "do", "now", "no", "up", "out", "how", "what", "he", "she",
    "his", "her", "or", "as", "im", "its", "an", "when", "from", "u", "n",
    "subject", "com", "www", "http", "https", "enron", "please", "would",
    "all", "our", "us", "am", "by", "re", "any", "here",
}


def tokenize(text):
    return [w.lower() for w in TOKEN_RE.findall(text) if len(w) > 2]


def extract_trigger_words(df, top_n=10, min_count=15):
    spam_counter, ham_counter = Counter(), Counter()
    n_spam = n_ham = 0
    for _, row in df.iterrows():
        text = f"{row['Subject']} {row['Message']}"
        tokens = set(tokenize(text)) - STOPWORDS
        if row["Spam/Ham"] == "spam":
            n_spam += 1
            spam_counter.update(tokens)
        else:
            n_ham += 1
            ham_counter.update(tokens)

    scores = []
    for word, count in spam_counter.items():
        if count < min_count:
            continue
        diff = count / n_spam - ham_counter.get(word, 0) / n_ham
        scores.append((word, diff))
    scores.sort(key=lambda x: x[1], reverse=True)
    return [w for w, _ in scores[:top_n]], n_spam, n_ham


print("Carregando Enron-Spam dataset...")
df = pd.read_csv("enron_spam_data.csv")
df = df.dropna(subset=["Message"])
df["Message"] = df["Message"].astype(str)
df["Subject"] = df["Subject"].fillna("").astype(str)
df["full_text"] = df["Subject"] + " " + df["Message"]
df["length"] = df["full_text"].str.len()
print(f"{len(df)} e-mails carregados (apos remover nulos)")
print(df["length"].describe())

keywords, n_spam, n_ham = extract_trigger_words(df, top_n=10)
print("Palavras-gatilho (Enron, data-driven):", keywords)

# ---------------------------------------------------------------------
# EXPERIMENTO Enron-1: varredura em uma amostra (dataset completo seria
# custoso demais dado o tamanho de alguns e-mails, ate 228 mil caracteres)
# ---------------------------------------------------------------------
print("\n[Enron Exp. 1] Varredura em amostra estratificada...")
sample = df.groupby("Spam/Ham", group_keys=False).apply(
    lambda g: g.sample(min(1000, len(g)), random_state=42)
)
texts = sample["full_text"].tolist()
print(f"Amostra: {len(texts)} e-mails, {sum(len(t) for t in texts):,} caracteres no total")

exp1_rows = []
for kw in keywords:
    bf_time = kmp_time = 0.0
    bf_cmp = kmp_cmp = 0
    for text in texts:
        t0 = time.perf_counter()
        _, c = brute_force_search(text, kw)
        bf_time += time.perf_counter() - t0
        bf_cmp += c

        t0 = time.perf_counter()
        _, c = kmp_search(text, kw)
        kmp_time += time.perf_counter() - t0
        kmp_cmp += c

    exp1_rows.append({
        "keyword": kw, "bf_time_s": bf_time, "kmp_time_s": kmp_time,
        "bf_comparisons": bf_cmp, "kmp_comparisons": kmp_cmp,
    })
    print(f"  '{kw}': BF={bf_time*1000:.1f}ms ({bf_cmp:,} cmp)  KMP={kmp_time*1000:.1f}ms ({kmp_cmp:,} cmp)")

with open(f"{OUT_DIR}/enron_exp1_full_scan.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=exp1_rows[0].keys())
    w.writeheader()
    w.writerows(exp1_rows)

fig, ax = plt.subplots(figsize=(10, 5))
x = range(len(keywords))
width = 0.35
ax.bar([i - width/2 for i in x], [r["bf_time_s"]*1000 for r in exp1_rows], width, label="Forca Bruta")
ax.bar([i + width/2 for i in x], [r["kmp_time_s"]*1000 for r in exp1_rows], width, label="KMP")
ax.set_xticks(list(x))
ax.set_xticklabels(keywords, rotation=45, ha="right")
ax.set_ylabel("Tempo total (ms) - amostra de 2.000 e-mails")
ax.set_title("Enron-Spam: tempo de busca por palavra-gatilho (textos longos)")
ax.legend()
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/enron_exp1_full_scan.png", dpi=150)
plt.close()

# ---------------------------------------------------------------------
# EXPERIMENTO Enron-2: escala com o tamanho do e-mail (de dezenas a
# centenas de milhares de caracteres) -- amostra espalhada pelo range de
# tamanhos para nao pesar o tempo total de execucao
# ---------------------------------------------------------------------
print("\n[Enron Exp. 2] Tempo em funcao do tamanho do e-mail (escala ampla)...")
df_sorted = df[df["length"] > 0].sort_values("length").reset_index(drop=True)
n_points = 150
idx = [int(i * (len(df_sorted) - 1) / (n_points - 1)) for i in range(n_points)]
scaling_sample = df_sorted.iloc[idx]

fixed_kw = keywords[0]
lens, bf_times, kmp_times = [], [], []
for _, row in scaling_sample.iterrows():
    text = row["full_text"]
    t0 = time.perf_counter()
    brute_force_search(text, fixed_kw)
    bf_times.append((time.perf_counter() - t0) * 1000)

    t0 = time.perf_counter()
    kmp_search(text, fixed_kw)
    kmp_times.append((time.perf_counter() - t0) * 1000)

    lens.append(len(text))

with open(f"{OUT_DIR}/enron_exp2_scaling.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["tamanho_email", "bf_tempo_ms", "kmp_tempo_ms"])
    w.writerows(zip(lens, bf_times, kmp_times))

fig, ax = plt.subplots(figsize=(9, 5))
ax.scatter(lens, bf_times, s=12, alpha=0.5, label="Forca Bruta")
ax.scatter(lens, kmp_times, s=12, alpha=0.5, label="KMP")
ax.set_xlabel("Tamanho do e-mail (caracteres)")
ax.set_ylabel(f"Tempo de busca (ms), padrao '{fixed_kw}'")
ax.set_title("Enron-Spam: tempo de busca x tamanho do e-mail (ate ~200 mil caracteres)")
ax.legend()
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/enron_exp2_scaling.png", dpi=150)
plt.close()

# versao log-log para melhor visualizar a relacao linear em ampla escala
fig, ax = plt.subplots(figsize=(9, 5))
ax.scatter(lens, bf_times, s=12, alpha=0.5, label="Forca Bruta")
ax.scatter(lens, kmp_times, s=12, alpha=0.5, label="KMP")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("Tamanho do e-mail, escala log (caracteres)")
ax.set_ylabel("Tempo de busca, escala log (ms)")
ax.set_title("Enron-Spam: tempo x tamanho do e-mail (escala log-log)")
ax.legend()
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/enron_exp2_scaling_loglog.png", dpi=150)
plt.close()

# ---------------------------------------------------------------------
# RESUMO
# ---------------------------------------------------------------------
total_bf = sum(r["bf_time_s"] for r in exp1_rows)
total_kmp = sum(r["kmp_time_s"] for r in exp1_rows)
biggest_email_len = int(df["length"].max())

summary = f"""RESUMO -- SEGUNDO DATASET (ENRON-SPAM)
========================================
Dataset: Enron-Spam (Metsis, Androutsopoulos e Paliouras, 2006)
  Fonte: https://github.com/MWiechmann/enron_spam_data
  {len(df)} e-mails totais ({n_spam} spam, {n_ham} ham no calculo de palavras-gatilho)
  Tamanho medio do e-mail: {df['length'].mean():.0f} caracteres (mediana {df['length'].median():.0f})
  Maior e-mail do dataset: {biggest_email_len:,} caracteres

Palavras-gatilho (selecao data-driven especifica deste dataset): {', '.join(keywords)}

[Enron Exp. 1] Amostra de {len(texts)} e-mails x {len(keywords)} palavras-gatilho:
  Tempo total Forca Bruta: {total_bf*1000:.1f} ms
  Tempo total KMP:         {total_kmp*1000:.1f} ms
  Razao BF/KMP: {total_bf/total_kmp:.2f}x

Observacao central: mesmo em textos ordens de magnitude maiores que SMS
(ate {biggest_email_len:,} caracteres), o padrao observado no dataset de
SMS se repete -- o KMP nao mostra vantagem pratica consistente sobre a
Forca Bruta em texto natural em ingles, reforçando que a variavel
determinante nao e o tamanho do texto (n), mas sim o tamanho do alfabeto
e o grau de redundancia/repeticao do conteudo (ver Experimento 6 do
dataset de SMS).
"""

with open(f"{OUT_DIR}/enron_summary.txt", "w") as f:
    f.write(summary)
print("\n" + summary)
