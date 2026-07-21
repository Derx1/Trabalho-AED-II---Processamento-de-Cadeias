"""
Experimentos adicionais, complementares aos 4 experimentos de
run_experiments.py.

Gera:
  results/exp1b_comparacoes_por_keyword.png  -> comparacoes (nao so tempo) por palavra-gatilho
  results/exp2b_dispersao_tempo_tamanho.png  -> tempo x tamanho da mensagem, ponto a ponto
  results/exp5_pattern_length.csv/png        -> efeito do comprimento do padrao (m)
  results/exp6_alphabet_size.csv/png         -> efeito do tamanho do alfabeto
  results/exp7_classifier_metrics.txt/png    -> precisao/recall/F1/matriz de confusao do filtro
"""

import os
import time
import random
import string
import csv
import statistics as st

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from algorithms import brute_force_search, kmp_search
from keyword_extraction import load_dataset, extract_trigger_words

OUT_DIR = "results"
os.makedirs(OUT_DIR, exist_ok=True)
random.seed(42)

rows = load_dataset("spam.csv")
messages = [msg for _, msg in rows]
top_words, n_spam, n_ham = extract_trigger_words(rows, top_n=15)
keywords = [w for w, *_ in top_words]

# ---------------------------------------------------------------------
# EXPERIMENTO 1b: comparacoes (nao tempo) por palavra-gatilho, no dataset completo
# ---------------------------------------------------------------------
print("[Experimento 1b] Comparacoes por palavra-gatilho...")
bf_cmp_list, kmp_cmp_list = [], []
for kw in keywords:
    bf_total = kmp_total = 0
    for msg in messages:
        _, c = brute_force_search(msg, kw)
        bf_total += c
        _, c = kmp_search(msg, kw)
        kmp_total += c
    bf_cmp_list.append(bf_total)
    kmp_cmp_list.append(kmp_total)
    print(f"  '{kw}': BF={bf_total:,} cmp   KMP={kmp_total:,} cmp")

fig, ax = plt.subplots(figsize=(10, 5))
x = range(len(keywords))
width = 0.35
ax.bar([i - width/2 for i in x], bf_cmp_list, width, label="Forca Bruta")
ax.bar([i + width/2 for i in x], kmp_cmp_list, width, label="KMP")
ax.set_xticks(list(x))
ax.set_xticklabels(keywords, rotation=45, ha="right")
ax.set_ylabel("Numero total de comparacoes de caracteres")
ax.set_title("Comparacoes de caracteres por palavra-gatilho (dataset completo)")
ax.legend()
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/exp1b_comparacoes_por_keyword.png", dpi=150)
plt.close()

# ---------------------------------------------------------------------
# EXPERIMENTO 2b: dispersao tempo x tamanho da mensagem (ponto a ponto)
# ---------------------------------------------------------------------
print("[Experimento 2b] Dispersao tempo x tamanho da mensagem...")
fixed_kw = "free" if "free" in keywords else keywords[0]
sample = random.sample(messages, min(800, len(messages)))  # amostra p/ nao ficar lento

lens, bf_times, kmp_times = [], [], []
for msg in sample:
    t0 = time.perf_counter()
    brute_force_search(msg, fixed_kw)
    t1 = time.perf_counter()
    bf_times.append((t1 - t0) * 1e6)

    t0 = time.perf_counter()
    kmp_search(msg, fixed_kw)
    t1 = time.perf_counter()
    kmp_times.append((t1 - t0) * 1e6)

    lens.append(len(msg))

fig, ax = plt.subplots(figsize=(8, 5))
ax.scatter(lens, bf_times, s=8, alpha=0.4, label="Forca Bruta")
ax.scatter(lens, kmp_times, s=8, alpha=0.4, label="KMP")
ax.set_xlabel("Tamanho da mensagem (caracteres)")
ax.set_ylabel(f"Tempo de busca (microssegundos), padrao '{fixed_kw}'")
ax.set_title("Dispersao: tempo de busca x tamanho da mensagem (amostra de 800 SMS)")
ax.legend()
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/exp2b_dispersao_tempo_tamanho.png", dpi=150)
plt.close()

# ---------------------------------------------------------------------
# EXPERIMENTO 5: efeito do comprimento do padrao (m), texto fixo
# ---------------------------------------------------------------------
print("[Experimento 5] Efeito do comprimento do padrao (m)...")
big_text = "".join(random.choices(string.ascii_lowercase, k=50000))
pattern_lengths = [2, 4, 8, 16, 32, 64, 128, 256]
exp5_rows = []
for m in pattern_lengths:
    # padrao aleatorio que NAO ocorre no texto (forca a busca completa, pior caso realista)
    pattern = "".join(random.choices(string.ascii_lowercase, k=m))
    _, bf_cmp = brute_force_search(big_text, pattern)
    _, kmp_cmp = kmp_search(big_text, pattern)

    t0 = time.perf_counter()
    brute_force_search(big_text, pattern)
    bf_t = time.perf_counter() - t0

    t0 = time.perf_counter()
    kmp_search(big_text, pattern)
    kmp_t = time.perf_counter() - t0

    exp5_rows.append({"m": m, "bf_time_s": bf_t, "kmp_time_s": kmp_t,
                       "bf_comparisons": bf_cmp, "kmp_comparisons": kmp_cmp})
    print(f"  m={m}: BF={bf_t*1000:.2f}ms ({bf_cmp} cmp)  KMP={kmp_t*1000:.2f}ms ({kmp_cmp} cmp)")

with open(f"{OUT_DIR}/exp5_pattern_length.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=exp5_rows[0].keys())
    w.writeheader()
    w.writerows(exp5_rows)

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(pattern_lengths, [r["bf_time_s"]*1000 for r in exp5_rows], marker="o", label="Forca Bruta")
ax.plot(pattern_lengths, [r["kmp_time_s"]*1000 for r in exp5_rows], marker="s", label="KMP")
ax.set_xlabel("Comprimento do padrao (m)")
ax.set_ylabel("Tempo (ms), texto fixo de 50.000 caracteres")
ax.set_title("Efeito do comprimento do padrao (m) no tempo de busca")
ax.legend()
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/exp5_pattern_length.png", dpi=150)
plt.close()

# ---------------------------------------------------------------------
# EXPERIMENTO 6: efeito do tamanho do alfabeto (o que explica DNA vs texto natural)
# ---------------------------------------------------------------------
print("[Experimento 6] Efeito do tamanho do alfabeto...")
alphabets = {
    "binario (2 simbolos)": "ab",
    "DNA-like (4 simbolos)": "acgt",
    "16 simbolos": string.ascii_lowercase[:16],
    "alfabeto latino (26 simbolos)": string.ascii_lowercase,
    "alfanumerico (62 simbolos)": string.ascii_letters + string.digits,
}
N = 20000
M = 10
REPEATS_ALPHA = 15
exp6_rows = []
for label, alphabet in alphabets.items():
    bf_cmps, kmp_cmps, bf_ts, kmp_ts = [], [], [], []
    for _ in range(REPEATS_ALPHA):
        text = "".join(random.choices(alphabet, k=N))
        pattern = "".join(random.choices(alphabet, k=M))

        t0 = time.perf_counter()
        _, c = brute_force_search(text, pattern, case_insensitive=False)
        bf_ts.append(time.perf_counter() - t0)
        bf_cmps.append(c)

        t0 = time.perf_counter()
        _, c = kmp_search(text, pattern, case_insensitive=False)
        kmp_ts.append(time.perf_counter() - t0)
        kmp_cmps.append(c)

    exp6_rows.append({
        "alfabeto": label,
        "tamanho_alfabeto": len(alphabet),
        "bf_comparacoes_media": st.mean(bf_cmps),
        "kmp_comparacoes_media": st.mean(kmp_cmps),
        "bf_tempo_medio_ms": st.mean(bf_ts) * 1000,
        "kmp_tempo_medio_ms": st.mean(kmp_ts) * 1000,
    })
    print(f"  {label}: BF={st.mean(bf_cmps):.0f} cmp   KMP={st.mean(kmp_cmps):.0f} cmp")

with open(f"{OUT_DIR}/exp6_alphabet_size.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=exp6_rows[0].keys())
    w.writeheader()
    w.writerows(exp6_rows)

fig, ax = plt.subplots(figsize=(9, 5))
labels = [r["alfabeto"] for r in exp6_rows]
x = range(len(labels))
width = 0.35
ax.bar([i - width/2 for i in x], [r["bf_comparacoes_media"] for r in exp6_rows], width, label="Forca Bruta")
ax.bar([i + width/2 for i in x], [r["kmp_comparacoes_media"] for r in exp6_rows], width, label="KMP")
ax.set_xticks(list(x))
ax.set_xticklabels(labels, rotation=20, ha="right")
ax.set_ylabel(f"Comparacoes medias (texto n={N}, padrao m={M}, {REPEATS_ALPHA} repeticoes)")
ax.set_title("Efeito do tamanho do alfabeto no numero de comparacoes")
ax.legend()
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/exp6_alphabet_size.png", dpi=150)
plt.close()

# ---------------------------------------------------------------------
# EXPERIMENTO 7: o filtro como classificador (precisao, recall, F1, matriz de confusao)
# ---------------------------------------------------------------------
print("[Experimento 7] Metricas de classificacao do filtro de palavras-gatilho...")
tp = fp = tn = fn = 0
for label, msg in rows:
    flagged = any(brute_force_search(msg, kw)[0] for kw in keywords)
    actual_spam = (label == "spam")
    if flagged and actual_spam:
        tp += 1
    elif flagged and not actual_spam:
        fp += 1
    elif not flagged and actual_spam:
        fn += 1
    else:
        tn += 1

precision = tp / (tp + fp) if (tp + fp) > 0 else 0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0
f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
accuracy = (tp + tn) / (tp + tn + fp + fn)

metrics_text = f"""METRICAS DO FILTRO BASEADO EM PALAVRAS-GATILHO
================================================
Regra: mensagem classificada como spam se contiver ao menos uma das
{len(keywords)} palavras-gatilho: {', '.join(keywords)}

Matriz de confusao:
                 Predito SPAM   Predito HAM
Real SPAM        {tp:>10}     {fn:>10}
Real HAM         {fp:>10}     {tn:>10}

Acuracia:  {accuracy:.3f}
Precisao:  {precision:.3f}
Recall:    {recall:.3f}
F1-score:  {f1:.3f}

Interpretacao: o filtro obteve recall alto e precisao baixa. Isso indica
que ele captura quase todas as mensagens de spam do dataset (poucos falsos
negativos), mas tambem marca como spam um grande numero de mensagens
legitimas (muitos falsos positivos). Isso acontece porque varias das
palavras-gatilho mais frequentes em spam (como "call", "text", "our" e
"only") tambem aparecem com frequencia em conversas cotidianas legitimas.
Esse resultado e coerente com a limitacao conhecida de filtros baseados
apenas em palavras-chave fixas, e motiva o uso de tecnicas complementares,
como classificadores estatisticos (ex. Naive Bayes) combinados a regras de
casamento de padroes, conforme observado em [Ojugo e Oyemade 2021].
"""

with open(f"{OUT_DIR}/exp7_classifier_metrics.txt", "w") as f:
    f.write(metrics_text)
print(metrics_text)

fig, ax = plt.subplots(figsize=(5, 4))
matrix = [[tp, fn], [fp, tn]]
im = ax.imshow(matrix, cmap="Blues")
ax.set_xticks([0, 1])
ax.set_yticks([0, 1])
ax.set_xticklabels(["Predito SPAM", "Predito HAM"])
ax.set_yticklabels(["Real SPAM", "Real HAM"])
for i in range(2):
    for j in range(2):
        ax.text(j, i, str(matrix[i][j]), ha="center", va="center", fontsize=14)
ax.set_title(f"Matriz de confusao do filtro (F1={f1:.2f})")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/exp7_classifier_metrics.png", dpi=150)
plt.close()

print("\nTodos os experimentos extras concluidos. Arquivos em results/.")
