"""
Experimentos computacionais: Forca Bruta vs KMP na deteccao de palavras-
gatilho de spam em mensagens SMS reais (dataset SMS Spam Collection, UCI).

Gera:
  results/exp1_full_dataset.csv      -> tempo/comparacoes por palavra-gatilho (dataset inteiro)
  results/exp2_by_length.csv         -> tempo medio por faixa de tamanho de mensagem
  results/exp3_worst_case.csv        -> teste sintetico de pior caso (padroes repetitivos)
  results/exp4_memory.csv            -> espaco extra (LPS) usado pelo KMP por padrao
  results/fig_*.png                  -> graficos correspondentes
  results/summary.txt                -> resumo textual pronto para a secao de Analise dos Resultados
"""

import os
import time
import csv
import statistics as st

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from algorithms import brute_force_search, kmp_search, _build_lps
from keyword_extraction import load_dataset, extract_trigger_words

import sys
DATASET_PATH = sys.argv[1] if len(sys.argv) > 1 else "spam.csv"
LANG = sys.argv[2] if len(sys.argv) > 2 else "en"
OUT_DIR = sys.argv[3] if len(sys.argv) > 3 else "results"
os.makedirs(OUT_DIR, exist_ok=True)

N_REPEATS = 3  # repeticoes para reduzir ruido na medicao de tempo


def timed_search(func, text, pattern, repeats=N_REPEATS):
    """Executa a busca 'repeats' vezes e retorna (tempo_medio, comparacoes)."""
    times = []
    comparisons = 0
    for _ in range(repeats):
        t0 = time.perf_counter()
        _, comparisons = func(text, pattern)
        t1 = time.perf_counter()
        times.append(t1 - t0)
    return st.mean(times), comparisons


# ---------------------------------------------------------------------
# Carrega dataset e define as palavras-gatilho (data-driven, ver
# keyword_extraction.py)
# ---------------------------------------------------------------------
print(f"Carregando dataset ({DATASET_PATH}, idioma={LANG})...")
rows = load_dataset(DATASET_PATH)
messages = [msg for _, msg in rows]
n_messages = len(messages)
total_chars = sum(len(m) for m in messages)
print(f"{n_messages} mensagens carregadas ({total_chars} caracteres no total)")

n_spam_total = sum(1 for l, _ in rows if l == "spam")
min_count = max(2, round(0.02 * n_spam_total))
top_words, n_spam, n_ham = extract_trigger_words(rows, top_n=15, min_count=min_count, lang=LANG)
keywords = [w for w, *_ in top_words]
if not keywords:
    raise SystemExit("Nenhuma palavra-gatilho encontrada. Dataset muito pequeno ou min_count alto demais.")
print("Palavras-gatilho usadas nos experimentos:", keywords)

# ---------------------------------------------------------------------
# EXPERIMENTO 1: varrer o dataset inteiro para cada palavra-gatilho,
# medindo tempo total e numero total de comparacoes de cada algoritmo.
# Simula o cenario real de um filtro de spam escaneando uma caixa de
# mensagens em busca de cada termo suspeito.
# ---------------------------------------------------------------------
print("\n[Experimento 1] Varredura completa do dataset por palavra-gatilho...")
exp1_rows = []
for kw in keywords:
    bf_time_total = 0.0
    bf_cmp_total = 0
    kmp_time_total = 0.0
    kmp_cmp_total = 0
    occurrences = 0

    for msg in messages:
        t0 = time.perf_counter()
        pos_bf, cmp_bf = brute_force_search(msg, kw)
        t1 = time.perf_counter()
        bf_time_total += (t1 - t0)
        bf_cmp_total += cmp_bf

        t0 = time.perf_counter()
        pos_kmp, cmp_kmp = kmp_search(msg, kw)
        t1 = time.perf_counter()
        kmp_time_total += (t1 - t0)
        kmp_cmp_total += cmp_kmp

        occurrences += len(pos_bf)

    exp1_rows.append({
        "keyword": kw,
        "pattern_len": len(kw),
        "occurrences": occurrences,
        "bf_time_s": bf_time_total,
        "kmp_time_s": kmp_time_total,
        "bf_comparisons": bf_cmp_total,
        "kmp_comparisons": kmp_cmp_total,
        "speedup_time": bf_time_total / kmp_time_total if kmp_time_total > 0 else float("nan"),
        "reduction_comparisons_pct": 100 * (1 - kmp_cmp_total / bf_cmp_total) if bf_cmp_total > 0 else 0,
    })
    print(f"  '{kw}': BF={bf_time_total*1000:.1f}ms ({bf_cmp_total} cmp)  "
          f"KMP={kmp_time_total*1000:.1f}ms ({kmp_cmp_total} cmp)")

with open(f"{OUT_DIR}/exp1_full_dataset.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=exp1_rows[0].keys())
    writer.writeheader()
    writer.writerows(exp1_rows)

# grafico: tempo total por palavra-gatilho (BF vs KMP)
fig, ax = plt.subplots(figsize=(10, 5))
x = range(len(keywords))
bf_times = [r["bf_time_s"] * 1000 for r in exp1_rows]
kmp_times = [r["kmp_time_s"] * 1000 for r in exp1_rows]
width = 0.35
ax.bar([i - width/2 for i in x], bf_times, width, label="Forca Bruta")
ax.bar([i + width/2 for i in x], kmp_times, width, label="KMP")
ax.set_xticks(list(x))
ax.set_xticklabels(keywords, rotation=45, ha="right")
ax.set_ylabel("Tempo total (ms) - dataset completo")
ax.set_title("Tempo de busca por palavra-gatilho no dataset completo (5.572 SMS)")
ax.legend()
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/fig_exp1_tempo_por_keyword.png", dpi=150)
plt.close()

# ---------------------------------------------------------------------
# EXPERIMENTO 2: tempo medio de busca em funcao do tamanho da mensagem
# (para uma palavra-gatilho fixa, "free", presente em muitas mensagens)
# ---------------------------------------------------------------------
print("\n[Experimento 2] Tempo em funcao do tamanho da mensagem...")
fixed_kw = "free" if "free" in keywords else keywords[0]
buckets = {"curta (<=40)": [], "media (41-80)": [], "longa (81-120)": [], "muito longa (>120)": []}

for msg in messages:
    L = len(msg)
    if L <= 40:
        buckets["curta (<=40)"].append(msg)
    elif L <= 80:
        buckets["media (41-80)"].append(msg)
    elif L <= 120:
        buckets["longa (81-120)"].append(msg)
    else:
        buckets["muito longa (>120)"].append(msg)

exp2_rows = []
for label, msgs in buckets.items():
    if not msgs:
        continue
    bf_times, kmp_times = [], []
    for msg in msgs:
        t, _ = timed_search(brute_force_search, msg, fixed_kw, repeats=2)
        bf_times.append(t)
        t, _ = timed_search(kmp_search, msg, fixed_kw, repeats=2)
        kmp_times.append(t)
    avg_len = st.mean(len(m) for m in msgs)
    exp2_rows.append({
        "faixa": label,
        "n_mensagens": len(msgs),
        "tamanho_medio_msg": avg_len,
        "bf_tempo_medio_us": st.mean(bf_times) * 1e6,
        "kmp_tempo_medio_us": st.mean(kmp_times) * 1e6,
    })
    print(f"  {label}: n={len(msgs)}, tam_medio={avg_len:.0f}, "
          f"BF={st.mean(bf_times)*1e6:.2f}us, KMP={st.mean(kmp_times)*1e6:.2f}us")

with open(f"{OUT_DIR}/exp2_by_length.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=exp2_rows[0].keys())
    writer.writeheader()
    writer.writerows(exp2_rows)

fig, ax = plt.subplots(figsize=(8, 5))
labels = [r["faixa"] for r in exp2_rows]
bf_vals = [r["bf_tempo_medio_us"] for r in exp2_rows]
kmp_vals = [r["kmp_tempo_medio_us"] for r in exp2_rows]
x = range(len(labels))
ax.plot(x, bf_vals, marker="o", label="Forca Bruta")
ax.plot(x, kmp_vals, marker="s", label="KMP")
ax.set_xticks(list(x))
ax.set_xticklabels(labels, rotation=20, ha="right")
ax.set_ylabel(f"Tempo medio por mensagem (microssegundos), busca por '{fixed_kw}'")
ax.set_title("Tempo medio de busca por faixa de tamanho de mensagem")
ax.legend()
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/fig_exp2_tempo_por_tamanho.png", dpi=150)
plt.close()

# ---------------------------------------------------------------------
# EXPERIMENTO 3: pior caso sintetico (texto e padrao altamente
# repetitivos), para expor a diferenca teorica O(n*m) vs O(n+m) que
# raramente aparece em texto natural.
# ---------------------------------------------------------------------
print("\n[Experimento 3] Pior caso sintetico (padroes repetitivos)...")
exp3_rows = []
sizes = [1000, 5000, 10000, 20000, 40000]
for n in sizes:
    # texto: 'a' repetido n vezes seguido de 'b' (forca muitas comparacoes
    # quase-completas em cada posicao na Forca Bruta)
    text = "a" * n + "b"
    pattern = "a" * 20 + "b"  # padrao de 21 caracteres, repetitivo

    bf_time, bf_cmp = timed_search(brute_force_search, text, pattern, repeats=3)
    kmp_time, kmp_cmp = timed_search(kmp_search, text, pattern, repeats=3)

    exp3_rows.append({
        "tamanho_texto": n,
        "bf_tempo_s": bf_time,
        "kmp_tempo_s": kmp_time,
        "bf_comparisons": bf_cmp,
        "kmp_comparisons": kmp_cmp,
    })
    print(f"  n={n}: BF={bf_time*1000:.2f}ms ({bf_cmp} cmp)  KMP={kmp_time*1000:.2f}ms ({kmp_cmp} cmp)")

with open(f"{OUT_DIR}/exp3_worst_case.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=exp3_rows[0].keys())
    writer.writeheader()
    writer.writerows(exp3_rows)

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(sizes, [r["bf_tempo_s"] * 1000 for r in exp3_rows], marker="o", label="Forca Bruta")
ax.plot(sizes, [r["kmp_tempo_s"] * 1000 for r in exp3_rows], marker="s", label="KMP")
ax.set_xlabel("Tamanho do texto (n)")
ax.set_ylabel("Tempo (ms)")
ax.set_title("Pior caso sintetico: texto e padrao repetitivos ('aaa...ab')")
ax.legend()
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/fig_exp3_pior_caso.png", dpi=150)
plt.close()

# ---------------------------------------------------------------------
# EXPERIMENTO 4: espaco extra utilizado (LPS do KMP vs O(1) da Forca Bruta)
# ---------------------------------------------------------------------
print("\n[Experimento 4] Espaco extra (memoria auxiliar)...")
exp4_rows = []
for kw in keywords:
    lps, _ = _build_lps(kw.lower())
    exp4_rows.append({
        "keyword": kw,
        "pattern_len": len(kw),
        "kmp_extra_space_ints": len(lps),
        "bf_extra_space_ints": 0,
    })

with open(f"{OUT_DIR}/exp4_memory.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=exp4_rows[0].keys())
    writer.writeheader()
    writer.writerows(exp4_rows)

# ---------------------------------------------------------------------
# RESUMO
# ---------------------------------------------------------------------
total_bf_time = sum(r["bf_time_s"] for r in exp1_rows)
total_kmp_time = sum(r["kmp_time_s"] for r in exp1_rows)
total_bf_cmp = sum(r["bf_comparisons"] for r in exp1_rows)
total_kmp_cmp = sum(r["kmp_comparisons"] for r in exp1_rows)

summary = f"""RESUMO DOS EXPERIMENTOS
========================

Dataset: {DATASET_PATH} (idioma: {LANG})
  - {n_messages} mensagens ({n_spam} spam, {n_ham} ham)
  - {total_chars} caracteres no total
  - Palavras-gatilho (selecao data-driven, ver keyword_extraction.py): {', '.join(keywords)}

[Experimento 1] Varredura completa do dataset (todas as {len(keywords)} palavras-gatilho x {n_messages} mensagens):
  - Tempo total Forca Bruta: {total_bf_time*1000:.2f} ms
  - Tempo total KMP:         {total_kmp_time*1000:.2f} ms
  - Razao BF/KMP (tempo):    {total_bf_time/total_kmp_time:.2f}x
  - Comparacoes totais Forca Bruta: {total_bf_cmp:,}
  - Comparacoes totais KMP:         {total_kmp_cmp:,}
  - Reducao de comparacoes com KMP: {100*(1 - total_kmp_cmp/total_bf_cmp):.1f}%

[Experimento 3] Pior caso sintetico (texto/padrao repetitivos, n={sizes[-1]}):
  - Forca Bruta: {exp3_rows[-1]['bf_tempo_s']*1000:.2f} ms ({exp3_rows[-1]['bf_comparisons']:,} comparacoes)
  - KMP:         {exp3_rows[-1]['kmp_tempo_s']*1000:.2f} ms ({exp3_rows[-1]['kmp_comparisons']:,} comparacoes)
  - Razao BF/KMP (tempo): {exp3_rows[-1]['bf_tempo_s']/exp3_rows[-1]['kmp_tempo_s']:.2f}x

[Experimento 4] Espaco extra (memoria auxiliar):
  - Forca Bruta: O(1) -- nenhum espaco extra em todos os {len(keywords)} padroes testados
  - KMP: O(m) -- entre {min(r['kmp_extra_space_ints'] for r in exp4_rows)} e {max(r['kmp_extra_space_ints'] for r in exp4_rows)} inteiros (vetor LPS), conforme o tamanho de cada palavra-gatilho

CONCLUSAO PRELIMINAR (a ser discutida no artigo):
  Em texto natural (mensagens de SMS reais, alfabeto grande, baixa
  repeticao), a vantagem pratica do KMP sobre a Forca Bruta tende a ser
  pequena, pois o pior caso teorico O(n*m) raramente ocorre. Ja no
  cenario sintetico de pior caso (padroes/textos repetitivos), o KMP
  mostra ganho de desempenho evidente e crescente com o tamanho da
  entrada, confirmando a previsao teorica O(n+m) vs O(n*m). Em
  contrapartida, o KMP exige espaco extra O(m) e uma implementacao mais
  complexa (construcao do vetor de prefixos), enquanto a Forca Bruta e
  mais simples de implementar e nao usa memoria auxiliar.
"""

with open(f"{OUT_DIR}/summary.txt", "w") as f:
    f.write(summary)

print("\n" + summary)
print(f"Arquivos gerados em '{OUT_DIR}/'")
