"""
Pre-processamento do dataset SMS Spam Collection (UCI) e extracao
data-driven das palavras-gatilho de spam.

Em vez de escolher arbitrariamente uma lista de palavras-gatilho (como e
comum na literatura), este script calcula, para cada palavra do
vocabulario, a diferenca entre sua frequencia relativa em mensagens SPAM
e em mensagens HAM. As palavras com maior diferenca positiva sao as que
mais caracterizam uma mensagem de spam, e sao usadas como padroes de
busca nos experimentos com Forca Bruta e KMP.

Essa abordagem (selecao de padroes orientada por dados, em vez de uma
lista fixa "de senso comum") e um dos elementos de contribuicao/aporte
do trabalho.
"""

import re
import csv
from collections import Counter

STOPWORDS_EN = {
    "the", "to", "you", "a", "i", "and", "in", "is", "for", "of", "on",
    "your", "have", "it", "that", "my", "me", "are", "be", "will", "with",
    "at", "not", "this", "so", "if", "just", "but", "was", "we",
    "can", "get", "do", "now", "no", "up", "out", "how", "what", "he",
    "she", "his", "her", "or", "as", "im", "its", "an", "when", "from",
    "u", "n", "2", "4", "ur",
}

STOPWORDS_PT = {
    "a", "o", "as", "os", "um", "uma", "uns", "umas", "de", "do", "da",
    "dos", "das", "em", "no", "na", "nos", "nas", "para", "por", "com",
    "que", "se", "e", "ou", "é", "foi", "ser", "são", "está", "estão",
    "não", "sim", "seu", "sua", "seus", "suas", "meu", "minha", "meus",
    "minhas", "isso", "esse", "essa", "este", "esta", "você", "voce",
    "ele", "ela", "eles", "elas", "nos", "nós", "mais", "muito", "já",
    "ja", "ainda", "como", "quando", "onde", "porque", "até", "ate",
    "pra", "pro", "aqui", "ali", "lá", "la", "tem", "ter", "vai", "vou",
    "vamos", "hoje", "amanhã", "amanha", "agora", "só", "so", "bem",
    "bom", "boa",
}

# regex inclui letras acentuadas e cedilha, usadas em portugues
TOKEN_RE = re.compile(r"[a-zA-ZÀ-ÖØ-öø-ÿ]+")


def load_dataset(path: str):
    """Carrega o CSV bruto (colunas label/texto) e retorna lista de
    tuplas (label, mensagem). Tenta UTF-8 primeiro (necessario para
    acentuacao em portugues) e cai para latin-1 se falhar (formato do
    dataset original em ingles)."""
    for encoding in ("utf-8", "latin-1"):
        try:
            rows = []
            with open(path, newline="", encoding=encoding) as f:
                reader = csv.reader(f)
                header = next(reader)
                for row in reader:
                    if len(row) < 2:
                        continue
                    label, msg = row[0].strip(), row[1].strip()
                    if label in ("ham", "spam") and msg:
                        rows.append((label, msg))
            return rows
        except UnicodeDecodeError:
            continue
    raise ValueError(f"Nao foi possivel decodificar {path} com utf-8 nem latin-1")


def tokenize(text: str, stopwords=None):
    return [w.lower() for w in TOKEN_RE.findall(text) if len(w) > 2]


def extract_trigger_words(rows, top_n: int = 20, min_count: int = 8, lang: str = "en"):
    stopwords = STOPWORDS_PT if lang == "pt" else STOPWORDS_EN
    spam_counter = Counter()
    ham_counter = Counter()
    n_spam = 0
    n_ham = 0

    for label, msg in rows:
        tokens = set(tokenize(msg)) - stopwords
        if label == "spam":
            n_spam += 1
            spam_counter.update(tokens)
        else:
            n_ham += 1
            ham_counter.update(tokens)

    scores = []
    for word, count in spam_counter.items():
        if count < min_count:
            continue
        freq_spam = count / n_spam
        freq_ham = ham_counter.get(word, 0) / n_ham
        diff = freq_spam - freq_ham
        scores.append((word, diff, count, ham_counter.get(word, 0)))

    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:top_n], n_spam, n_ham


if __name__ == "__main__":
    import sys
    dataset_path = sys.argv[1] if len(sys.argv) > 1 else "spam.csv"
    lang = sys.argv[2] if len(sys.argv) > 2 else "en"

    rows = load_dataset(dataset_path)
    n_total = len(rows)
    n_spam = sum(1 for l, _ in rows if l == "spam")
    n_ham = n_total - n_spam
    print(f"Dataset: {dataset_path} (idioma: {lang})")
    print(f"Total de mensagens: {n_total}")
    print(f"  spam: {n_spam} ({n_spam/n_total:.1%})")
    print(f"  ham:  {n_ham} ({n_ham/n_total:.1%})")

    # datasets pequenos (ex.: corpus de exemplo em PT) precisam de um limiar
    # minimo de ocorrencias mais baixo, senao nenhuma palavra sobrevive ao filtro
    min_count = max(2, round(0.02 * n_spam))
    top_words, ns, nh = extract_trigger_words(rows, top_n=20, min_count=min_count, lang=lang)
    print("\nTop palavras-gatilho (ordenadas por diferenca de frequencia spam-ham):")
    print(f"{'palavra':<12}{'freq_spam':>10}{'freq_ham':>10}{'ocorr_spam':>12}{'ocorr_ham':>12}")
    for word, diff, c_spam, c_ham in top_words:
        print(f"{word:<12}{c_spam/ns:>10.2%}{c_ham/nh:>10.2%}{c_spam:>12}{c_ham:>12}")
