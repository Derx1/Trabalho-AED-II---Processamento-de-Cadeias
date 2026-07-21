"""
Implementacoes instrumentadas dos algoritmos de casamento de padroes:
- Forca Bruta (Naive String Matching)
- Knuth-Morris-Pratt (KMP)

Cada funcao de busca retorna:
- lista de posicoes (indices) onde o padrao ocorre no texto
- numero total de comparacoes de caracteres realizadas (metrica teorica)

As implementacoes sao case-insensitive por padrao (mensagens de SMS/spam
tipicamente misturam maiusculas/minusculas), mas isso pode ser desativado.
"""

from typing import List, Tuple


def brute_force_search(text: str, pattern: str, case_insensitive: bool = True) -> Tuple[List[int], int]:
    """
    Busca por 'pattern' em 'text' usando o algoritmo de Forca Bruta (naive).

    Complexidade de tempo: O(n*m) no pior caso, O(n) no melhor caso.
    Complexidade de espaco extra: O(1).
    """
    if case_insensitive:
        text = text.lower()
        pattern = pattern.lower()

    n = len(text)
    m = len(pattern)
    positions = []
    comparisons = 0

    if m == 0 or n < m:
        return positions, comparisons

    for i in range(n - m + 1):
        j = 0
        while j < m:
            comparisons += 1
            if text[i + j] != pattern[j]:
                break
            j += 1
        if j == m:
            positions.append(i)

    return positions, comparisons


def _build_lps(pattern: str) -> Tuple[List[int], int]:
    """
    Constroi o vetor de prefixos (failure function / LPS - Longest Proper
    Prefix which is also Suffix) usado pelo KMP.

    Complexidade: O(m) tempo, O(m) espaco extra.
    Retorna tambem o numero de comparacoes feitas no pre-processamento,
    pois esse custo deve ser contabilizado na analise pratica.
    """
    m = len(pattern)
    lps = [0] * m
    length = 0
    i = 1
    comparisons = 0

    while i < m:
        comparisons += 1
        if pattern[i] == pattern[length]:
            length += 1
            lps[i] = length
            i += 1
        else:
            if length != 0:
                length = lps[length - 1]
            else:
                lps[i] = 0
                i += 1

    return lps, comparisons


def kmp_search(text: str, pattern: str, case_insensitive: bool = True) -> Tuple[List[int], int]:
    """
    Busca por 'pattern' em 'text' usando o algoritmo de Knuth-Morris-Pratt.

    Complexidade de tempo: O(n + m) no pior e no melhor caso.
    Complexidade de espaco extra: O(m), referente ao vetor LPS.

    O numero de comparacoes retornado inclui tanto o pre-processamento
    (construcao do LPS) quanto a fase de busca propriamente dita, para
    permitir uma comparacao justa com a Forca Bruta.
    """
    if case_insensitive:
        text = text.lower()
        pattern = pattern.lower()

    n = len(text)
    m = len(pattern)
    positions = []

    if m == 0 or n < m:
        return positions, 0

    lps, pre_comparisons = _build_lps(pattern)
    comparisons = pre_comparisons

    i = 0  # indice em text
    j = 0  # indice em pattern

    while i < n:
        comparisons += 1
        if text[i] == pattern[j]:
            i += 1
            j += 1
            if j == m:
                positions.append(i - j)
                j = lps[j - 1]
        else:
            if j != 0:
                j = lps[j - 1]
            else:
                i += 1

    return positions, comparisons


if __name__ == "__main__":
    # teste rapido de sanidade
    texto = "aaaaaaaaaaaaaaaaab"
    padrao = "aaaab"
    print("Forca Bruta:", brute_force_search(texto, padrao))
    print("KMP:        ", kmp_search(texto, padrao))
