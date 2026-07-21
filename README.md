# Força Bruta vs. KMP — Detecção de Palavras-Gatilho de Spam

Comparação entre os algoritmos de casamento de padrões Força Bruta e
Knuth-Morris-Pratt (KMP), aplicados à detecção de palavras-gatilho de spam
em mensagens de texto reais.

## Arquivos

- **`algorithms.py`**: implementações instrumentadas dos algoritmos de
  Força Bruta e KMP. Cada função de busca retorna as posições onde o
  padrão ocorre no texto e o número de comparações de caracteres
  realizadas.

- **`keyword_extraction.py`**: carrega um dataset rotulado como spam/ham e
  seleciona palavras-gatilho de forma data-driven, calculando a diferença
  de frequência de cada palavra entre mensagens spam e ham.

- **`run_experiments.py`**: executa os experimentos principais sobre o
  dataset SMS Spam Collection (tempo de execução, número de comparações,
  efeito do tamanho da mensagem, cenário sintético de pior caso, espaço
  extra utilizado) e gera os gráficos e arquivos de resultado
  correspondentes.

- **`run_experiments_extra.py`**: executa experimentos complementares
  (efeito do comprimento do padrão, efeito do tamanho do alfabeto, e
  métricas de classificação do filtro de palavras-gatilho).

- **`run_experiments_enron.py`**: repete os experimentos principais sobre
  o dataset Enron-Spam, com mensagens de e-mail bem mais longas que as de
  SMS.

- **`requirements.txt`**: dependências Python necessárias para rodar os
  scripts.

- **`spam.csv`**: dataset SMS Spam Collection (Almeida, Gómez Hidalgo e
  Yamakami, 2011), 5.572 mensagens de SMS rotuladas como spam ou ham.

- **`enron_spam_data.csv`**: dataset Enron-Spam (Metsis, Androutsopoulos e
  Paliouras, 2006), 33.716 e-mails rotulados como spam ou ham.

## Como rodar

```bash
pip install -r requirements.txt
python3 run_experiments.py
python3 run_experiments_extra.py
python3 run_experiments_enron.py
```

Os resultados (arquivos CSV e gráficos) são gerados na pasta `results/`.
