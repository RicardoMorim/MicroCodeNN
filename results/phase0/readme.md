# Resultados do treino da Phase 0

Teste realizado após o treino da phase 0 com 20k samples de treino e 5M samples de teste.

## Métricas gerais

- Register Accuracy: 0.9387
- State Accuracy: 0.7556
- Total de samples: 5.000.000
- Total de registros corretos: 18.774.754
- Total de estados corretos: 3.778.230

## Acurácia por opcode

| Opcode | Acurácia | Correto / Total |
| --- | ---: | ---: |
| DEC | 0.9815 | 816.992 / 832.429 |
| SWAP | 0.9317 | 775.744 / 832.602 |
| INC | 0.9878 | 824.599 / 834.774 |
| COPY | 0.9477 | 790.287 / 833.903 |
| ADD | 0.3404 | 283.455 / 832.750 |
| SUB | 0.3445 | 287.153 / 833.542 |

## Distribuição por opcode

- Total por opcode: `{1: 832429, 5: 832602, 3: 833542, 0: 834774, 4: 833903, 2: 832750}`
- Correto por opcode: `{1: 816992, 5: 775744, 0: 824599, 4: 790287, 2: 283455, 3: 287153}`
- Opcode Accuracy: `{'DEC': 0.9814554754819931, 'SWAP': 0.931710469107689, 'SUB': 0.3444973378665982, 'INC': 0.987811072218349, 'COPY': 0.9476965546352514, 'ADD': 0.3403842689882918}`

## Conclusões

- O modelo apresenta boa acurácia para os opcodes `DEC`, `SWAP`, `INC` e `COPY`.
- A acurácia para os opcodes `ADD` e `SUB` é muito baixa, o que sugere que o modelo tem dificuldade em aprender a lógica desses dois casos.
- Isso indica que a arquitetura atual consegue lidar bem com operações simples de manipulação direta, mas ainda não generaliza adequadamente para operações que exigem cálculo/transformação de valores entre registros.

## Próximos passos

1. Aumentar a quantidade de samples de treino para tentar melhorar a acurácia dos opcodes `ADD` e `SUB`.
2. Ajustar a `loss function` para penalizar mais fortemente apenas os registros que realmente devem ser alterados, reduzindo a tendência do modelo de simplesmente copiar valores.
3. Testar um embedding em formato de vetor ou em valores decimais diretos para verificar se isso melhora a acurácia dos opcodes `ADD` e `SUB`.
4. Avaliar a possibilidade de separar melhor o treinamento por tipo de operação, focando mais nos casos mais difíceis.

---

Resumo rápido:

- Register Accuracy: `0.9387`
- State Accuracy: `0.7556`
- Melhor desempenho: `DEC`, `INC`, `COPY`, `SWAP`
- Maior desafio: `ADD` e `SUB`
