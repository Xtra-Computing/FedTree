# Engenharia Reversa do FedTree – Prompt de Análise

## Contexto

Estou realizando uma análise de engenharia reversa do projeto **FedTree** como parte de uma pesquisa acadêmica.

O objetivo **NÃO** é compreender os detalhes matemáticos do algoritmo de aprendizado de máquina, mas sim entender completamente o **protocolo de aprendizado federado** implementado pelo FedTree para que ele possa ser posteriormente modelado utilizando **Redes de Petri Coloridas (CPN Tools)**.

O objetivo final é produzir uma documentação detalhada que permita construir um modelo formal capaz de verificar propriedades como sincronização, comunicação, corretude, concorrência e possíveis pontos de ataque durante o processo de aprendizado federado.

---

# Seu Papel

Atue como um **arquiteto de software**, **engenheiro de protocolos distribuídos** e **analista de sistemas distribuídos**.

Não faça apenas uma explicação superficial do código.

Sua função é reconstruir o comportamento do sistema durante sua execução.

Sempre que possível, acompanhe o fluxo de execução desde a função `main()` até o encerramento completo do treinamento.

O foco deve ser entender **como o sistema funciona**, e não apenas **como o código está escrito**.

---

# Objetivo Principal

Neste momento, o objetivo principal **não é documentar todos os detalhes do projeto**, mas sim construir **um fluxograma evolutivo** do funcionamento do FedTree.

Esse fluxograma deverá representar o comportamento observado no código-fonte e será refinado continuamente conforme novas partes do projeto forem sendo analisadas.

Inicialmente, o diagrama pode conter apenas as etapas de mais alto nível. À medida que novas informações forem descobertas, essas etapas deverão ser detalhadas, substituindo abstrações por comportamentos reais encontrados no código.

---

# Fluxograma

O fluxograma deverá ser escrito em **Mermaid** e salvo em um arquivo **Markdown (.md)**.

O arquivo deverá ser atualizado a cada nova análise, preservando o histórico do fluxo já conhecido e refinando apenas as partes necessárias.

Exemplo da estrutura esperada:

```text
main()

↓

Inicialização

↓

Servidor

↓

Clientes

↓

Treinamento

↓

Finalização
```

Com o avanço da análise, cada um desses blocos poderá ser expandido em novos subprocessos.

Por exemplo:

```text
Treinamento

↓

Construção da Árvore

↓

Cálculo dos Gradientes

↓

Troca de Histogramas

↓

Escolha do Split

↓

Atualização da Árvore
```

E posteriormente:

```text
Troca de Histogramas

↓

Cliente calcula histograma

↓

Cliente envia ao servidor

↓

Servidor aguarda todos

↓

Servidor agrega

↓

Servidor responde aos clientes
```

O fluxograma deve evoluir naturalmente conforme o entendimento do sistema aumenta.

---

# Processo de Trabalho

A análise deve ser incremental.

Analise poucos arquivos por vez.

Ao final de cada etapa:

* Atualize o arquivo `.md` contendo o fluxograma.
* Refine apenas os blocos que puderem ser comprovados pelo código.
* Evite expandir partes ainda não analisadas.
* Preserve a legibilidade do diagrama.

Sempre prefira uma visão geral correta a um diagrama excessivamente detalhado ou baseado em suposições.

---

# Regras

* Não invente comportamentos.
* Não faça suposições.
* Toda expansão do fluxograma deve ser baseada exclusivamente no código analisado.
* Caso uma parte do fluxo ainda não esteja clara, mantenha-a como um bloco abstrato até que existam evidências suficientes para detalhá-la.

---

# Objetivo Final

Ao final da análise, o arquivo Markdown deverá conter um fluxograma completo e fiel ao funcionamento do FedTree, permitindo compreender:

* O fluxo geral de execução.
* A interação entre clientes e servidor.
* Os pontos de sincronização.
* As etapas do treinamento federado.
* Os estados do protocolo.

Esse fluxograma servirá como a principal referência para a futura modelagem em Redes de Petri Coloridas (CPN Tools).
