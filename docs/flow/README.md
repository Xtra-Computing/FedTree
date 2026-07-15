# Flow Documentation

Documentação hierárquica e incremental do FedTree.

## Níveis

- [00-overview.md](./00-overview.md)
- [01-train-main.md](./01-train-main.md)
- [02-distributed-server.md](./02-distributed-server.md)
- [03-distributed-party.md](./03-distributed-party.md)
- [04-horizontal-training.md](./04-horizontal-training.md)
- [05-vertical-training.md](./05-vertical-training.md)
- [06-network.md](./06-network.md)

## Regras

- Cada diagrama deve manter no máximo 20 blocos.
- Exceção: `06-network.md` é a visão central da comunicação cliente-servidor e pode exceder esse limite.
- Se uma etapa ficar complexa, ela ganha um novo arquivo.
- O conteúdo deve evoluir conforme novos arquivos do código forem analisados.

## Checklist De Expansao Do 06

- [ ] Mapear o fluxo de entrada e a barreira inicial.
- [ ] Detalhar o bootstrap de privacidade.
- [ ] Separar o ramo horizontal por etapas de rede.
- [ ] Separar o ramo vertical por etapas de rede.
- [ ] Explicitar o ponto em que a party salva o modelo.
- [ ] Explicitar o handshake de `StopServer()`.
- [ ] Confirmar quais mensagens são comuns aos dois modos.
- [ ] Remover qualquer duplicação entre os ramos.
- [ ] Revisar os rótulos para manter linguagem de protocolo.
- [ ] Validar o diagrama final contra o código-fonte.
