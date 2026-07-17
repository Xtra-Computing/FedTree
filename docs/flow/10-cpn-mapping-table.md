<details>
<summary>10-cpn-mapping-table</summary>

Tabela de mapeamento direto para o CPN Tools. O objetivo e sair da leitura conceitual e chegar em uma estrutura pratica para desenhar a rede: cada lugar tem um tipo, uma pagina dona e uma relacao clara com transicoes de entrada e saida.

## Como Ler Esta Tabela

- `Lugar`: nome do port place ou lugar interno.
- `Tipo`: color set ou tipo conceitual do token.
- `Entrada`: transicao ou pagina que produz esse lugar.
- `Saida`: transicao ou pagina que consome esse lugar.
- `Pagina dona`: pagina substituta principal responsavel pelo estado.

## Tabela Base

| Lugar | Tipo | Entrada | Saida | Pagina dona |
|---|---|---|---|---|
| `P_init` | `SYSTEMSTATE` | entrada do executavel | `T_create_parties` | `P0` |
| `P_parties_created` | `PARTYSTATE` | `T_create_parties` | `T_connect_server`, `P2` | `P1` |
| `P_channel_open` | `CONTROLMSG` | `T_connect_server` | `T_send_pid` | `P2` |
| `P_pid_sent` | `CONTROLMSG` | `T_send_pid` | `T_receive_ready` | `P2` |
| `P_connected` | `SERVERSTATE` | `T_receive_ready` | `T_begin_barrier` | `P2`, `P3` |
| `P_privacy_ready` | `CONTROLMSG` | `T_begin_barrier` | `T_bootstrap_privacy` | `P3` |
| `P_privacy_bootstrap_done` | `CONTROLMSG` | `T_bootstrap_privacy` | `T_load_partition` | `P4` |
| `P_data_ready` | `PARTITIONEDDATA` | `T_load_partition` | `T_start_training_horizontal`, `T_start_training_vertical`, `T_start_training_alt` | `P5` |
| `P_mode_selected` | `MODE` | `T_select_mode` | `T_start_training_horizontal`, `T_start_training_vertical`, `T_start_training_alt` | `P0` |
| `P_model_saved` | `MODELSTATE` | `T_prune_and_score`, `T_merge_trees`, `T_train_local_model` | `T_save_model` | `P6`, `P7`, `P8`, `P9` |
| `P_session_closed` | `STOPMSG` | `T_save_model` | `T_stop_server` | `P9` |
| `P_server_stop_requested` | `STOPMSG` | `T_stop_server` | `T_finalize` | `P9` |
| `P_finished` | `SYSTEMSTATE` | `T_finalize` | fim do executavel | `P0`, `P9` |

## Tipos De Controle

| Tipo | Significado | Uso principal |
|---|---|---|
| `SYSTEMSTATE` | estado global do fluxo | entrada e saida do protocolo |
| `PARTYSTATE` | contexto local da party | criacao e inicializacao |
| `SERVERSTATE` | estado de coordenacao do servidor | handshake e sincronizacao |
| `MODE` | ramo operacional | escolha horizontal, vertical, ensemble ou solo |
| `PRIVACYTECH` | estrategia de privacidade | decisao de bootstrap |
| `BARRIER` | liberacao de sincronizacao | barreira inicial |
| `CONTROLMSG` | mensagem curta de protocolo | PID, Ready, bootstrap, liberacao |
| `STOPMSG` | mensagem de encerramento | fim da sessao RPC |

## Tipos De Dados

| Tipo | Significado | Onde aparece |
|---|---|---|
| `DATASETBUNDLE` | conjunto inicial de dados | carregamento local |
| `PARTITIONSPEC` | descricao da particionamento | decisao de modo |
| `PARTITIONEDDATA` | dados ja recortados | preparacao para treino |
| `FEATUREBATCH` | lote de features | ramo vertical |
| `GHPAIR` | gradients e hessians resumidos | treino horizontal e vertical |
| `GHBATCH` | batch de gradients | ramo vertical sem label |
| `GHENCBATCH` | batch cifrado de gradients | ramo vertical com `HE` |
| `HISTOGRAMMSG` | histograma local ou agregado | ambos os modos |
| `FIDBATCH` | identificadores de features | ramo vertical |
| `SPLITPOINT` | corte escolhido | decisao de arvore |
| `NODEMSG` | no sincronizado | atualizacao de arvore |
| `SCOREMSG` | score local ou medio | fim de rodada |
| `HOMOKEY` | chave publica homomorfica | bootstrap `HE` |
| `DHPUBLICKEY` | chave publica Diffie-Hellman | bootstrap `SA` |
| `SANOISE` | ruido trocado entre parties | bootstrap `SA` |
| `DPNOISE` | ruido local de privacidade diferencial | bootstrap `DP` |

## Fluxo De Referencia

### Controle inicial

`P_init -> T_create_parties -> P_parties_created -> T_connect_server -> P_channel_open -> T_send_pid -> P_pid_sent -> T_receive_ready -> P_connected`

### Sincronizacao inicial

`P_connected -> T_begin_barrier -> P_privacy_ready -> T_bootstrap_privacy -> P_privacy_bootstrap_done -> T_load_partition -> P_data_ready`

### Treino horizontal

`P_data_ready -> T_start_training_horizontal -> ... -> T_prune_and_score -> P_model_saved`

### Treino vertical

`P_data_ready -> T_start_training_vertical -> ... -> T_prune_and_score -> P_model_saved`

### Encerramento

`P_model_saved -> T_save_model -> P_session_closed -> T_stop_server -> P_server_stop_requested -> T_finalize -> P_finished`

## Mapa Rapido Por Pagina

- `P0`: contexto global e fechamento.
- `P1`: criacao das parties e preparacao local.
- `P2`: handshake, identificacao e encerramento da sessao.
- `P3`: barreira e prontidao.
- `P4`: bootstrap de privacidade.
- `P5`: particionamento e carregamento.
- `P6`: treino horizontal.
- `P7`: treino vertical.
- `P8`: ensemble ou solo.
- `P9`: persistencia, stop e finalizacao.

## Observacao De Modelagem

Esta tabela e suficiente para:

- definir lugares no CPN Tools;
- decidir quais tipos entram em cada port place;
- ligar paginas pai e filha sem ambiguidade;
- separar controle de dados de treino;
- manter o ramo vertical como referencia principal sem excluir o horizontal.

O proximo refinamento natural e converter esta tabela em uma lista de paginas com:

- lugares internos;
- transicoes internas;
- guards;
- inscricoes de arco.

</details>
