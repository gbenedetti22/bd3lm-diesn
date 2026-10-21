# Analisi dell'addestramento: run `diesn_smdm_bs16`

BD3-LM con backbone DiESN e soft masking SMDM, block size 16, su OpenWebText. La run è partita l'11 ottobre 2026 con un budget di 25.000 step, poi è stata ripresa il 12 ottobre con un budget a epoche (50, LR costante). Il 21 ottobre è stata fermata allo **step 312.000 (18,3 epoche)** perché il server veniva dismesso. I dati sono in [`results/diesn_smdm_bs16/`](../results/diesn_smdm_bs16), gli script in [`analysis/`](../analysis).

## 1. Configurazione

| | |
|---|---|
| Modello | DiESN: 12 blocchi (ParalESN bidirezionale fisso con 768 unità per direzione e 12 teste, ρ ∈ [0,4, 0,9], più conv depthwise k=5 e MLP 4×), 1 strato di attention softmax (12 teste, RoPE, QK norm), d = 768, AdaLN, dropout 0,1. 144M parametri addestrati |
| BD3-LM | block size 16, rumore loglinear, parametrizzazione SUBS, campionamento antitetico, ricerca dell'intervallo di mascheramento (`var_min`), EMA 0,9999 |
| Soft masking | SMDM (top-3), `sm_prob` 0,8, `tran_head_lr` 0,01, `init_scale` 0 (ricetta OpenWebText di IBM) |
| Ottimizzazione | AdamW, LR 3e-4 costante dopo 2,5k step di warmup, weight decay 0, gradient clipping 1, batch 512 × 1024 token |
| Dati | OpenWebText (8,9 miliardi di token di training per epoca; gli ultimi 100k documenti sono la validazione), tokenizer GPT-2 |
| Hardware | 4× A100 80GB PCIe, DDP, `torch.compile`; 0,734 h ogni 1000 step, cioè ~12,5 h per epoca |
| Durata | 312.000 step, 164 miliardi di token, **229,5 h** di soli step di training (`TrainTimer`) |

## 2. Curva di validazione

![curva](../results/diesn_smdm_bs16/val_curve.png)

val/nll è la NELBO per token sull'intervallo completo [1e-3, 1], con i pesi EMA, su 2048 sequenze di validazione, calcolata ogni 5000 step. È la metrica di BD3-LM: PPL = e^nll. La train loss non è confrontabile, perché viene misurata sull'intervallo di mascheramento ristretto scelto dalla ricerca.

| Epoca (indice) | val/nll medio | PPL | Δ rispetto all'epoca precedente |
|---|---|---|---|
| 0 | 3,946 | 51,7 | |
| 1 | 3,705 | 40,6 | −0,242 |
| 2 | 3,631 | 37,7 | −0,074 |
| 3 | 3,581 | 35,9 | −0,049 |
| 4 | 3,557 | 35,1 | −0,024 |
| 5 | 3,540 | 34,5 | −0,018 |
| 7 | 3,516 | 33,7 | |
| 9 | 3,487 | 32,7 | |
| 11 | 3,473 | 32,2 | |
| 13 | 3,458 | 31,8 | |
| 15 | 3,447 | 31,4 | |
| 17 | 3,436 | 31,0 | |

La tabella completa è in `epochs.csv`. La miglior validazione singola è **3,426 (PPL 30,8)** allo step 300k.

- **Rumore di misura.** Con gli stessi pesi, una validazione oscilla di σ ≈ 0,007: è stato misurato ripetendola 6 volte allo step 25k. Le oscillazioni tra validazioni vicine, per esempio 3,426 → 3,456 → 3,440, sono quindi rumore. Conviene guardare la media per epoca, che ha un rumore di circa ±0,004.
- **Andamento.** Nelle prime 3 epoche la discesa è rapida. Poi rallenta: −0,02 per epoca intorno all'epoca 5, circa −0,006 per epoca nelle ultime cinque.
- **Fit a legge di potenza** su tutte le validazioni dallo step 30k, `L(s) = L∞ + A·s^-α`. L'asintoto stimato è L∞ = 3,20 ± 0,07 (α = 0,30), con 3,42 previsto a 25 epoche, 3,38 a 50 e 3,37 a 60. È un'estrapolazione indicativa, ma a LR costante il 3,10 del paper non sembra raggiungibile in un budget paragonabile al suo.

## 3. Confronto con BD3-LM

| | val/nll | PPL | Budget |
|---|---|---|---|
| BD3-LM paper, DiT 110M, L'=16 | 3,10 | 22,27 | 60 epoche: 850k step di pre-training MDLM più 150k a L'=16 |
| BD3-LM paper, DiT, L'=4 | 3,03 | 20,73 | 60 epoche |
| **DiESN + SMDM, L'=16 (questa run)** | **3,44** (media epoca 17) | **31,0** | 18,3 epoche, solo L'=16 |

A parità di dati e di ricetta, la distanza dal paper ha tre cause che qui non si possono separare:

1. **Architettura.** Nel DiT il mescolamento lungo la sequenza è fatto da 12 strati di attention addestrati. In DiESN è fatto da reservoir lineari fissi (con mixer e merge fissi), da una convoluzione k=5 e da un solo strato di attention addestrato.
2. **Budget.** Il 30% delle epoche del paper.
3. **Pre-training.** Il paper parte da 850k step di pre-training MDLM.

Separare il contributo dell'architettura da quello del soft masking richiede le due baseline allo stesso budget (`dit_bs16`, `diesn_bs16`), che non sono state addestrate. Solo come contesto: le run DiESN precedenti, basate sul codice SMDM di IBM con un'impostazione diversa, non erano mai scese sotto 3,7.

## 4. Intervallo di mascheramento appreso

BD3-LM sceglie a ogni validazione l'intervallo U[β, ω] delle frazioni di mascheramento che minimizza la varianza della NELBO. Qui la scelta è andata così:

| Step | Intervallo scelto |
|---|---|
| 5k | U[0,50, 1] |
| 10k-125k | U[0,45, 0,95] |
| 130k | U[0,40, 0,90] |
| 140k | U[0,35, 0,85] |
| dal 145k in poi | **U[0,30, 0,80]** |

U[0,30, 0,80] è l'intervallo che il paper indica come migliore per L'=16 (Tabella 8). La ricerca ha quindi ritrovato il risultato del paper anche con DiESN.

## 5. Soft masking

La testa di trasparenza calcola `λ = scale · σ(steepness · (-H - centre))`, dove H è l'entropia della predizione preliminare.

| | scale | centre | steepness |
|---|---|---|---|
| Inizializzazione | 0 (soft masking spento) | −0,75 | 6,66 |
| Step 25k | 0,96 | −5,84 | 0,28 |
| Step 312k | **1,00** (saturo dallo step ~117k) | −3,39 | 0,29 |

Valori finali di λ: 0,73 con una predizione sicura (H = 0), 0,53 a 3 nats, 0,32 a 6 nats, 0,21 a 8 nats.

La testa ha "aperto" il soft masking fino in fondo. Allo step 25k, con gli stessi pesi, sostituire le soft mask con maschere binarie peggiorava la NELBO di 0,09 nats (PPL da 40,3 a 44,0): il modello usa davvero l'informazione delle soft mask. È un controllo diagnostico, non un confronto con DiESN senza soft masking.

## 6. Norme degli embedding e token `[MASK]`

Norma dell'embedding d'ingresso di `[MASK]` con i pesi EMA, da `mask_norm_epochs.csv` (rango: 1 = la norma più alta, su 50258 righe):

| Checkpoint | Norma `[MASK]` | Rango | Norma mediana | Coseno massimo con un token |
|---|---|---|---|---|
| step 25k | 1,31 | 18.460 | 1,29 | 0,22 |
| fine epoca 1 | 1,47 | 33.986 | 1,50 | 0,23 |
| fine epoca 5 | 2,24 | 49.737 | 2,61 | 0,25 |
| fine epoca 9 | 2,74 | 49.979 | 3,48 | 0,25 |
| fine epoca 13 | 3,06 | 50.047 | 4,19 | 0,25 |
| fine epoca 17 | 3,33 | 50.068 | 4,78 | 0,27 |

- **Crescita generale.** Con AdamW senza weight decay tutte le righe crescono più o meno come la radice del numero di step: la mediana passa da 1,29 a 4,83.
- **[MASK] cresce meno.** Finisce tra le norme **più piccole** del vocabolario: rango 50.068 su 50.258.
  - Nel BD3-LM ufficiale (`kuleshov-group/bd3lm-owt-block_size16`, senza soft masking) è invece la **4ª più alta**: 11,5 contro una mediana di 8,1. In MDLM è l'11ª.
  - L'ipotesi più plausibile è il soft masking. L'input di una posizione mascherata è dominato dalla miscela dei token predetti (la `scale` della testa è satura a 1, λ ≥ 0,5 per predizioni con meno di ~3,4 nats di entropia), e `e([MASK])` resta un piccolo marcatore che segnala la posizione mascherata.
  - Verificarla richiede la run DiESN senza soft masking.
- **Direzione.** `[MASK]` resta una direzione propria: il coseno con qualunque token non supera 0,27.

## 7. Salute del training

Controlli su `step312000.ckpt` (`analysis/ckpt_analysis.py` e `analysis/direction_analysis.py`):

- **Pesi.** Nessun NaN/Inf, né nei pesi grezzi né in quelli EMA. Reservoir, mixer e merge sono identici bit per bit all'inizializzazione ricostruita con lo stesso seed: le parti congelate non sono state toccate.
- **Gradiente.** Norma RMS ≈ 0,19 nelle ultime ~1000 iterazioni (stimata dai momenti di Adam), sotto la soglia di clipping 1,0.
- **Riga di uscita di `[MASK]`.** È esattamente 0, come deve: con SUBS il suo logit è −∞.
- **Bias d'uscita.** I più alti sono `,` ` the` ` in` ` and`: il modello ha appreso le frequenze dei token.
- **Gate AdaLN.** Il percorso ESN + conv dei blocchi è diventato molto più importante durante il training: |gate| da 0,1-0,8 allo step 25k a 0,6-2,0 allo step 312k, massimo nei blocchi centrali. Il gate dell'unico strato di attention è sceso da 1,25 a 0,48. Il modello ha spostato parte del lavoro dall'attention ai reservoir e alla convoluzione.
- **Direzione dei pesi.**
  - Il movimento degli ultimi ~10k step è ancora allineato alla traiettoria complessiva (coseno 0,23 su 144M di pesi; per direzioni casuali sarebbe ~0,0001), a circa metà della velocità media.
  - Gli aggiornamenti per step sono però dominati dal rumore: |m|/√v = 0,14-0,17, il livello del rumore puro.
  - È il regime tipico di un LR costante. Il modello migliora ancora, ma lentamente, ed è plausibile che un decadimento finale del LR (deviazione dalla ricetta di BD3-LM) dia un calo rapido.

## 8. Campioni

Quattro sequenze da 1024 token dallo step 312k (pesi EMA, sampler first-hitting semi-autoregressivo, nucleus 0,9) sono in [`samples_step312000.txt`](../results/diesn_smdm_bs16/samples_step312000.txt). Sono inglese scorrevole che resta sullo stesso argomento: baseball, una serie TV, pinguini, una squadra di esport. Il senso globale resta vago, come ci si aspetta a PPL ~31. Un estratto:

> The big question? Scientists know how many penguins will survive. We have an estimated some 130 million penguins, including at sea. As added advice, scientists reported in the journal Nature that it would help penguins and other females the ability to shear with energy and movement. […]

## 9. Note tecniche

- **Storia su wandb incompleta.** Nella run wandb (`gbenedetti22-unipi/BD3-LMs-DiESN`, run `diesn_smdm_bs16_1`) mancano le righe degli step 25k-166k della sessione ripresa. La storia completa è stata ricostruita dai file locali `.wandb` (`analysis/read_wandb_local.py`) ed è in `validation.csv` e `train.csv`.
- **Rumore uguale su tutte le GPU.** Tutte le 4 GPU usano lo stesso seed, quindi estraggono gli stessi t, le stesse maschere e la stessa scelta tra soft e binario su dati diversi. È così anche nel BD3-LM originale.
- **Checkpoint durante il training.** Gli ultimi 3 checkpoint a step più uno a fine di ogni epoca (`lightning/train.py: checkpoint_callbacks`). Sono stati salvati nella release i pesi dello step 312k e il checkpoint completo di fine epoca 17.

## 10. Cosa resta da fare

1. **Riga ufficiale della tesi** (PPL sull'intera validazione | MAUVE | tempo di addestramento | tempo di inferenza): `lightning/evaluate.py` sui pesi della release. Il tempo di inferenza dipende dall'hardware, e il training è stato fatto su 4× A100.
2. **Baseline allo stesso budget:** `dit_bs16` (BD3-LM originale) e `diesn_bs16` (senza soft masking), con i comandi in `lightning/README.md`. Servono a separare l'effetto dell'architettura da quello del soft masking.
3. **Decadimento finale del LR**, da applicare a tutte le run: è una deviazione dalla ricetta di BD3-LM.
4. **Block size 4**, per il confronto con il miglior risultato del paper, dopo aver misurato il costo di DiESN con blocchi così piccoli.
5. **KV cache per DiESN**, rimandata di proposito.
