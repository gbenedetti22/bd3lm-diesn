# BD3LM con backbone DiESN (PyTorch Lightning)

BD3LM ([arXiv:2503.09573](https://arxiv.org/abs/2503.09573)) con il tuo DiESN (ParalESN, [arXiv:2601.22296](https://arxiv.org/abs/2601.22296)) al posto del Transformer, più il soft masking di SMDM. Il codice originale del progetto non è modificato.

| File | Contenuto |
|---|---|
| `bd3lm.py` | Modulo Lightning: la classe `Diffusion` di `diffusion.py` con `algo=bd3lm`, portata metodo per metodo con gli stessi nomi e la stessa logica. Le aggiunte sono marcate `lightning:`. |
| `diesn_bd3.py` | `DiESNBD3`: `diffusion/diesn.py` (invariato) come backbone di BD3LM, con la stessa interfaccia del `DIT` di `models/dit.py` |
| `config.yaml` | Le chiavi della config Hydra di BD3LM con i valori di `configs/` e `scripts/train/train_owt_bd3lm.sh`, più backbone e soft masking |
| `train.py` | Addestramento: `Trainer`, `ModelCheckpoint`, `WandbLogger`, DDP |
| `evaluate.py` | Le quattro metriche di un checkpoint, aggiunte come riga a `runs/results.md` |
| `test_bd3lm.py` | 26 test (servono le GPU) |

La cartella non ha `__init__.py` perché ha lo stesso nome del pacchetto `lightning`, che altrimenti verrebbe oscurato. Gli script vanno lanciati come file dalla root del progetto (`uv run python lightning/train.py`), non con `-m`.

## Come funziona

### DiESN al posto del Transformer

BD3LM addestra sulla concatenazione `[x_t, x0]` con la maschera `block_diff_mask` di `models/dit.py`:

- **M_BD**: ogni blocco legge tutto il proprio blocco, in entrambe le direzioni;
- **M_OBC**: il blocco rumoroso b legge i blocchi puliti < b;
- **M_BC**: il blocco pulito b legge i blocchi puliti ≤ b.

In DiESN l'informazione si sposta lungo la sequenza in tre moduli; il resto (embedding, AdaLN, MLP, output) agisce posizione per posizione. `DiESNBD3` sostituisce questi tre moduli con versioni che applicano la stessa maschera. I parametri e i buffer sono gli stessi oggetti, quindi lo `state_dict` è identico a quello di `DIESN`. `DiESNBlock.forward` e il resto di `diesn.py` girano invariati.

| Modulo | Dentro il blocco | Tra blocchi |
|---|---|---|
| ParalESN bidirezionale | entrambi i reservoir leggono tutto il blocco | il reservoir forward attraversa i blocchi precedenti e porta il contesto pulito; il backward riparte a ogni blocco. Il blocco rumoroso b riparte dallo stato pulito di fine blocco b-1 (per linearità: `h_t = h_t^blocco + λ^(k+1) h^x0_(s_b-1)`) |
| Conv depthwise (k=5) | legge ±2 posizioni | le 2 posizioni prima del blocco vengono da `x0`, quelle dopo valgono 0 |
| Attention (`model.diesn.attn_impl: softmax` nel config) | completa | `block_diff_mask` di BD3LM, con SDPA. Con `gla` (default di `diesn.py`) `DiESNBD3` si ferma con un errore: è una ricorrenza di fla i cui stati ai confini dei blocchi non sono accessibili |

Senza la maschera, il reservoir backward e l'attention farebbero leggere al blocco rumoroso b il proprio target `x0^b`. Come in BD3LM, la metà `x0` si calcola una volta (è la forward aggiuntiva dell'Algoritmo 1 del paper) e la metà `x_t` la legge. In campionamento c'è una sola passata su `[blocchi puliti, blocco in denoising]`, come il `DIT` in `sample_mode`. I test verificano che:

- su un singolo blocco l'output coincide con il `DIESN` originale;
- il training su `[x_t, x0]` dà gli stessi logit del campionamento;
- il blocco b non dipende da `x0` dei blocchi ≥ b.

### Soft masking (`algo.soft_masking=smdm`, solo DiESN)

Come in SMDM, l'input di un token mascherato è `(1-λ)·e([MASK]) + λ·Σ_k p_k·e(top-k)`. Qui `p` è una predizione preliminare del token e λ dipende dalla sua entropia (`TransparencyHead`).

In training la metà `x0` si calcola una volta e il suo contesto serve a due passate sui blocchi rumorosi:
1. una passata preliminare su `x_t` con maschere binarie, senza gradiente, che fornisce `p`;
2. la passata su `x_t` con le maschere soft, che dà l'output usato nella NELBO.

Il costo aggiuntivo rispetto a BD3LM è quindi una sola passata sui blocchi rumorosi senza gradiente, non una seconda forward completa (`x0` più `x_t`). In inferenza il primo passo di ogni blocco usa maschere binarie e i passi successivi usano `p` dal passo precedente. Le maschere del blocco b non leggono mai `x0` dei blocchi ≥ b, quindi la PPL resta un bound della likelihood.

Gli iperparametri seguono la ricetta OpenWebText dell'originale ([IBM/soft-masked-diffusion-language-models](https://github.com/IBM/soft-masked-diffusion-language-models), `language/algo.py` e `language/scripts/`):

- `optim.sm_prob=0.8`: il soft masking si usa nell'80% degli step di training, negli altri si usano maschere binarie; in validazione e campionamento si usa sempre;
- `optim.tran_head_lr=0.01`: learning rate della testa;
- `algo.tran_head` con `init_scale` 0, `init_centre` -0,75, `init_steep` 6,66 e top-k 3 (`configs/algo/mdlm_sm.yaml`). Con `init_scale` 0 il soft masking parte spento e la testa impara ad aprirlo. Il default di `diffusion/transparency_head.py` è invece 0,1; si ripristina con `algo.tran_head.init_scale=0.1`.

La temperatura della testa entra nel calcolo solo con `transparency_alg="mixinputs_with_temp"`, una variante densa sull'intero vocabolario che `diffusion/transparency_head.py` non include. SMDM si addestra con `mixinputs_with_topk`, dove non è usata neanche nell'originale. Qui è congelata: valore e `state_dict` restano gli stessi. Negli step senza soft masking la testa non riceve gradiente, quindi con il soft masking DDP gira con `find_unused_parameters=True`, come negli script di SMDM. Il costo misurato è circa lo 0,5% per step.

## Preparazione

```bash
uv sync
```

```bash
uv run wandb login
```

In alternativa a `wandb login` puoi aggiungere `wandb.mode=offline` ai comandi. La cache tokenizzata di OpenWebText (ultimi 100k documenti come validazione, come in BD3LM) è già in `~/.cache/gpr_datasets`. Se mancasse, si ricrea con `uv run python -m data.prepare_data --dataset openwebtext`.

```bash
uv run pytest -q lightning/test_bd3lm.py
```

## Addestramento (dentro tmux)

Ogni run usa tutte le GPU visibili: DDP, batch globale 512 × 1024 token, 64 sequenze per GPU con accumulo del gradiente. Le run vanno quindi eseguite una alla volta. Il comando apre una sessione tmux staccata e salva il log in `runs/<nome>.log`; `tmux attach -t <nome>` mostra l'avanzamento. Rilanciare lo stesso comando riprende da `runs/<nome>/checkpoints/last.ckpt` (oppure dal checkpoint indicato in `checkpointing.resume_ckpt_path`) e continua la stessa run wandb. La barra di avanzamento è quella di tqdm, che wandb mostra nei log della run (quella di rich no).

```bash
tmux new-session -d -s dit_bs16 "cd /home/giacomofsm/bd3lms && mkdir -p runs && uv run python lightning/train.py name=dit_bs16 algo.backbone=dit 2>&1 | tee -a runs/dit_bs16.log; exec bash"
```

```bash
tmux new-session -d -s diesn_bs16 "cd /home/giacomofsm/bd3lms && mkdir -p runs && uv run python lightning/train.py name=diesn_bs16 algo.backbone=diesn 2>&1 | tee -a runs/diesn_bs16.log; exec bash"
```

```bash
tmux new-session -d -s diesn_smdm_bs16 "cd /home/giacomofsm/bd3lms && mkdir -p runs && uv run python lightning/train.py name=diesn_smdm_bs16 algo.soft_masking=smdm 2>&1 | tee -a runs/diesn_smdm_bs16.log; exec bash"
```

### Velocità

Il profiling del training originale (DiESN, eager) mostrava che la GPU passava solo il 31% del tempo nelle GEMM. Il resto andava in operazioni limitate dalla banda di memoria:
- LayerNorm in fp32, modulate, residui, dropout e attivazioni, ciascuna come kernel separato;
- la convoluzione depthwise, i cui kernel nativi sono lenti nel backward;
- la parametrizzazione SUBS su tutti i logit (batch × 1024 × 50258).

La scan ParalESN pesava solo il 3%. Tre interventi, nessuno dei quali cambia la matematica (i test lo verificano):

1. `trainer.compile=True`: `torch.compile` del calcolo per-token dei blocchi (`BD3LM.compile_backbone`), sia per DiESN sia per DiT. In DiESN i tre mixer con la maschera restano eager e la scan resta il kernel Triton, come raccomanda il README del kit.
2. La convoluzione depthwise di DiESN è riscritta come 5 moltiplicazioni-somme traslate, compilate (`_depthwise_conv`).
3. La loss calcola `log p(x0 | x_t)` solo sulle posizioni mascherate (`_subs_log_p_theta`). Il valore è quello di `_subs_parameterization` + gather di `diffusion.py`; la SUBS originale resta per il campionamento.

Il risultato è da 1,4× (`smdm`) a 1,7× (DiT) più veloce, con meno memoria. Così entra un batch di 64 sequenze per GPU, che senza compilazione non entra (con `trainer.compile=False` serve `loader.batch_size=32`). Tempi misurati su una A100 e durata stimata con `trainer.max_steps=25000` (13 miliardi di token, circa 1,5 epoche di OpenWebText) su 4 A100, esclusi overhead DDP e validazioni:

| Run | Prima (eager, batch 32) | Ora (compilato, batch 64) | Ore stimate | Ore per epoca |
|---|---|---|---|---|
| `dit_bs16` | 0,78 s / 32 seq | 0,94 s / 64 seq | ~13 | ~9 |
| `diesn_bs16` | 0,70 s / 32 seq | 0,91 s / 64 seq | ~13 | ~9 |
| `diesn_smdm_bs16` | 0,91 s / 32 seq | 1,20 s / 64 seq (media: soft masking nell'80% degli step) | ~17 | ~11 |

Per altre configurazioni si sovrascrivono le chiavi, per esempio `block_size=8` o `trainer.max_steps=50000`.

### Durata in epoche e checkpoint

Con `trainer.max_steps=-1 trainer.max_epochs=N` l'addestramento dura N epoche di OpenWebText. Un'epoca sono circa 17.000 step, cioè 8,9 miliardi di token; BD3LM si addestra per 1M di step, 60 epoche. Ogni checkpoint pesa circa 2,6 GB:

- ogni `trainer.ckpt_every` step (1000) si salva `step<step>.ckpt` e si tengono gli ultimi `trainer.ckpt_keep` (3). `last.ckpt` è un link all'ultimo di questi e serve alla ripresa automatica;
- alla fine di ogni epoca si salva `epoch<epoca>-step<step>.ckpt`, con l'epoca contata da 0 come su wandb. Si tengono tutti (`trainer.ckpt_epochs=-1`; con un valore N > 0 solo gli ultimi N).

I checkpoint di fine epoca sono i punti puliti da cui ripartire: riprendendo a metà epoca, i sampler di Lightning ricominciano l'epoca dall'inizio della sua permutazione. Per ripartire da uno di questi si aggiunge `checkpointing.resume_ckpt_path=runs/<nome>/checkpoints/epoch<epoca>-step<step>.ckpt` al comando. `/home` ha poco spazio libero: 50 checkpoint di fine epoca occupano circa 130 GB.

## Valutazione

Basta una GPU, quindi più valutazioni possono girare in parallelo, una per GPU (`CUDA_VISIBLE_DEVICES`).

```bash
uv run python lightning/evaluate.py runs/dit_bs16/checkpoints/last.ckpt "BD3LM (DiT)"
```

```bash
uv run python lightning/evaluate.py runs/diesn_bs16/checkpoints/last.ckpt "BD3LM + DiESN"
```

```bash
uv run python lightning/evaluate.py runs/diesn_smdm_bs16/checkpoints/last.ckpt "BD3LM + DiESN + soft masking smdm"
```


Definizione delle metriche:

- **PPL**: `exp(NELBO per token)` sull'intera validazione di OpenWebText, con i pesi EMA e senza il primo token (`ignore_bos`). È il bound riportato da BD3LM (`mode=ppl_eval` di `main.py`).
- **MAUVE**: 1000 sequenze generate contro 1000 sequenze umane della validazione con lo stesso numero di token, con `mauve-text` e feature di GPT-2 large. La generazione è quella di `mode=sample_eval` di `main.py`: pesi EMA, sampler semi-autoregressivo first-hitting (una chiamata al modello per token), nucleus 0,9.
- **Tempo di addestramento**: ore dei soli step di training (forward, backward, ottimizzatore), senza validazioni, checkpoint e attese dei dati. Le misura `TrainTimer` di `train.py` e sono salvate nel checkpoint, quindi si sommano tra le riprese. Hardware: quello del training (4 A100).
- **Tempo di inferenza**: secondi per sequenza generata di 1024 token, su 1 GPU, con batch da 50 e senza KV cache.
