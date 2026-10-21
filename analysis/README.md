# Script di analisi

Gli script usati per [`docs/ANALISI.md`](../docs/ANALISI.md). Si lanciano dalla root del repository con `uv run python analysis/<script> ...`; dove non serve la GPU si può impostare `CUDA_VISIBLE_DEVICES=""`.

| Script | Cosa fa | Argomenti |
|---|---|---|
| `read_wandb_local.py` | ricostruisce la storia completa di una run dai file locali `.wandb` (il formato a blocchi di wandb), anche quando sul cloud mancano righe | `OUT.pkl RUN.wandb [RUN.wandb ...]` |
| `make_results.py` | `validation.csv`, `train.csv`, `epochs.csv`, fit a legge di potenza e `val_curve.png` | `HISTORY.pkl OUT_DIR` |
| `mask_norm_epochs.py` | norma e rango dell'embedding di `[MASK]` (pesi EMA) in una serie di checkpoint | `OUT.csv CKPT [CKPT ...]` |
| `ckpt_analysis.py` | salute di un checkpoint: NaN/Inf, buffer congelati rispetto all'inizializzazione ricostruita con il seed, norme dei parametri, embedding e `[MASK]`, gate AdaLN, gradiente stimato dai momenti di Adam, curva λ della testa di trasparenza | `CKPT` |
| `direction_analysis.py` | direzione dei pesi: movimento recente (pesi grezzi meno EMA) contro la traiettoria complessiva (EMA meno inizializzazione), allineamento con l'aggiornamento di Adam, rapporto segnale/rumore del gradiente | `CKPT TOKENS.npy` (stream di token di training, per le frequenze) |
| `gpu_diag.py` | val/nll alla fine del training, confronto a parità di rumore tra pesi EMA, pesi grezzi ed estrapolazioni lungo il movimento recente, soft masking acceso e spento, λ, norme del flusso residuo, campioni | `CKPT SAMPLES.txt` |
| `val_noise.py` | rumore di una validazione: stessi pesi, t e maschere diversi | `CKPT` |
| `final_samples.py` | campioni da 1024 token (pesi EMA, sampler di BD3-LM) | `CKPT OUT.txt [N]` |
| `weights_only.py` | copia "solo pesi" di un checkpoint (modello, EMA, configurazione), caricabile da `evaluate.py` | `CKPT OUT` |
| `owt_val_tokens.py` | token di validazione di OpenWebText senza la cache completa: scarica l'ultimo shard parquet del dataset (dentro gli ultimi 100k documenti) e lo tokenizza come la pipeline | `DOWNLOAD_DIR OUT.npy [N_SEQ]` |
| `depth_profiles.py` | profili per profondità dai pesi EMA di più checkpoint: AdaLN di ogni blocco (scale, shift, gate, guadagno d'ingresso), norme (Frobenius e spettrali) di MLP, attention e convoluzioni; memoria dei reservoir (modulo degli autovalori, emivita) | `OUT.csv CKPT [CKPT ...]` |
| `paralesn_role.py` | ruolo di ParalESN e dell'attention: contributi di ogni ramo al flusso residuo per blocco; ablazioni (togliere ESN, conv, attention, MLP; togliere solo il contesto tra blocchi), anche blocco per blocco | `CKPT TOKENS.npy OUT_PREFIX [--per-block]` |
| `plot_depth.py` | grafici di `depth_profiles.py` e `paralesn_role.py` | `RESULTS_DIR` |
| `test_ckpt_callbacks.py` | test su CPU della logica dei checkpoint di `lightning/train.py` (ultimi 3 a step, uno a fine epoca, riprese senza duplicati) | nessuno |
