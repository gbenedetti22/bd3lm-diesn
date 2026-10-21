"""Validation tokens of OpenWebText without the full token cache: downloads the last parquet shard of
Skylion007/openwebtext (100,172 documents, inside the last 100k documents = the validation split of BD3-LM and of
data/openwebtext.py), tokenizes the second half of it as the data pipeline does (GPT-2, <|endoftext|> after every
document) and saves N_SEQ x 1024 tokens (uint16). Usage: owt_val_tokens.py DOWNLOAD_DIR OUT.npy [N_SEQ]"""
import sys

import numpy as np
import pyarrow.parquet as pq
import tiktoken
from huggingface_hub import hf_hub_download

out_dir, out, n_seq = sys.argv[1], sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 600
path = hf_hub_download('Skylion007/openwebtext', 'plain_text/train-00079-of-00080.parquet', repo_type='dataset',
                       local_dir=out_dir)
docs = pq.read_table(path, columns=['text']).column('text').to_pylist()
enc, eot = tiktoken.get_encoding('gpt2'), 50256
tokens = []
for d in docs[len(docs) // 2:]:
    if d and d.strip():
        tokens.extend(enc.encode_ordinary(d))
        tokens.append(eot)
    if len(tokens) >= 1024 * n_seq:
        break
np.save(out, np.array(tokens[:1024 * n_seq], dtype=np.uint16))
print(f'{len(docs)} documents in the shard; saved {n_seq} sequences of 1024 tokens to {out}')
