"""Full history of a W&B run from its local .wandb files (leveldb-log format: 32 KiB blocks, records with a 7-byte
header <checksum u32, length u16, type u8>, types FULL=1 FIRST=2 MIDDLE=3 LAST=4; payload = wandb Record protobuf).
Usage: read_wandb_local.py OUT.pkl RUN_FILE.wandb [RUN_FILE.wandb ...]"""
import json
import struct
import sys

import pandas as pd
from wandb.proto import wandb_internal_pb2 as pb

BLOCK, HEADER = 32768, 7


def records(path):
    data = open(path, 'rb').read()
    ident, magic, version = struct.unpack('<4sHB', data[:7])
    assert ident == b':W&B', ident
    pos, buf = 7, b''
    while pos + HEADER <= len(data):
        space = BLOCK - pos % BLOCK
        if space < HEADER:
            pos += space
            continue
        _, length, rtype = struct.unpack('<IHB', data[pos:pos + HEADER])
        pos += HEADER
        if rtype == 0 and length == 0:  # zero padding up to the end of the block
            pos += BLOCK - pos % BLOCK if pos % BLOCK else 0
            continue
        chunk = data[pos:pos + length]
        pos += length
        if rtype == 1:
            yield chunk
        elif rtype == 2:
            buf = chunk
        elif rtype == 3:
            buf += chunk
        elif rtype == 4:
            yield buf + chunk
            buf = b''


rows = []
for path in sys.argv[2:]:
    n = 0
    for raw in records(path):
        rec = pb.Record()
        rec.ParseFromString(raw)
        if rec.WhichOneof('record_type') != 'history':
            continue
        row = {'_file': path}
        for item in rec.history.item:
            key = item.key or '/'.join(item.nested_key)
            try:
                row[key] = json.loads(item.value_json)
            except ValueError:
                row[key] = item.value_json
        rows.append(row)
        n += 1
    print(path, 'history rows:', n)
df = pd.DataFrame(rows)
df.to_pickle(sys.argv[1])
v = df[df['val/nll'].notna()][['trainer/global_step', 'epoch', 'val/nll', 'val/ppl']]
print(len(df), 'rows,', len(v), 'validations')
print(v.to_string())
