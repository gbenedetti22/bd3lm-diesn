"""
Prepare the token streams of a dataset.

    python -m data.prepare_data --dataset openwebtext [--val-docs 100000]
    python -m data.prepare_data --dataset tinystories

OpenWebText: token streams of the provided dataset/ classes (dataset.OpenWebTextDataset, cached in
~/.cache/gpr_datasets), validation = the last --val-docs documents, training = all the others (data/openwebtext.py);
prints the two files to use as data.token_files. The other datasets are tokenized into uint16 token files
(<data_dir>/<name>/{train,val}.bin, data/prepare.py).
"""
import argparse

from data.prepare import HF_DATASETS, prepare_hf_dataset


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True, choices=sorted(HF_DATASETS) + ["openwebtext"])
    ap.add_argument("--data-dir", default="data_cache")
    ap.add_argument("--max-docs", type=int, default=0, help="limit the number of training documents")
    ap.add_argument("--val-docs", type=int, default=100_000, help="openwebtext: last documents used as validation")
    a = ap.parse_args(argv)
    if a.dataset == "openwebtext":
        from data.openwebtext import ensure_split
        print(ensure_split(a.val_docs))
        return
    prepare_hf_dataset(a.dataset, a.data_dir, a.max_docs)


if __name__ == "__main__":
    main()
