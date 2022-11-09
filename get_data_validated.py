from dp.phonemizer import Phonemizer
import argparse
import librosa
import json
import os
import random
from pathlib import Path
from glob import glob
import csv

import numpy as np
from nemo_text_processing.text_normalization.normalize import Normalizer


def get_args():
    parser = argparse.ArgumentParser(description='Download openSLR dataset and create manifests with predefined split')

    parser.add_argument("--data-root", type=Path, help="where the resulting dataset will reside", default="data")
    parser.add_argument("--val-size", default=0.1, type=float)
    parser.add_argument("--test-size", default=0.2, type=float)
    parser.add_argument(
        "--seed-for-ds-split",
        default=28,
        type=float,
        help="Seed for deterministic split of train/dev/test, NVIDIA's default is 100",
    )

    args = parser.parse_args()
    return args


def __process_transcript(file_path: str):
    phonemizer = Phonemizer.from_checkpoint('checkpoints/dp.pt')
    text_normalizer = Normalizer(
        lang="it", input_case="cased", overwrite_cache=False, cache_dir=str(file_path / "cache_dir"), whitelist=str(file_path / "whitelist.tsv")
    )
    text_normalizer_call_kwargs = {"punct_pre_process": True, "punct_post_process": True}
    normalizer_call = lambda x: text_normalizer.normalize(x, **text_normalizer_call_kwargs)
    entries = []

    file = 'data/cv-corpus/validated.tsv'
    print(file)
    with open(file, 'r') as f:
        index = list(csv.reader(f, delimiter="\t"))
    for fragment in index[1:]:
        audio = {}
        audio['audio_filepath'] = 'data/cv-corpus/clips/'+fragment[1][:-3]+'wav'
        audio['duration'] = librosa.get_duration(filename=audio['audio_filepath'])
        txt = fragment[2]
        audio['text'] = phonemizer(normalizer_call(txt), lang='it')
        entries.append(audio)

    return entries


def __process_data(dataset_path, val_size, test_size, seed_for_ds_split):
    entries = __process_transcript(dataset_path)

    random.Random(seed_for_ds_split).shuffle(entries)

    train_size = 1.0 - val_size - test_size
    train_entries, validate_entries, test_entries = np.split(
        entries, [int(len(entries) * train_size), int(len(entries) * (train_size + val_size))]
    )

    assert len(train_entries) > 0, "Not enough data for train, val and test"

    def save(p, data):
        with open(p, 'w') as f:
            for d in data:
                f.write(json.dumps(d) + '\n')

    save(dataset_path / "train_manifest_stt.json", train_entries)
    save(dataset_path / "val_manifest_stt.json", validate_entries)
    save(dataset_path / "test_manifest_stt.json", test_entries)


def main():
    args = get_args()
    dataset_root = args.data_root
    dataset_root.mkdir(parents=True, exist_ok=True)
    __process_data(
        dataset_root, args.val_size, args.test_size, args.seed_for_ds_split,
    )


if __name__ == "__main__":
    main()
