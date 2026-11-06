"""从训练特征统计候选曝光及有效历史支持，不额外复制数据缓存。"""

from pathlib import Path

import numpy as np

from ..DataReader.data_reader_mmap import DataReaderMmap
from ..DataReader.data_reader import DataReader


def load_support(config):
    item_count = int(config.embeddings.embedding_table_size.item_id)
    user_count = int(config.embeddings.embedding_table_size.user_id)
    if not config.dataset.load_npz:
        reader = DataReader(config, str(Path(config.paths.out_basedir) / "extractored_feature/train_features.txt"))
        candidate, users, historical = np.zeros(item_count, dtype=int), np.zeros(user_count, dtype=int), np.zeros(item_count, dtype=int)
        requests = set()
        for row in reader:
            candidate[row["item_id"]] += 1
            users[row["user_id"]] += 1
            request = (row["user_id"], row["impression_id"])
            if request not in requests:
                requests.add(request)
                history = row["user_history"].numpy()
                np.add.at(historical, history[history > 0], 1)
        return {"item": candidate + historical, "user": users, "candidate": candidate}
    directory = Path(config.paths.out_basedir) / "extractored_feature/train_features_mmap"
    reader = DataReaderMmap(config, directory)
    part = reader.part_dir
    load = lambda name: np.load(part / f"{name}.npy", mmap_mode="r").reshape(-1)
    candidate = np.bincount(load("item_id"), minlength=item_count)
    users = np.bincount(load("user_id"), minlength=user_count)
    _, first = np.unique(load("impression_id"), return_index=True)
    history = np.load(part / "user_history.npy", mmap_mode="r")[first]
    historical = np.bincount(history[history > 0], minlength=item_count)
    return {"item": candidate + historical, "user": users, "candidate": candidate}
