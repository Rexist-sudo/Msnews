import json
import os
from pathlib import Path
from typing import Dict, Optional, Union

import numpy as np
import torch
from torch.utils.data import Dataset


class DataReaderMmap(Dataset):
    """
    基于 mmap 目录的数据读取器（单 part 版本）。

    期望目录结构：
    - split_dir/
      - manifest.json
      - part_000001/
        - <feature>.npy
        - <feature>_mask.npy (array feature)
        - label.npy
    """

    def __init__(self, config, feature_file_path: str = None, train_user_id_set: set = None, filter_negative: bool = False):
        self.sparse_features = set(config.features.sparse_feature_names)
        self.dense_features = set(config.features.dense_feature_names)
        self.array_features = set(config.features.array_feature_names)
        self.array_max_length = config.features.array_max_length
        self.statistics_mode = config.get("statistics_cfg", {}).get("mode")

        if feature_file_path is None:
            raise ValueError("Data file path must be provided.")
        self.data_path = Path(feature_file_path)
        if not self.data_path.exists():
            raise FileNotFoundError(f"Data file path does not exist: {self.data_path}")

        self.manifest = self._load_manifest(self.data_path)
        self.part_info = self._resolve_single_part(self.manifest, self.data_path)
        self.part_dir = self.data_path / self.part_info["path"]
        self.total_rows = int(self.part_info["rows"])
        # Windows spawn 会 pickle Dataset；不要把 mmap 数组内容传给 worker。
        # 每个进程首次取样时独立打开文件，之后复用该进程内的映射。
        self.arrays = None
        self._arrays_pid = None
        if not (self.part_dir / "label.npy").exists():
            raise FileNotFoundError(f"label.npy missing in part directory: {self.part_dir}")
        if self.statistics_mode is not None:
            with (self.data_path.parent / "news_statistics.json").open(encoding="utf-8") as stream:
                statistics = json.load(stream)
            split = self.manifest["split"]
            if statistics["splits"][split]["rows"] != self.total_rows:
                raise ValueError("统计缓存与候选行数不一致，请重新生成 news_statistics")
            if statistics["history_length"] != int(self.array_max_length["user_history"]):
                raise ValueError("统计缓存与有效历史长度不一致")

        self.selected_indices: Optional[np.ndarray] = None
        if train_user_id_set is not None:
            self.filter_data_by_user_id(train_user_id_set)
        if filter_negative:
            self.filter_negative_data()

    def __getstate__(self):
        state = self.__dict__.copy()
        state["arrays"] = None
        state["_arrays_pid"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.arrays = None
        self._arrays_pid = None

    def close(self):
        """释放本进程的映射；返回给调用方的 Tensor 已复制，不依赖映射。"""
        arrays = self.arrays
        self.arrays = None
        self._arrays_pid = None
        if arrays is not None:
            for array in arrays.values():
                if isinstance(array, np.memmap):
                    array._mmap.close()

    def _ensure_arrays(self):
        pid = os.getpid()
        if self.arrays is None or self._arrays_pid != pid:
            self.close()
            self.arrays = self._load_single_part_arrays(self.part_dir)
            self._arrays_pid = pid
        return self.arrays

    def _load_manifest(self, split_dir: Path) -> Dict:
        manifest_path = split_dir / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"manifest.json not found in mmap directory: {split_dir}")
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
        if manifest.get("format") != "npy_mmap":
            raise ValueError(f"Unsupported manifest format: {manifest.get('format')}, expected 'npy_mmap'")
        if "parts" not in manifest or not manifest["parts"]:
            raise ValueError(f"Invalid manifest parts in: {manifest_path}")
        return manifest

    def _resolve_single_part(self, manifest: Dict, split_dir: Path) -> Dict:
        """
        解析并校验单 part 信息。
        若 manifest 里有多个 part，直接报错，避免读到与当前约定不一致的数据格式。
        """
        parts = manifest["parts"]
        if len(parts) != 1:
            raise ValueError(
                f"DataReaderMmap(single-part) expects exactly 1 part, "
                f"but got {len(parts)} in {split_dir / 'manifest.json'}"
            )

        part_info = parts[0]
        part_path = split_dir / part_info["path"]
        if not part_path.exists():
            raise FileNotFoundError(f"Part directory does not exist: {part_path}")

        rows = int(part_info.get("rows", 0))
        if rows <= 0:
            raise ValueError(f"Invalid part rows={rows} in manifest: {split_dir / 'manifest.json'}")
        return part_info

    def _load_single_part_arrays(self, part_dir: Path) -> Dict[str, np.ndarray]:
        """
        一次性 mmap 加载单 part 的全部需要字段。
        """
        arrays = {}

        # 只加载 __getitem__ 会用到的字段，统一 mmap_mode='r' 避免额外内存拷贝。
        for fea in self.sparse_features:
            fp = part_dir / f"{fea}.npy"
            arrays[fea] = np.load(fp, mmap_mode="r") if fp.exists() else None
        for fea in self.dense_features:
            fp = part_dir / f"{fea}.npy"
            arrays[fea] = np.load(fp, mmap_mode="r") if fp.exists() else None
        for fea in self.array_features:
            f_fp = part_dir / f"{fea}.npy"
            m_fp = part_dir / f"{fea}_mask.npy"
            arrays[fea] = np.load(f_fp, mmap_mode="r") if f_fp.exists() else None
            arrays[f"{fea}_mask"] = np.load(m_fp, mmap_mode="r") if m_fp.exists() else None

        label_fp = part_dir / "label.npy"
        if not label_fp.exists():
            raise FileNotFoundError(f"label.npy missing in part directory: {part_dir}")
        arrays["label"] = np.load(label_fp, mmap_mode="r")
        # 请求 ID 仅用于评估分组，不作为模型特征。
        impression_fp = part_dir / "impression_id.npy"
        if impression_fp.exists():
            arrays["impression_id"] = np.load(impression_fp, mmap_mode="r")
        if self.statistics_mode is not None:
            names = ["item_statistics", "history_statistics"]
            if self.statistics_mode == "smooth":
                names += ["item_state", "history_state"]
            for name in names:
                arrays[name] = np.load(self.data_path.parent / f"{self.manifest['split']}_{name}.npy", mmap_mode="r")
        return arrays

    def __len__(self) -> int:
        if self.selected_indices is None:
            return self.total_rows
        return int(self.selected_indices.shape[0])

    def __getitem__(self, idx):
        return {name: value[0] for name, value in self.get_batch([idx]).items()}

    def __getitems__(self, indices):
        batch = self.get_batch(indices)
        return [{name: values[i] for name, values in batch.items()} for i in range(len(indices))]

    def get_batch(self, indices):
        """批量读取同一套字段；训练 DataLoader 与 checkpoint 重放共用。"""
        indices = np.asarray(indices, dtype=np.int64)
        if indices.ndim != 1 or np.any(indices < 0) or np.any(indices >= len(self)):
            raise IndexError("Batch indices out of range")
        rows = self.selected_indices[indices] if self.selected_indices is not None else indices
        arrays = self._ensure_arrays()
        batch = {}
        for name in self.sparse_features | self.dense_features | {"label", "impression_id"}:
            values = arrays.get(name)
            if values is None and name == "impression_id":
                continue
            dtype = np.float32 if name in self.dense_features or name == "label" else np.int64
            values = np.zeros(len(rows), dtype=dtype) if values is None else np.array(values[rows], dtype=dtype)
            batch[name] = torch.from_numpy(values)
        for name in self.array_features:
            length = int(self.array_max_length[name])
            values = arrays.get(name)
            sequence = (np.full((len(rows), length), -1, dtype=np.int64) if values is None
                        else np.array(values[rows], dtype=np.int64))
            if sequence.shape != (len(rows), length):
                raise ValueError(f"Array feature '{name}' length mismatch: expected {length}")
            masks = arrays.get(name + "_mask")
            mask = (sequence != -1).astype(np.float32) if masks is None else np.array(masks[rows], dtype=np.float32)
            batch[name] = torch.from_numpy(sequence)
            batch[name + "_mask"] = torch.from_numpy(mask)
        if self.statistics_mode is not None:
            requests = arrays["impression_id"][rows].reshape(-1)
            for prefix, index in (("item", rows), ("history", requests)):
                values = np.array(arrays[prefix + "_statistics"][index], dtype=np.float32)
                features = values[..., :4].copy()
                if self.statistics_mode == "smooth":
                    features[..., 1] = values[..., 4]
                    batch[prefix + "_state"] = torch.from_numpy(np.array(arrays[prefix + "_state"][index], dtype=np.int64))
                batch[prefix + "_statistics"] = torch.from_numpy(features)
        return batch

    def _merge_selected(self, new_indices: np.ndarray):
        new_indices = np.asarray(new_indices, dtype=np.int64)
        if self.selected_indices is None:
            self.selected_indices = np.sort(new_indices)
            return
        mask = np.isin(self.selected_indices, new_indices)
        self.selected_indices = self.selected_indices[mask]

    def get_user_id_set(self, feature_name: str = "user_id") -> set:
        unique_ids = set()
        fp = self.part_dir / f"{feature_name}.npy"
        if not fp.exists():
            return unique_ids
        vals = np.load(fp, mmap_mode="r")
        for v in np.unique(vals):
            unique_ids.add(int(v))
        return unique_ids

    def filter_data_by_user_id(self, user_id_set: set, feature_name: str = "user_id"):
        if not user_id_set:
            self.selected_indices = np.empty((0,), dtype=np.int64)
            return

        candidate_indices = []
        target_ids = np.asarray(list(user_id_set))
        fp = self.part_dir / f"{feature_name}.npy"
        if fp.exists():
            ids = np.load(fp, mmap_mode="r")
            keep_mask = np.isin(ids, target_ids)
            local_keep = np.nonzero(keep_mask)[0]
            if local_keep.size > 0:
                candidate_indices.append(local_keep)

        selected = (
            np.concatenate(candidate_indices).astype(np.int64)
            if candidate_indices
            else np.empty((0,), dtype=np.int64)
        )
        self._merge_selected(selected)

    def filter_negative_data(self):
        candidate_indices = []
        fp = self.part_dir / "label.npy"
        if fp.exists():
            labels = np.asarray(np.load(fp, mmap_mode="r"), dtype=np.float32)
            if labels.ndim == 1:
                keep_mask = labels > 0
            else:
                keep_mask = labels[:, 0] > 0
            local_keep = np.nonzero(keep_mask)[0]
            if local_keep.size > 0:
                candidate_indices.append(local_keep)

        selected = (
            np.concatenate(candidate_indices).astype(np.int64)
            if candidate_indices
            else np.empty((0,), dtype=np.int64)
        )
        self._merge_selected(selected)
