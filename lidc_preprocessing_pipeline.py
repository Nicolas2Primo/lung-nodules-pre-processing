"""lidc_preproc_general.py

Pipeline *neutro* de pré_processamento do LIDC_IDRI.
Gera artefatos **genéricos** — *volumes reamostrados, máscaras de nódulos e máscara pulmonar* —
para que qualquer modelo (2_D, 2.5_D, 3_D) possa ser preparado em passos posteriores.

Estrutura de saída
==================
processed_lidc_general/
├─ volumes/           patientID.npz   «volume» int16 HU (Z,Y,X)
├─ masks/             patientID_mask.npz   «nodule_mask» uint8 (Z,Y,X)
├─ lung_masks/        patientID_lung.npz   «lung_mask» uint8 (Z,Y,X)
└─ metadata.parquet   tabela por *scan* (id, dims, n_nódulos, split)

Características principais
-------------------------
* **Reamostragem** → voxel_spacing = 1 mm³ (trilinear).
* **Consenso** → majority vote ≥ 50 %.
* **Máscara pulmonar** → threshold (HU < –320) + morfologia → 2 maiores CC.
* **Split estratificado** por paciente (train/val/test).
* Tudo salvo em **np.savez_compressed** (leve) + parquet.
"""

from __future__ import annotations

import argparse
import random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import pylidc as pl
import scipy.ndimage as ndi
from sklearn.model_selection import GroupShuffleSplit

# ----------------------------------------------------------------------------
# Config ---------------------------------------------------------------------
# ----------------------------------------------------------------------------

@dataclass
class Config:
    out_dir: Path = Path("processed_lidc_general")
    voxel_spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    consensus_thr: float = 0.5
    random_seed: int = 42

    def mkdirs(self):
        (self.out_dir / "volumes").mkdir(parents=True, exist_ok=True)
        (self.out_dir / "masks").mkdir(parents=True, exist_ok=True)
        (self.out_dir / "lung_masks").mkdir(parents=True, exist_ok=True)

# ----------------------------------------------------------------------------
# Utils ----------------------------------------------------------------------
# ----------------------------------------------------------------------------

RNG = np.random.default_rng(42)


def resample(vol: np.ndarray, spacing: np.ndarray, tgt: Tuple[float, float, float]) -> np.ndarray:
    zoom = spacing / np.array(tgt)
    return ndi.zoom(vol, zoom, order=1)


def build_consensus(scan: pl.Scan, shape: Tuple[int, int, int], thr: float) -> np.ndarray:
    stacks = []
    for ann in scan.cluster_annotations():
        m, (z0, y0, x0) = ann.boolean_mask(loc=True)
        tmp = np.zeros(shape, np.uint8)
        tmp[z0:z0+m.shape[0], y0:y0+m.shape[1], x0:x0+m.shape[2]] = m
        stacks.append(tmp)
    if not stacks:
        return np.zeros(shape, np.uint8)
    return (np.mean(stacks, axis=0) >= thr).astype(np.uint8)


def lung_mask(vol: np.ndarray) -> np.ndarray:
    mask = vol < -320
    mask = ndi.binary_closing(mask, iterations=3)
    mask = ndi.binary_fill_holes(mask)
    lbl, n = ndi.label(mask)
    if n == 0:
        return mask.astype(np.uint8)
    areas = ndi.sum(mask, lbl, index=range(1, n+1))
    keep = np.argsort(areas)[-2:] + 1
    return np.isin(lbl, keep).astype(np.uint8)

# ----------------------------------------------------------------------------
# Main loop -------------------------------------------------------------------
# ----------------------------------------------------------------------------

def main(cfg: Config):
    random.seed(cfg.random_seed)
    np.random.seed(cfg.random_seed)
    cfg.mkdirs()

    rows: List[dict] = []
    for scan in pl.query(pl.Scan):
        try:
            vol_hu, spacing, _ = scan.to_volume()
            vol_hu = vol_hu.astype(np.int16)
            vol_hu = resample(vol_hu, spacing, cfg.voxel_spacing)

            nodule_mask = build_consensus(scan, vol_hu.shape, cfg.consensus_thr)
            lung = lung_mask(vol_hu)

            # Salvar arquivos -------------------------------------------------
            vol_path = cfg.out_dir / "volumes" / f"{scan.patient_id}.npz"
            mask_path = cfg.out_dir / "masks" / f"{scan.patient_id}_mask.npz"
            lung_path = cfg.out_dir / "lung_masks" / f"{scan.patient_id}_lung.npz"
            np.savez_compressed(vol_path, volume=vol_hu)
            np.savez_compressed(mask_path, nodule_mask=nodule_mask)
            np.savez_compressed(lung_path, lung_mask=lung)

            rows.append({
                "patient_id": scan.patient_id,
                "scan_id": scan.id,
                "z": vol_hu.shape[0], "y": vol_hu.shape[1], "x": vol_hu.shape[2],
                "n_nodules": int(nodule_mask.any()),
            })
            print(f"✔ {scan.patient_id}")
        except Exception as e:
            print(f"⚠ {scan.patient_id}: {e}")

    meta = pd.DataFrame(rows)

    # Split paciente ---------------------------------------------------------
    gss = GroupShuffleSplit(test_size=0.20, n_splits=1, random_state=cfg.random_seed)
    tr_idx, test_idx = next(gss.split(meta, meta["n_nodules"], groups=meta["patient_id"]))
    meta.loc[test_idx, "split"] = "test"
    tr_meta = meta.loc[tr_idx]
    gss2 = GroupShuffleSplit(test_size=0.125, n_splits=1, random_state=cfg.random_seed)
    tr2_idx, val_idx = next(gss2.split(tr_meta, tr_meta["n_nodules"], groups=tr_meta["patient_id"]))
    meta.loc[tr_meta.index[tr2_idx], "split"] = "train"
    meta.loc[tr_meta.index[val_idx], "split"] = "val"

    meta.to_parquet(cfg.out_dir / "metadata.parquet")
    print("\n✅ Geral concluído em", cfg.out_dir.resolve())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", type=Path, default="processed_lidc_general")
    cfg = Config(out_dir=ap.parse_args().out_dir)
    main(cfg)
