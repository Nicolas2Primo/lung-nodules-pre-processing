import random
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def main(out_dir="processed_lidc_general", seed=0):
    out_dir = Path(out_dir)
    meta = pd.read_parquet(out_dir / "metadata.parquet")
    rng = random.Random(seed)
    row = meta.sample(1, random_state=rng.randint(0, 10000)).iloc[0]

    base = f"{row.patient_id}_{row.scan_id}"
    vol = np.load(out_dir / "volumes" / f"{base}.npz")['volume']
    mask = np.load(out_dir / "masks" / f"{base}_mask.npz")['nodule_mask']
    lung = np.load(out_dir / "lung_masks" / f"{base}_lung.npz")['lung_mask']

    z = vol.shape[0] // 2
    fig, ax = plt.subplots(1, 1)
    ax.imshow(vol[z], cmap='gray')
    ax.imshow(lung[z], alpha=0.2, cmap='Blues')
    ax.imshow(mask[z], alpha=0.4, cmap='Reds')
    ax.set_title(f"{base} slice {z}")
    ax.axis('off')
    plt.show()


if __name__ == "__main__":
    main()
