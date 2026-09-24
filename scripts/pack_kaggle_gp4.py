"""Zip 6 train segments (image+box+hkp) for a second Kaggle dataset. POSIX paths."""

from __future__ import annotations

import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "waymo_v2" / "training"
OUT = ROOT / "data" / "kaggle_gp4_train_segments.zip"
SEGS = [
    "7566697458525030390_1440_000_1460_000",
    "16951470340360921766_2840_000_2860_000",
    "11252086830380107152_1540_000_1560_000",
    "454855130179746819_4580_000_4600_000",
    "4114454788208078028_660_000_680_000",
    "10485926982439064520_4980_000_5000_000",
]


def _hkp(seg: str) -> Path:
    hits = list((DATA / "camera_hkp").glob(f"**/{seg}.parquet"))
    if not hits:
        raise FileNotFoundError(seg)
    return hits[0]


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(OUT, "w", compression=zipfile.ZIP_STORED) as zf:
        for seg in SEGS:
            img = DATA / "camera_image" / f"{seg}.parquet"
            box = DATA / "camera_box" / f"{seg}.parquet"
            hkp = _hkp(seg)
            for src, arc in (
                (img, f"training/camera_image/{seg}.parquet"),
                (box, f"training/camera_box/{seg}.parquet"),
                (hkp, f"training/camera_hkp/{seg}.parquet"),
            ):
                if not src.exists():
                    raise FileNotFoundError(src)
                print("add", arc, src.stat().st_size)
                zf.write(src, arcname=arc)
    print("wrote", OUT, OUT.stat().st_size)


if __name__ == "__main__":
    main()
