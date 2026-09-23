"""Print the GP1 dataset catalog and optional local-slice checklist."""

from __future__ import annotations

import argparse
from pathlib import Path

from pose_av.catalog import CATALOG, NUSCENES_PRIOR_FILTER, WAYMO_PUBLIC


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--waymo-root",
        type=Path,
        default=None,
        help="Local Perception v2 root with camera_hkp/ if already downloaded",
    )
    args = parser.parse_args()

    print("== Catalog (GP1) ==")
    for row in CATALOG:
        print(f"- {row.name}")
        print(f"    keypoints: {row.keypoints}")
        print(f"    download:  {row.download_slice}")
        print(f"    role:      {row.role}")

    print("\n== Waymo published sizes ==")
    for key, value in WAYMO_PUBLIC.items():
        print(f"  {key}: {value}")

    print("\n== Prior nuScenes filter (from vekha 4 notes) ==")
    for key, value in NUSCENES_PRIOR_FILTER.items():
        print(f"  {key}: {value}")

    if args.waymo_root is None:
        print("\nNo --waymo-root given. EDA on parquet is deferred until a slice is on disk.")
        print("  camera_hkp only: ~45 MB, enough to count objects per segment.")
        print("  + camera_image for those segments: needed for visual QA and training.")
        return

    hkp = args.waymo_root / "camera_hkp"
    n_files = len(list(hkp.glob("*.parquet"))) if hkp.is_dir() else 0
    print(f"\nLocal camera_hkp parquet files: {n_files} under {hkp}")
    if n_files == 0:
        print("  Directory missing or empty — download instructions are in README.md")


if __name__ == "__main__":
    main()
