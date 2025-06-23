#!/usr/bin/env python3
import argparse
import os
import re
from zipfile import ZipFile

import numpy as np
import zfpy
from PIL import Image

def parse_args():
    p = argparse.ArgumentParser(
        description="Extract depth frames from a zfpy zip into contrast-stretched 16-bit PNGs"
    )
    p.add_argument("zipfile", help="Path to depth_zfpy.zip")
    p.add_argument("outdir", help="Directory to write depth_XXXX.png files into")
    p.add_argument("--start", type=int, default=None,
                   help="Minimum frame index to extract (e.g. 10 for d10.zfp)")
    p.add_argument("--end",   type=int, default=None,
                   help="Maximum frame index to extract (e.g. 50 for d50.zfp)")
    p.add_argument("--vmin", type=float, default=None,
                   help="Depth value to map to black (auto=per-frame min)")
    p.add_argument("--vmax", type=float, default=None,
                   help="Depth value to map to white (auto=per-frame max)")
    return p.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    # match files like “d5.zfp”, “d10.zfp”, etc.
    pattern = re.compile(r"^d(\d+)\.zfp$")
    with ZipFile(args.zipfile, "r") as zf:
        for entry in sorted(zf.namelist()):
            name = os.path.basename(entry)
            m = pattern.match(name)
            if not m:
                continue

            idx = int(m.group(1))
            if args.start is not None and idx < args.start:
                continue
            if args.end is not None and idx > args.end:
                continue

            # print(f"[+] Processing {name} → frame #{idx}")
            compressed = zf.read(entry)
            arr = zfpy.decompress_numpy(compressed)  # returns a NumPy array

            # determine min/max for contrast stretch
            mn = args.vmin if args.vmin is not None else float(arr.min())
            mx = args.vmax if args.vmax is not None else float(arr.max())

            # normalize & convert to 16-bit
            if mx > mn:
                norm = (arr - mn) / (mx - mn)
                norm = np.clip(norm, 0.0, 1.0)
                arr_uint = (norm * 65535).astype(np.uint16)
            else:
                arr_uint = np.zeros_like(arr, dtype=np.uint16)

            # save as 16-bit PNG
            img = Image.fromarray(arr_uint, mode="I;16")
            out_path = os.path.join(args.outdir, f"depth_{idx:04d}.png")
            img.save(out_path)

    print("Done.")

if __name__ == "__main__":
    main()
