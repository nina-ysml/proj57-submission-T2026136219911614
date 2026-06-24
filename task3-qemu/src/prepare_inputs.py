import json
from pathlib import Path

import numpy as np
from PIL import Image


PACKAGE_DIR = Path(__file__).resolve().parents[1]

IMAGE_PATH = PACKAGE_DIR / "data" / "frame_000227.jpg"
STATS_PATH = PACKAGE_DIR / "data" / "stats.json"
OUT_DIR = Path(__file__).resolve().parent / "input_bins"


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    with open(STATS_PATH, "r", encoding="utf-8") as f:
        stats = json.load(f)

    state_q01 = np.array(stats["observation.state"]["q01"], dtype=np.float32)
    state_q99 = np.array(stats["observation.state"]["q99"], dtype=np.float32)
    action_q01 = np.array(stats["action"]["q01"], dtype=np.float32)
    action_q99 = np.array(stats["action"]["q99"], dtype=np.float32)

    img = Image.open(IMAGE_PATH).convert("RGB")
    img = img.resize((224, 224))

    image = np.asarray(img).astype(np.float32) / 255.0
    image = image.transpose(2, 0, 1)

    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)
    image = (image - mean) / std

    image = image[np.newaxis, np.newaxis, :, :, :].astype(np.float32)

    state = np.array([[0.0, 0.0]], dtype=np.float32)
    denom = np.where(state_q99 - state_q01 == 0, 1e-8, state_q99 - state_q01)
    state = (2 * (state - state_q01) / denom - 1).astype(np.float32)

    image.tofile(OUT_DIR / "image_1x1x3x224x224.bin")
    state.tofile(OUT_DIR / "state_1x2.bin")
    action_q01.tofile(OUT_DIR / "action_q01_3.bin")
    action_q99.tofile(OUT_DIR / "action_q99_3.bin")

    print("wrote:", OUT_DIR)
    print("image shape:", image.shape)
    print("state:", state)


if __name__ == "__main__":
    main()
