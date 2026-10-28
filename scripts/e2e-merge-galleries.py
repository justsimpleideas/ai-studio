"""Two synthetic galleries for the end-to-end test's gallery merge (scripts/e2e-studio.mjs).

Usage: python e2e-merge-galleries.py <folder>

Writes <folder>/pc1/outputs and <folder>/pc2/outputs, made by the engine's own Gallery:
  pc1: two images, one a favourite, in a folder "Beach"; one in a hidden folder "Private"
       locked with the PIN 135790.
  pc2: one image in a folder "beach" (a clash with pc1's "Beach", differing only in case),
       and an exact copy of one of pc1's items (the same id and file), which the merge skips.
Needs Pillow.
"""

import shutil
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "image-engine"))

from PIL import Image  # noqa: E402

from image_engine.gallery import Gallery  # noqa: E402

REQUEST = {"prompt": "", "negative_prompt": "", "width": 64, "height": 48, "steps": 8, "guidance": 0.0,
           "seed": 1, "count": 1}


def save(gallery: Gallery, prompt: str, colour: tuple[int, int, int], seed: int) -> dict:
    return gallery.save(Image.new("RGB", (64, 48), colour), job_id="e2e-merge", model="fake-image",
                        request={**REQUEST, "prompt": prompt, "seed": seed}, seed=seed, duration_ms=10,
                        provenance={"app": "e2e"})


def main(folder: Path) -> None:
    pc1 = Gallery(folder / "pc1" / "outputs")
    beach = pc1.create_folder("Beach")
    shared = save(pc1, "a merged sandcastle", (230, 200, 120), 101)
    pc1.set_folder(shared["id"], beach["id"])
    pc1.set_favorite(shared["id"], True)
    pc1.set_folder(save(pc1, "a merged pier", (90, 120, 200), 102)["id"], beach["id"])
    private = pc1.create_folder("Private")
    pc1.set_pin(private["id"], "135790")
    pc1.set_hidden(private["id"], True)
    pc1.set_folder(save(pc1, "a merged diary page", (240, 240, 230), 103)["id"], private["id"], everything=True)
    pc1.close()

    pc2 = Gallery(folder / "pc2" / "outputs")
    pc2.set_folder(save(pc2, "a merged wave", (20, 90, 160), 201)["id"], pc2.create_folder("beach")["id"])
    pc2.close()

    # pc1's sandcastle copied into pc2 as is: the same id and the same bytes.
    src = folder / "pc1" / "outputs"
    dst = folder / "pc2" / "outputs"
    db = sqlite3.connect(src / "gallery.db")
    db.row_factory = sqlite3.Row
    row = dict(db.execute("SELECT * FROM images WHERE id = ?", (shared["id"],)).fetchone())
    db.close()
    for rel in (row["file"], row["thumb"]):
        (dst / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src / rel, dst / rel)
    row["folder_id"] = ""
    db = sqlite3.connect(dst / "gallery.db")
    db.execute(f"INSERT INTO images ({', '.join(row)}) VALUES ({', '.join('?' * len(row))})", list(row.values()))
    db.commit()
    db.close()


if __name__ == "__main__":
    main(Path(sys.argv[1]))
