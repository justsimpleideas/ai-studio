"""Where the gallery folder is, when the user has moved it (the studio's locations.ts)."""

import tempfile
import unittest
from pathlib import Path

from image_engine.__main__ import gallery_root
from image_engine.gallery import Gallery


class GalleryRootTests(unittest.TestCase):
    def test_sits_beside_the_settings_by_default(self):
        data = Path(tempfile.gettempdir()) / "studio-data"
        self.assertEqual(gallery_root(data, None), data / "outputs")

    def test_uses_the_folder_the_studio_passes(self):
        chosen = Path(tempfile.gettempdir()) / "somewhere-else"
        self.assertEqual(gallery_root(Path(tempfile.gettempdir()), chosen), chosen)

    def test_refuses_a_relative_folder(self):
        with self.assertRaises(ValueError):
            gallery_root(Path(tempfile.gettempdir()), Path("outputs"))

    def test_a_chosen_folder_is_made_and_keeps_what_the_old_one_had(self):
        with tempfile.TemporaryDirectory() as tmp:
            data = Path(tmp) / "data"
            first = Gallery(gallery_root(data, None))
            first.close()
            self.assertTrue((data / "outputs" / "gallery.db").exists())

            moved = Path(tmp) / "moved"
            second = Gallery(gallery_root(data, moved))
            second.close()
            self.assertTrue((moved / "gallery.db").exists())
            self.assertTrue((moved / "images").is_dir())
            # The folder left behind is untouched: pointing back at it finds it whole.
            self.assertTrue((data / "outputs" / "gallery.db").exists())


if __name__ == "__main__":
    unittest.main()
