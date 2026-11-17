from pathlib import Path
import stat
import tempfile
import unittest
from zipfile import ZipFile, ZipInfo
from PIL import Image
from config import CLASS_NAMES
from scripts.download_dataset import safe_extract
from smartwaste.dataset import inventory,split_dataset


class DatasetTests(unittest.TestCase):
    def test_safe_archive(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); archive=root/"safe.zip"
            with ZipFile(archive,"w") as z: z.writestr("dataset-resized/paper/example.txt","ok")
            safe_extract(archive,root/"out")
            self.assertEqual((root/"out/dataset-resized/paper/example.txt").read_text(),"ok")

    def test_unsafe_members_rejected_before_any_write(self):
        for name in ("../escape", "/absolute", "C:/escape", "a\\escape", "a/NUL.txt", "a/file."):
            with self.subTest(name=name),tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp); archive=root/"bad.zip"
                with ZipFile(archive,"w") as z:
                    z.writestr("safe.txt","safe")
                    info=ZipInfo(name)
                    info.filename=name  # retain backslashes on Windows for the adversarial fixture
                    z.writestr(info,"bad")
                with self.assertRaises(ValueError): safe_extract(archive,root/"out")
                self.assertFalse((root/"out/safe.txt").exists())

    def test_symlink_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); archive=root/"link.zip"
            with ZipFile(archive,"w") as z:
                info=ZipInfo("link"); info.external_attr=(stat.S_IFLNK|0o777)<<16
                z.writestr(info,"../outside")
            with self.assertRaises(ValueError): safe_extract(archive,root/"out")

    def test_distribution_determinism_and_no_split_leakage(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            for ci,category in enumerate(CLASS_NAMES):
                (root/category).mkdir()
                for j in range(20):
                    Image.new("RGB",(8,8),(ci*30,j*10,12)).save(root/category/f"{j}.png")
            # Same image within a class is deduplicated before splitting.
            (root/"paper/duplicate.png").write_bytes((root/"paper/0.png").read_bytes())
            frame=inventory(root)
            self.assertEqual(len(frame),120)
            self.assertEqual(frame.category.value_counts().to_dict(),{x:20 for x in CLASS_NAMES})
            splits=split_dataset(frame)
            self.assertEqual(splits.to_csv(index=False),split_dataset(frame).to_csv(index=False))
            self.assertEqual(splits.split.value_counts().to_dict(),{"train":84,"validation":18,"test":18})
            self.assertEqual(splits.sha256.nunique(),len(splits))
            for name in ("train","validation","test"):
                self.assertEqual(set(splits[splits.split==name].category),set(CLASS_NAMES))

    def test_missing_classes_fail(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError): inventory(tmp)

    def test_all_conflicting_duplicates_are_quarantined(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            for ci,category in enumerate(CLASS_NAMES):
                (root/category).mkdir()
                Image.new("RGB",(8,8),(ci*30,20,15)).save(root/category/"unique.png")
            ambiguous=Image.new("RGB",(8,8),(3,6,9))
            ambiguous.save(root/"paper/conflict.png")
            ambiguous.save(root/"plastic/conflict.png")
            frame=inventory(root)
            self.assertEqual(len(frame),6)
            self.assertTrue(all("unique" in path for path in frame.path))
            self.assertEqual(len(frame.attrs["excluded_images"]),2)
            self.assertEqual({x["reason"] for x in frame.attrs["excluded_images"]},{"conflicting labels"})
            self.assertEqual(frame.attrs["raw_class_distribution"]["paper"],2)
