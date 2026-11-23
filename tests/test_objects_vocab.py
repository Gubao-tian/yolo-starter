"""全物体类别词表测试：英文词表 / 中文名 / 分组 / 汇总。"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from yolo_starter import objects_vocab as ov  # noqa: E402


class TestVocab(unittest.TestCase):
    def test_words_is_nonempty_unique_lowercase(self):
        self.assertGreater(len(ov.WORDS), 40)
        self.assertEqual(len(ov.WORDS), len(set(ov.WORDS)))
        self.assertTrue(all(w == w.strip().lower() and " " not in w or w in ov.MULTIWORD
                            for w in ov.WORDS))

    def test_key_objects_present(self):
        # 用户截图饭盒图里出现、且 COCO 80 类漏掉的对象
        for w in ("bread", "plate", "bowl", "broccoli", "pineapple", "nuts", "meatball"):
            self.assertIn(w, ov.WORDS, w)

    def test_household_objects_present(self):
        # 用户截图卧室图里出现、但旧词表漏掉的家居物品
        for w in ("basket", "curtain", "cabinet", "dresser", "bookshelf",
                  "clothes", "carpet", "frame", "shoes", "blanket", "nightstand"):
            self.assertIn(w, ov.WORDS, w)

    def test_household_objects_chinese_names(self):
        expected = {
            "basket": "篮子", "curtain": "窗帘", "cabinet": "柜子",
            "dresser": "抽屉柜", "bookshelf": "书架", "clothes": "衣服",
            "carpet": "地毯", "frame": "相框", "shoes": "鞋",
            "blanket": "毯子", "nightstand": "床头柜",
        }
        for word, zh in expected.items():
            self.assertEqual(ov.zh_name(word), zh, word)
            self.assertEqual(ov.group_of(word), "家居", word)

    def test_every_word_has_chinese_name(self):
        for w in ov.WORDS:
            self.assertIn(w, ov.ZH_NAMES, w)
            self.assertTrue(ov.ZH_NAMES[w].strip(), w)

    def test_groups_cover_all_words(self):
        seen = [w for group in ov.GROUPS.values() for w in group]
        self.assertEqual(set(seen), set(ov.WORDS))
        self.assertEqual(len(seen), len(set(seen)))

    def test_group_of(self):
        self.assertEqual(ov.group_of("bread"), "食物")
        self.assertEqual(ov.group_of("bowl"), "餐具")
        self.assertEqual(ov.group_of("unknown_xyz"), "其他")

    def test_finetuned_model_class_names_have_chinese(self):
        # 微调模型 data.yaml 用单数类名，与开放词表的复数不同，需别名映射
        self.assertEqual(ov.zh_name("nut"), "坚果")
        self.assertEqual(ov.zh_name("NUT"), "坚果")


class TestSummarizeObjects(unittest.TestCase):
    def test_counts_dedup_and_sorts(self):
        dets = [
            {"label": "bread", "score": 0.39},
            {"label": "bread", "score": 0.33},
            {"label": "plate", "score": 0.61},
            {"label": "bowl", "score": 0.63},
            {"label": "person", "score": 0.90},
        ]
        rows = ov.summarize_objects(dets)
        self.assertEqual([r["label"] for r in rows], ["bread", "bowl", "person", "plate"])
        counts = {r["label"]: r["count"] for r in rows}
        self.assertEqual(counts, {"bread": 2, "bowl": 1, "plate": 1, "person": 1})
        self.assertEqual(rows[0]["zh"], "面包")
        self.assertEqual(rows[0]["group"], "食物")
        self.assertEqual(rows[0]["best_score"], 0.39)

    def test_empty_input(self):
        self.assertEqual(ov.summarize_objects([]), [])

    def test_unknown_label_falls_back(self):
        rows = ov.summarize_objects([{"label": "mystery", "score": 0.5}])
        self.assertEqual(rows[0]["zh"], "mystery")
        self.assertEqual(rows[0]["group"], "其他")


if __name__ == "__main__":
    unittest.main()
