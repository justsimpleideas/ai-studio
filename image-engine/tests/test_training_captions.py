import json
import tempfile
import unittest
from pathlib import Path

from image_engine.training import captions as cap
from image_engine.training.captions import (UNKNOWN, CaptionRecord, CaptionStore, audit_estimate, caption_text,
                                            choose_audit, flag_outliers, judge, parse_answer, prompts, review,
                                            review_queue)
from image_engine.training.dataset import Dataset, Item
from image_engine.training.subject import CARS, GENERAL, parse_subject

SPECS = CARS.fields
GOOD = {"view": "front three-quarter", "colour": "Silver", "body_style": "coupe", "setting": "wet city street",
        "lighting": "at night", "framing": "whole car in frame", "occlusion": "not hidden"}


def reply(**changes):
    return "Sure! Here it is:\n```json\n" + json.dumps({**GOOD, **changes}) + "\n```"


def item(item_id, **facts):
    return Item(id=item_id, path=f"{item_id}.jpg", size=1, mtime_ns=0, sha256=item_id * 4, width=800,
                height=600, dhash="0" * 16, facts=facts, holdout=False)


class ParseTests(unittest.TestCase):
    def test_parses_json_inside_chatter_and_normalises(self):
        got = parse_answer(reply(view="Front-Three-Quarter", body_style="Station Wagon"), SPECS)
        self.assertEqual(got["view"], "front three-quarter")
        self.assertEqual(got["body_style"], "wagon")
        self.assertEqual(got["colour"], "silver")

    def test_off_list_and_missing_answers_become_unknown(self):
        text = json.dumps({"view": "diagonal-ish", "body_style": "spaceship"})
        got = parse_answer(text, SPECS)
        self.assertEqual(got["view"], UNKNOWN)
        self.assertEqual(got["body_style"], UNKNOWN)
        self.assertEqual(got["lighting"], UNKNOWN)
        self.assertEqual(set(got), {s.name for s in SPECS})

    def test_not_json(self):
        self.assertIsNone(parse_answer("I see a red car.", SPECS))
        self.assertIsNone(parse_answer("{broken", SPECS))
        self.assertIsNone(parse_answer("[1, 2]", SPECS))

    def test_prompts_differ_list_every_field_and_never_ask_for_a_name(self):
        first, second = prompts(CARS)
        self.assertNotEqual(first, second)
        for p in (first, second):
            for spec in SPECS:
                self.assertIn(f'"{spec.name}"', p)
            self.assertIn("front three-quarter", p)
            self.assertIn("car", p)
        self.assertIn("Do not name the brand, the model or any other specific name", first)

    def test_prompts_speak_of_the_users_subject(self):
        watch = parse_subject({**GENERAL.to_json(), "noun": "wristwatch"})
        first, second = prompts(watch)
        self.assertIn("Describe the wristwatch in this photo", first)
        self.assertIn("about the wristwatch", second)
        self.assertIn("which side of the wristwatch faces the camera", first)  # "{noun}" in a question
        self.assertNotIn("{noun}", first + second)
        self.assertNotIn("car", (first + second).replace("cannot", ""))
        nameless = prompts(GENERAL)[0]
        self.assertIn("Describe the main subject in this photo", nameless)


class JudgeTests(unittest.TestCase):
    def test_agreeing_confident_answer_has_no_flags(self):
        rec = judge("a", (reply(), reply(setting="rainy street")), SPECS, {},
                    confidence={"view": 0.97, "body_style": 0.9})
        self.assertEqual(rec.flags, [])  # free-text fields may be worded differently
        self.assertEqual(rec.description["body_style"], "coupe")

    def test_each_flag(self):
        rec = judge("a", (reply(view="side"), reply(view="rear")), SPECS, {})
        self.assertEqual(rec.flags, ["disagree:view"])
        rec = judge("a", (reply(), reply()), SPECS, {}, confidence={"body_style": 0.3})
        self.assertEqual(rec.flags, ["low_confidence:body_style"])
        rec = judge("a", (reply(view="unknown", lighting="unknown"),) * 2, SPECS, {})
        self.assertEqual(rec.flags, ["unknown:view"])  # lighting is not important
        rec = judge("a", (reply(),) * 2, SPECS, {"body_style": "SUV"})
        self.assertEqual(rec.flags, ["conflict:body_style"])
        rec = judge("a", (reply(),) * 2, SPECS, {"body_style": "Coupe"})
        self.assertEqual(rec.flags, [])

    def test_unusable_replies(self):
        rec = judge("a", ("no idea", "still no"), SPECS, {})
        self.assertEqual(rec.flags, ["unparseable"])
        self.assertTrue(all(v == UNKNOWN for v in rec.description.values()))
        rec = judge("a", ("no idea", reply()), SPECS, {})
        self.assertEqual(rec.flags, ["single_pass"])
        self.assertEqual(rec.description["view"], "front three-quarter")


class OutlierTests(unittest.TestCase):
    def test_minority_answer_within_one_subject_is_flagged(self):
        items = [item(f"{i:02}", make="Ex", model="Coupe") for i in range(8)]
        items += [item(f"o{i}", make="Ex", model="Pickup") for i in range(3)]  # too few to judge
        dataset = Dataset(root="/r", schema="cars", min_side=512, items=items)
        records = [judge(f"{i:02}", (reply(),) * 2, SPECS, {}) for i in range(7)]
        records.append(judge("07", (reply(body_style="suv"),) * 2, SPECS, {}))
        records += [judge(f"o{i}", (reply(body_style=bs),) * 2, SPECS, {})
                    for i, bs in enumerate(("pickup", "suv", "van"))]
        flag_outliers(records, dataset, CARS)
        flagged = [r.item_id for r in records if r.flags]
        self.assertEqual(flagged, ["07"])
        self.assertEqual(records[7].flags, ["outlier:body_style"])
        flag_outliers(records, dataset, CARS)  # idempotent
        self.assertEqual(records[7].flags, ["outlier:body_style"])

    def test_no_clear_majority_no_flag(self):
        items = [item(f"{i}", make="Ex", model="M") for i in range(6)]
        dataset = Dataset(root="/r", schema="cars", min_side=512, items=items)
        styles = ["coupe", "coupe", "coupe", "convertible", "convertible", "coupe"]
        records = [judge(str(i), (reply(body_style=s),) * 2, SPECS, {}) for i, s in enumerate(styles)]
        flag_outliers(records, dataset, CARS)  # 4 of 6 = 67% < 70%
        self.assertTrue(all(not r.flags for r in records))

    def test_only_consistent_fields_are_checked(self):
        items = [item(f"{i}", make="Ex", model="M") for i in range(8)]
        dataset = Dataset(root="/r", schema="cars", min_side=512, items=items)
        records = [judge(str(i), (reply(view="side" if i else "rear"),) * 2, SPECS, {}) for i in range(8)]
        flag_outliers(records, dataset, CARS)  # one rear view among side views is normal
        self.assertTrue(all(not r.flags for r in records))


class ReviewTests(unittest.TestCase):
    def records(self):
        recs = [CaptionRecord(f"{i:03}", dict(GOOD), []) for i in range(100)]
        recs[5].flags = ["unknown:view"]
        recs[6].flags = ["conflict:body_style", "disagree:view"]
        recs[7].flags = ["disagree:view"]
        return recs

    def test_queue_worst_first_then_audit_sample(self):
        recs = self.records()
        audit = choose_audit(recs, size=10, seed=1)
        self.assertEqual(len(audit), 10)
        self.assertFalse({"005", "006", "007"} & set(audit))
        self.assertEqual(choose_audit(self.records(), size=10, seed=1), audit)  # reproducible
        queue = review_queue(recs)
        self.assertEqual([r.item_id for r in queue[:3]], ["006", "007", "005"])
        self.assertEqual(sorted(r.item_id for r in queue[3:]), sorted(audit))
        review(queue[0], "accept", None, SPECS)
        self.assertNotIn(queue[0], review_queue(recs))

    def test_edits_win_and_are_validated(self):
        rec = CaptionRecord("a", dict(GOOD), ["disagree:view"])
        review(rec, "edit", {"view": "Rear", "colour": "Dark  Green"}, SPECS)
        self.assertEqual(rec.status, "edited")
        self.assertEqual(rec.effective()["view"], "rear")
        self.assertEqual(rec.effective()["colour"], "dark green")
        self.assertEqual(rec.description["view"], "front three-quarter")  # the model's answer is kept
        with self.assertRaises(ValueError):
            review(rec, "edit", {"view": "sideways"}, SPECS)
        with self.assertRaises(ValueError):
            review(rec, "edit", {"wheels": "4"}, SPECS)
        with self.assertRaises(ValueError):
            review(rec, "approve", None, SPECS)

    def test_audit_estimate(self):
        self.assertIsNone(audit_estimate(self.records()))
        recs = self.records()
        audited = [r for r in recs if r.item_id in choose_audit(recs, size=50, seed=0)]
        for i, r in enumerate(audited):
            review(r, "edit" if i < 5 else "accept", {"colour": "red"} if i < 5 else None, SPECS)
        est = audit_estimate(recs)
        self.assertEqual((est["checked"], est["wrong"]), (50, 5))
        self.assertAlmostEqual(est["rate"], 0.1)
        self.assertAlmostEqual(est["low"], 0.0435, places=3)
        self.assertAlmostEqual(est["high"], 0.2136, places=3)
        flagged = recs[5]
        review(flagged, "edit", {"view": "side"}, SPECS)
        self.assertIsNone(flagged.audit_wrong)  # flagged records are not part of the estimate
        self.assertEqual(audit_estimate(recs)["checked"], 50)


class CaptionTextTests(unittest.TestCase):
    def test_full_caption(self):
        facts = {"make": "Example Motors", "model": "Coupe GT", "generation": "Mk2", "years": "2019-2023"}
        rec = CaptionRecord("a", {s.name: cap.normalise(s, GOOD[s.name]) for s in SPECS}, [])
        self.assertEqual(caption_text(CARS, facts, rec),
                         "Example Motors Coupe GT Mk2 2019-2023, car, front three-quarter view, silver, coupe, "
                         "wet city street, at night, whole car in frame, not hidden")

    def test_unknowns_are_left_out(self):
        rec = CaptionRecord("a", {s.name: UNKNOWN for s in SPECS} | {"body_style": "suv",
                                                                       "occlusion": "partly hidden"}, [])
        self.assertEqual(caption_text(CARS, {"make": "Ex", "model": "Trail"}, rec), "Ex Trail, car, suv, partly hidden")
        rec = CaptionRecord("a", {s.name: UNKNOWN for s in SPECS}, [])
        self.assertEqual(caption_text(CARS, {"make": "Ex", "model": "Trail"}, rec), "Ex Trail, car")

    def test_a_users_subject(self):
        watch = parse_subject({
            "noun": "wristwatch", "trigger": "mywatch",
            "levels": [{"label": "Brand", "required": True}, {"label": "Model", "required": True},
                       {"label": "Reference", "in_caption": False}],
            "fields": [{"label": "View", "question": "which side faces the camera", "choices": ["front", "back"],
                        "phrase": "{value} view"},
                       {"label": "Dial colour", "question": "the dial colour", "phrase": "{value} dial"}],
        })
        rec = CaptionRecord("a", {"view": "front", "dial_colour": "black"}, [])
        facts = {"brand": "Seiko", "model": "SKX007", "reference": "SKX007K2"}
        self.assertEqual(caption_text(watch, facts, rec), "mywatch, Seiko SKX007, wristwatch, front view, black dial")

    def test_answers_to_fields_removed_later_are_kept_but_not_used(self):
        rec = CaptionRecord("a", {"view": "side", "colour": "red", "gone": "x"}, [])
        rec.edits = {"view": "rear"}
        fewer = parse_subject({**CARS.to_json(), "fields": [f for f in CARS.to_json()["fields"]
                                                             if f["key"] != "colour"]})
        self.assertEqual(rec.current(fewer.fields)["view"], "rear")
        self.assertNotIn("colour", rec.current(fewer.fields))
        self.assertEqual(caption_text(fewer, {"make": "Ex", "model": "M"}, rec), "Ex M, car, rear view")
        self.assertEqual(rec.effective()["colour"], "red")  # still stored

    def test_a_field_whose_choices_changed_reads_as_unknown(self):
        rec = CaptionRecord("a", {"view": "front three-quarter"}, [])
        self.assertEqual(rec.current(GENERAL.fields)["view"], UNKNOWN)  # not one of General's views
        self.assertEqual(rec.current(GENERAL.fields)["colour"], UNKNOWN)  # a field the record never had


class StoreTests(unittest.TestCase):
    def test_round_trip(self):
        rec = judge("a", (reply(view="side"), reply()), SPECS, {})
        review(rec, "edit", {"view": "side"}, SPECS)
        store = CaptionStore(schema="cars", captioner="vlm@1|prompts@1", records=[rec])
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "captions.json"
            store.save(path)
            loaded = CaptionStore.load(path)
        self.assertEqual(loaded, store)
        self.assertEqual(loaded.by_item()["a"].effective()["view"], "side")


if __name__ == "__main__":
    unittest.main()
