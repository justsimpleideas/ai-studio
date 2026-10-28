import unittest

from image_engine.training.subject import (CARS, GENERAL, MAX_FIELDS, MAX_LEVELS, PRESETS, SubjectError,
                                           from_project, parse_subject, slug)


def define(**changes):
    base = {"noun": "wristwatch", "levels": [{"label": "Brand", "required": True}, {"label": "Model"}],
            "fields": [{"label": "View", "question": "which side faces the camera", "choices": ["front", "back"]}]}
    return {**base, **changes}


class SubjectTests(unittest.TestCase):
    def test_keys_come_from_labels(self):
        self.assertEqual(slug("Body style"), "body_style")
        self.assertEqual(slug("  Dial / Colour! "), "dial_colour")
        self.assertEqual(slug("μάρκα όνομα"), "μάρκα_όνομα")  # any language's letters
        subject = parse_subject(define(fields=[{"label": "Dial colour", "question": "q"}]))
        self.assertEqual(subject.fact_fields, ("brand", "model"))
        self.assertEqual(subject.required_facts, ("brand",))
        self.assertEqual(subject.fields[0].name, "dial_colour")
        self.assertIsNone(subject.fields[0].choices)
        self.assertEqual(subject.fields[0].phrase, "{value}")

    def test_round_trip_through_json(self):
        for preset in PRESETS.values():
            self.assertEqual(parse_subject(preset.to_json()), preset)
        custom = parse_subject(define(trigger="my-watch_1", preset="custom"))
        self.assertEqual(parse_subject(custom.to_json()), custom)

    def test_label_fields_are_levels_and_consistent_fields(self):
        self.assertEqual(CARS.label_fields, ("make", "model", "generation", "years", "body_style"))
        self.assertEqual(CARS.consistent_fields, ("body_style",))
        self.assertEqual(GENERAL.consistent_fields, ())

    def test_presets_are_not_about_one_topic(self):
        self.assertEqual(GENERAL.noun, "")
        text = repr(GENERAL.to_json()).lower()
        for word in ("car", "vehicle", "make"):
            self.assertNotIn(f" {word}", text.replace("{noun}", ""))

    def test_old_projects_were_about_cars(self):
        self.assertEqual(from_project({"schema": "cars"}), CARS)
        self.assertEqual(from_project({}), GENERAL)
        self.assertEqual(from_project({"schema": "cars", "subject": define()}).noun, "wristwatch")

    def test_bad_definitions_are_refused_with_a_reason(self):
        bad = [
            ("not an object", []),
            ("unknown setting", define(colour="red")),
            ("no levels", define(levels=[])),
            ("too many levels", define(levels=[{"label": f"L{i}", "required": True} for i in range(MAX_LEVELS + 1)])),
            ("no required level", define(levels=[{"label": "Brand"}])),
            ("an empty label", define(levels=[{"label": " ", "required": True}])),
            ("a label of symbols", define(levels=[{"label": "!!", "required": True}])),
            ("the same name twice", define(levels=[{"label": "Brand", "required": True}, {"label": "brand"}])),
            ("a field named like a level", define(fields=[{"label": "Model", "question": "q"}])),
            ("unknown as a name", define(fields=[{"label": "Unknown", "question": "q"}])),
            ("a trigger with a space", define(trigger="my watch")),
            ("a trigger with punctuation", define(trigger="watch!")),
            ("a field without a question", define(fields=[{"label": "View", "choices": ["a", "b"]}])),
            ("a single choice", define(fields=[{"label": "View", "question": "q", "choices": ["front"]}])),
            ("a choice twice", define(fields=[{"label": "View", "question": "q", "choices": ["Front", "front"]}])),
            ("unknown as a choice", define(fields=[{"label": "View", "question": "q", "choices": ["a", "unknown"]}])),
            ("a phrase without {value}", define(fields=[{"label": "View", "question": "q", "phrase": "view"}])),
            ("a phrase with {value} twice", define(fields=[{"label": "V", "question": "q", "phrase": "{value} {value}"}])),
            ("consistent free text", define(fields=[{"label": "Colour", "question": "q", "consistent": True}])),
            ("a flag that is not a flag", define(levels=[{"label": "Brand", "required": "yes"}])),
            ("too many fields", define(fields=[{"label": f"F{i}", "question": "q"} for i in range(MAX_FIELDS + 1)])),
            ("a long noun", define(noun="x" * 41)),
            ("an unknown preset", define(preset="boats")),
        ]
        for why, data in bad:
            with self.assertRaises(SubjectError, msg=why):
                parse_subject(data)

    def test_text_is_tidied(self):
        subject = parse_subject(define(noun="  wrist   watch ", fields=[
            {"label": "View", "question": " which   side ", "choices": [" Front ", "BACK"]}]))
        self.assertEqual(subject.noun, "wrist watch")
        self.assertEqual(subject.fields[0].hint, "which side")
        self.assertEqual(subject.fields[0].choices, ("front", "back"))


if __name__ == "__main__":
    unittest.main()
