"""Captions: facts from the folders, a description from the captioning model, and flags.

The captioning model only describes what it can see, answering the description
fields of the project's subject (view, colour, setting...; subject.py). It is never
asked to name the subject; the name comes from the dataset's facts
(docs/training.md, "Caption").

Vision models state wrong answers confidently, so a description is never taken
on the model's word. It is flagged for review when any of these trip:

- disagree:<field>    two passes with differently worded prompts answered differently
- low_confidence:<f>  the tokens of the answer had low probability
- unknown:<field>     no usable answer for a field that matters
- conflict:<field>    the answer contradicts a fact from labels.json
- outlier:<field>     most photos of the same subject say one thing, this one another
                      (often a photo filed in the wrong folder)
- single_pass         only one of the two replies was usable, so nothing cross-checks it
- unparseable         neither reply was the JSON asked for

Every photo is also asked whether it shows the subject at all (RELEVANCE): a
toy or a picture of it, no subject at all, or the subject small in the
background marks it off_subject, and it is left out of training until the user
includes it; only a part or the inside of it is flagged partial, for a look.

Flags only order the review queue. To learn whether the unflagged rest can be
trusted, a random audit sample of it is reviewed too, and its error rate is
reported (`audit_estimate`).

This module is model-agnostic: it builds the prompts and judges the answers;
the captioning backend runs the model.
"""

from __future__ import annotations

import json
import math
import os
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .dataset import Dataset
from .subject import FieldSpec, Subject

UNKNOWN = "unknown"
STORE_VERSION = 1
LOW_CONFIDENCE = 0.6
OUTLIER_MIN_PHOTOS = 5
OUTLIER_MAJORITY = 0.7
DEFAULT_AUDIT_SIZE = 50
STATUSES = ("auto", "accepted", "edited", "excluded")
SEVERITY = {"off_subject": 6, "unparseable": 5, "conflict": 4, "outlier": 3, "disagree": 2, "low_confidence": 2,
            "single_pass": 2, "unknown": 1, "partial": 1}

# Whether a photo is of the subject at all: asked about every photo, before the subject's own
# fields. The keys are reserved (subject.py). Answers the model gives about these are never
# written into captions.
SHOWN, SIZE = "subject_shown", "subject_size"
REAL_WHOLE, REAL_PART, INSIDE = "the real thing, whole", "the real thing, part or close-up", "the inside of it"
TOY, PICTURE, ABSENT = "a toy, model or replica", "a picture of it", "not in the photo"
MAIN, SHARED, SMALL = "the main subject", "one of several things", "small or in the background"
RELEVANCE = (
    FieldSpec(SHOWN, (REAL_WHOLE, REAL_PART, INSIDE, TOY, PICTURE, ABSENT), True,
              "what the photo shows of the {noun}: the real thing (whole, or only a part or close-up), its inside, "
              "a toy, model or replica of it, a picture of it (a drawing, poster or screen), or no {noun} at all",
              label="Shows"),
    FieldSpec(SIZE, (MAIN, SHARED, SMALL), True, "how the {noun} sits in the photo", label="Size"),
)
OFF_SUBJECT = {SHOWN: {TOY, PICTURE, ABSENT}, SIZE: {SMALL}}
PARTIAL = {SHOWN: {REAL_PART, INSIDE}}


def all_specs(subject: Subject) -> tuple[FieldSpec, ...]:
    """What the captioning model is asked: relevance first, then the subject's own fields."""
    return RELEVANCE + subject.fields


# Common synonyms. An alias only matters when what it maps to is one of the field's choices.
ALIASES = {
    "sport utility vehicle": "suv", "crossover": "suv", "truck": "pickup", "pickup truck": "pickup",
    "estate": "wagon", "station wagon": "wagon", "cabriolet": "convertible", "saloon": "sedan",
    "three quarter front": "front three-quarter", "front 3/4": "front three-quarter",
    "three quarter rear": "rear three-quarter", "rear 3/4": "rear three-quarter", "profile": "side",
}


def _norm(value: str) -> str:
    return " ".join(value.lower().replace("_", " ").replace("-", " ").split())


def normalise(spec: FieldSpec, value: object) -> str:
    """The canonical answer for a field, or UNKNOWN."""
    if not isinstance(value, str) or not value.strip():
        return UNKNOWN
    text = _norm(value)
    if text == UNKNOWN:
        return UNKNOWN
    text = _norm(ALIASES.get(text, text))
    if spec.choices is None:
        return text[:40]
    for choice in spec.choices:
        if _norm(choice) == text:
            return choice
    return UNKNOWN


def prompts(subject: Subject) -> tuple[str, str]:
    """Two differently worded prompts for the same JSON. Disagreement between them is a flag."""
    specs = all_specs(subject)
    keys = ", ".join(f'"{s.name}"' for s in specs)
    lines = []
    for s in specs:
        allowed = " | ".join(s.choices) if s.choices else "free text"
        lines.append(f'- "{s.name}": {subject.hint(s)}. One of: {allowed}.' if s.choices
                     else f'- "{s.name}": {subject.hint(s)} ({allowed}).')
    rules = "\n".join(lines)
    first = (f"Describe the {subject.thing} in this photo. Reply with only a JSON object with the keys "
             f"{keys}.\n{rules}\nUse \"unknown\" for anything you cannot see clearly. "
             "Do not name the brand, the model or any other specific name.")
    second = (f"Look at the photo and fill in this JSON form about the {subject.thing}. Answer each field "
              "from what is visible, and write \"unknown\" when unsure.\n"
              f"{rules}\nOutput the JSON object only, with exactly these keys: {keys}.")
    return first, second


def parse_answer(text: str, specs: tuple[FieldSpec, ...]) -> dict[str, str] | None:
    """The fields of the first JSON object in a reply, normalised; None if there is none."""
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end < start:
        return None
    try:
        data = json.loads(text[start:end + 1])
    except ValueError:
        return None
    if not isinstance(data, dict):
        return None
    return {s.name: normalise(s, data.get(s.name)) for s in specs}


@dataclass
class CaptionRecord:
    item_id: str
    #: The model's answer (first prompt), normalised.
    description: dict[str, str]
    flags: list[str]
    status: str = "auto"
    #: Fields the user corrected; they win over `description`.
    edits: dict[str, str] = field(default_factory=dict)
    #: In the audit sample of unflagged captions.
    audit: bool = False
    #: Audit outcome: True if the user had to correct it (edit or exclude).
    audit_wrong: bool | None = None
    #: Whether the photo shows the subject (RELEVANCE); None until the captioning model has looked.
    relevant: bool | None = None
    #: The captioning model, prompts and subject that produced `description` ("" for facts only).
    by: str = ""

    @property
    def excluded(self) -> bool:
        """Left out of training: by the user, or as off subject until the user includes or edits it."""
        return self.status == "excluded" or (self.relevant is False and self.status == "auto")

    @property
    def severity(self) -> int:
        return sum(SEVERITY.get(flag.split(":", 1)[0], 1) for flag in self.flags)

    def effective(self) -> dict[str, str]:
        return {**self.description, **self.edits}

    def current(self, specs: tuple[FieldSpec, ...]) -> dict[str, str]:
        """The effective answers for the subject's fields as they are now. Answers to fields
        removed since stay stored, so nothing is lost if a field comes back, but are left out."""
        values = self.effective()
        return {s.name: normalise(s, values.get(s.name, UNKNOWN)) for s in specs}


def judge(item_id: str, replies: tuple[str, str], specs: tuple[FieldSpec, ...], facts: dict[str, str],
          confidence: dict[str, float] | None = None) -> CaptionRecord:
    """Turn the two replies for one photo into a record with its per-photo flags.

    `confidence` maps a field to the lowest token probability within its answer
    in the first reply, when the backend can provide it."""
    first, second = (parse_answer(r, specs) for r in replies)
    if first is None:
        if second is None:
            return CaptionRecord(item_id, {s.name: UNKNOWN for s in specs}, ["unparseable"])
        first, second = second, None
    relevant = None
    flags = []
    for spec in specs:
        value = first[spec.name]
        if value == UNKNOWN:
            if spec.important:
                flags.append(f"unknown:{spec.name}")
            continue
        if second is not None and spec.choices is not None and second[spec.name] != value:
            flags.append(f"disagree:{spec.name}")
        if confidence and confidence.get(spec.name, 1.0) < LOW_CONFIDENCE:
            flags.append(f"low_confidence:{spec.name}")
        fact = facts.get(spec.name)
        if fact and normalise(spec, fact) not in (value, UNKNOWN):
            flags.append(f"conflict:{spec.name}")
    if second is None:
        flags.append("single_pass")
    if SHOWN in first:
        off = [name for name, values in OFF_SUBJECT.items() if first.get(name) in values]
        flags += [f"off_subject:{name}" for name in off]
        flags += [f"partial:{name}" for name, values in PARTIAL.items() if first.get(name) in values]
        relevant = not off
    return CaptionRecord(item_id, first, flags, relevant=relevant)


def flag_outliers(records: list[CaptionRecord], dataset: Dataset, subject: Subject) -> None:
    """Flag answers that disagree with a clear majority of photos of the same subject.

    Only for the subject's consistent fields (a car's body style, say); photos show
    the same subject when all their folder facts match."""
    facts = {item.id: tuple(item.facts.get(f, "") for f in subject.fact_fields) for item in dataset.items}
    groups: dict[tuple, list[CaptionRecord]] = {}
    for rec in records:
        groups.setdefault(facts.get(rec.item_id, ()), []).append(rec)
    for spec in subject.fields:
        if not spec.consistent:
            continue
        answer = lambda r: r.effective().get(spec.name, UNKNOWN)  # noqa: E731
        for members in groups.values():
            known = [r for r in members if answer(r) != UNKNOWN]
            if len(known) < OUTLIER_MIN_PHOTOS:
                continue
            counts: dict[str, int] = {}
            for r in known:
                counts[answer(r)] = counts.get(answer(r), 0) + 1
            top, top_count = max(counts.items(), key=lambda kv: kv[1])
            if top_count / len(known) < OUTLIER_MAJORITY:
                continue
            for r in known:
                flag = f"outlier:{spec.name}"
                if answer(r) != top and flag not in r.flags:
                    r.flags.append(flag)


def review_queue(records: list[CaptionRecord]) -> list[CaptionRecord]:
    """Unreviewed records that need a look: flagged ones, worst first, then the audit sample."""
    pending = [r for r in records if r.status == "auto" and (r.flags or r.audit)]
    return sorted(pending, key=lambda r: (not r.flags, -r.severity, -len(r.flags), r.item_id))


def choose_audit(records: list[CaptionRecord], size: int = DEFAULT_AUDIT_SIZE, seed: int = 0) -> list[str]:
    """Mark a random sample of the unflagged, unreviewed records for audit."""
    pool = sorted((r for r in records if not r.flags and r.status == "auto"), key=lambda r: r.item_id)
    chosen = random.Random(seed).sample(pool, min(size, len(pool)))
    for r in chosen:
        r.audit = True
    return [r.item_id for r in chosen]


def review(record: CaptionRecord, action: str, edits: dict[str, str] | None,
           specs: tuple[FieldSpec, ...]) -> None:
    """Apply a user decision: "accept", "edit" (with field values) or "exclude"."""
    if action == "accept":
        record.status = "accepted"
    elif action == "exclude":
        record.status = "excluded"
    elif action == "edit":
        by_name = {s.name: s for s in specs}
        clean = {}
        for name, value in (edits or {}).items():
            if name not in by_name:
                raise ValueError(f"Unknown field: {name}")
            clean[name] = normalise(by_name[name], value)
            if clean[name] == UNKNOWN and value.strip().lower() != UNKNOWN:
                allowed = ", ".join(by_name[name].choices or ())
                raise ValueError(f"{name}: {value!r} is not one of {allowed}")
        record.edits.update(clean)
        record.status = "edited"
    else:
        raise ValueError(f"Unknown review action: {action}")
    if record.audit:
        record.audit_wrong = record.status != "accepted"


def audit_estimate(records: list[CaptionRecord]) -> dict | None:
    """Share of audited (unflagged) captions the user had to correct, with a 95% Wilson interval."""
    checked = [r for r in records if r.audit and r.audit_wrong is not None]
    n = len(checked)
    if n == 0:
        return None
    wrong = sum(1 for r in checked if r.audit_wrong)
    p, z = wrong / n, 1.96
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    margin = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return {"checked": n, "wrong": wrong, "rate": p,
            "low": max(0.0, centre - margin), "high": min(1.0, centre + margin)}


def caption_text(subject: Subject, facts: dict[str, str], record: CaptionRecord) -> str:
    """The training caption of one photo, as the subject is defined now."""
    return subject.caption(facts, record.current(subject.fields), UNKNOWN)


@dataclass
class CaptionStore:
    #: The subject's preset name, for reference only; the project holds the definition.
    schema: str
    #: Identifies the captioning model and prompts, so a store built by another one is not mixed in.
    captioner: str
    records: list[CaptionRecord]

    def by_item(self) -> dict[str, CaptionRecord]:
        return {r.item_id: r for r in self.records}

    def save(self, path: Path) -> None:
        data = {"version": STORE_VERSION, "schema": self.schema, "captioner": self.captioner,
                "records": [asdict(r) for r in self.records]}
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(data, indent=1), encoding="utf-8")
        os.replace(tmp, path)

    @classmethod
    def load(cls, path: Path) -> "CaptionStore":
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("version") != STORE_VERSION:
            raise ValueError(f"Unsupported caption store version: {data.get('version')!r}")
        return cls(schema=data["schema"], captioner=data["captioner"],
                   records=[CaptionRecord(**r) for r in data["records"]])
