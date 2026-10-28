"""What a training project is about: its subject definition (docs/training.md, "Subject").

The user defines it and it is stored in the project's project.json. Nothing about
any one topic is built into the engine: the presets below ("General", "Cars") are
only starting points that the user edits.

- noun     the class word the subject belongs to ("wristwatch"). Every caption
           carries it, so the LoRA attaches to a concept the model already knows,
           and the captioning prompts ask about it.
- trigger  an optional made-up word put first in every caption ("mywatch"), so a
           prompt can call the LoRA up on purpose.
- levels   the folder levels below the photo folder, in order (Brand / Model).
           Each fills one fact of every photo below it; a labels.json in any
           folder can set them exactly. Photos missing a `required` level are not
           trained on. The `in_caption` levels make up the subject's name.
- fields   what the captioning model is asked about each photo (view, colour,
           setting...). A field with choices is answered from that list and can be
           cross-checked; one without is free text. A `consistent` field is the same
           for every photo of one subject: labels.json may set it, and an answer
           that contradicts it or most of its siblings is flagged.

Keys are derived from labels ("Body style" -> "body_style"); labels.json uses them.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

MAX_LEVELS = 4
MAX_FIELDS = 10
MAX_CHOICES = 20
LABEL_MAX = 30
NOUN_MAX = 40
TRIGGER_MAX = 32
HINT_MAX = 200
CHOICE_MAX = 40
PHRASE_MAX = 60
VALUE = "{value}"
NOUN = "{noun}"
PRESET_NAMES = ("general", "cars", "custom")
# Keys the captioning prompts use for themselves (captions.RELEVANCE), and the word for no answer.
RESERVED_KEYS = {"unknown", "subject_shown", "subject_size"}
TRIGGER = re.compile(r"^[\w-]*$", re.UNICODE)


class SubjectError(ValueError):
    """The definition is not valid. The message is safe to show to the user."""


@dataclass(frozen=True)
class Level:
    key: str
    label: str
    required: bool = False
    in_caption: bool = True


@dataclass(frozen=True)
class FieldSpec:
    name: str
    #: Allowed answers; None for free text.
    choices: tuple[str, ...] | None
    #: An "unknown" here is flagged: without it the caption misleads.
    important: bool
    #: What the captioning model is asked; "{noun}" stands for the subject's noun.
    hint: str
    label: str = ""
    #: The same for every photo of one subject (see the module docstring).
    consistent: bool = False
    #: How a known answer is written into the caption.
    phrase: str = VALUE


@dataclass(frozen=True)
class Subject:
    preset: str
    noun: str
    trigger: str
    levels: tuple[Level, ...]
    fields: tuple[FieldSpec, ...]

    @property
    def fact_fields(self) -> tuple[str, ...]:
        return tuple(level.key for level in self.levels)

    @property
    def required_facts(self) -> tuple[str, ...]:
        return tuple(level.key for level in self.levels if level.required)

    @property
    def consistent_fields(self) -> tuple[str, ...]:
        return tuple(f.name for f in self.fields if f.consistent)

    @property
    def label_fields(self) -> tuple[str, ...]:
        """What a labels.json may set."""
        return self.fact_fields + self.consistent_fields

    @property
    def thing(self) -> str:
        """How prompts refer to the subject."""
        return self.noun or "main subject"

    def hint(self, spec: FieldSpec) -> str:
        return spec.hint.replace(NOUN, self.thing)

    def name(self, facts: dict[str, str]) -> str:
        return " ".join(facts[level.key] for level in self.levels if level.in_caption and facts.get(level.key))

    def caption(self, facts: dict[str, str], description: dict[str, str], unknown: str = "unknown") -> str:
        """e.g. "mywatch, Seiko SKX007, wristwatch, front view, black dial, on a wrist"."""
        parts = [self.trigger, self.name(facts), self.noun]
        for spec in self.fields:
            value = description.get(spec.name, unknown)
            if value and value != unknown:
                parts.append(spec.phrase.replace(VALUE, value))
        return ", ".join(p for p in parts if p)

    def to_json(self) -> dict:
        return {
            "preset": self.preset, "noun": self.noun, "trigger": self.trigger,
            "levels": [{"key": lv.key, "label": lv.label, "required": lv.required, "in_caption": lv.in_caption}
                       for lv in self.levels],
            "fields": [{"key": f.name, "label": f.label, "question": f.hint, "choices": list(f.choices or []),
                        "important": f.important, "consistent": f.consistent, "phrase": f.phrase}
                       for f in self.fields],
        }


# ------------------------------------------------------------------ parsing


def slug(label: str) -> str:
    """The key for a label: lower case, runs of anything but letters and digits become "_"."""
    return re.sub(r"[\W_]+", "_", label.lower(), flags=re.UNICODE).strip("_")


def _text(value: object, what: str, limit: int, *, required: bool = False) -> str:
    if value is None:
        value = ""
    if not isinstance(value, str):
        raise SubjectError(f"{what} must be text.")
    clean = " ".join(value.split())
    if required and not clean:
        raise SubjectError(f"{what} is required.")
    if len(clean) > limit:
        raise SubjectError(f"{what} must be at most {limit} characters.")
    return clean


def _flag(value: object, what: str, default: bool) -> bool:
    if value is None:
        return default
    if not isinstance(value, bool):
        raise SubjectError(f"{what} must be true or false.")
    return value


def _object(value: object, what: str, allowed: set[str]) -> dict:
    if not isinstance(value, dict):
        raise SubjectError(f"{what} must be an object.")
    unknown = set(value) - allowed
    if unknown:
        raise SubjectError(f"{what}: unknown setting {sorted(unknown)[0]!r}.")
    return value


def _key(label: str, what: str, taken: set[str]) -> str:
    key = slug(label)
    if not key:
        raise SubjectError(f"{what}: the name needs a letter or a digit.")
    if key in RESERVED_KEYS:
        raise SubjectError(f"{what}: {label!r} is a name the studio uses itself; choose another.")
    if key in taken:
        raise SubjectError(f"{what}: another level or field already has the name {label!r}.")
    taken.add(key)
    return key


def parse_subject(data: object) -> Subject:
    """Check a definition field by field; raise SubjectError for anything off."""
    d = _object(data, "The subject", {"preset", "noun", "trigger", "levels", "fields"})
    preset = d.get("preset", "custom")
    if preset not in PRESET_NAMES:
        raise SubjectError(f"preset must be one of {', '.join(PRESET_NAMES)}.")
    noun = _text(d.get("noun"), "The noun", NOUN_MAX)
    trigger = _text(d.get("trigger"), "The trigger word", TRIGGER_MAX)
    if not TRIGGER.match(trigger):
        raise SubjectError("The trigger word must be one word: letters, digits, \"_\" or \"-\".")

    raw_levels = d.get("levels")
    if not isinstance(raw_levels, list) or not 1 <= len(raw_levels) <= MAX_LEVELS:
        raise SubjectError(f"Give 1 to {MAX_LEVELS} folder levels.")
    taken: set[str] = set()
    levels = []
    for i, raw in enumerate(raw_levels):
        what = f"Folder level {i + 1}"
        r = _object(raw, what, {"key", "label", "required", "in_caption"})
        label = _text(r.get("label"), f"{what}: the name", LABEL_MAX, required=True)
        levels.append(Level(key=_key(label, what, taken), label=label, required=_flag(r.get("required"), what, False),
                            in_caption=_flag(r.get("in_caption"), what, True)))
    if not any(lv.required for lv in levels):
        raise SubjectError("At least one folder level must be required, or loose photos would be trained on.")

    raw_fields = d.get("fields", [])
    if not isinstance(raw_fields, list) or len(raw_fields) > MAX_FIELDS:
        raise SubjectError(f"Give at most {MAX_FIELDS} description fields.")
    fields = []
    for i, raw in enumerate(raw_fields):
        what = f"Description field {i + 1}"
        r = _object(raw, what, {"key", "label", "question", "choices", "important", "consistent", "phrase"})
        label = _text(r.get("label"), f"{what}: the name", LABEL_MAX, required=True)
        key = _key(label, f"{what} ({label})", taken)
        hint = _text(r.get("question"), f"{label}: the question", HINT_MAX, required=True)
        raw_choices = r.get("choices", [])
        if not isinstance(raw_choices, list) or len(raw_choices) > MAX_CHOICES:
            raise SubjectError(f"{label}: give at most {MAX_CHOICES} choices.")
        choices = []
        for c in raw_choices:
            choice = _text(c, f"{label}: a choice", CHOICE_MAX, required=True).lower()
            if choice == "unknown":
                raise SubjectError(f"{label}: \"unknown\" is always allowed; do not list it.")
            if choice in choices:
                raise SubjectError(f"{label}: the choice {choice!r} is listed twice.")
            choices.append(choice)
        if len(choices) == 1:
            raise SubjectError(f"{label}: give at least two choices, or none for free text.")
        phrase = _text(r.get("phrase"), f"{label}: the caption phrase", PHRASE_MAX) or VALUE
        if phrase.count(VALUE) != 1:
            raise SubjectError(f"{label}: the caption phrase must contain {VALUE} once, e.g. \"{VALUE} view\".")
        consistent = _flag(r.get("consistent"), label, False)
        if consistent and not choices:
            raise SubjectError(f"{label}: only a field with choices can be the same for the whole subject.")
        fields.append(FieldSpec(name=key, choices=tuple(choices) or None, important=_flag(r.get("important"), label, False),
                                hint=hint, label=label, consistent=consistent, phrase=phrase))
    return Subject(preset=preset, noun=noun, trigger=trigger, levels=tuple(levels), fields=tuple(fields))


# ------------------------------------------------------------------ presets


def _field(label: str, question: str, choices: tuple[str, ...] = (), *, important: bool = False,
           consistent: bool = False, phrase: str = VALUE) -> dict:
    return {"label": label, "question": question, "choices": list(choices), "important": important,
            "consistent": consistent, "phrase": phrase}


GENERAL = parse_subject({
    "preset": "general", "noun": "", "trigger": "",
    "levels": [{"label": "Name", "required": True}, {"label": "Variant"}],
    "fields": [
        _field("View", "which side of the {noun} faces the camera",
               ("front", "three-quarter", "side", "back", "top-down", "close-up detail"), important=True,
               phrase="{value} view"),
        _field("Colour", "its main colour in one or two words, e.g. \"dark blue\""),
        _field("Setting", "where the {noun} is, in a few words, e.g. \"wooden table\""),
        _field("Lighting", "the light", ("daylight", "overcast", "golden hour light", "at night", "studio lighting",
                                         "indoors")),
        _field("Framing", "how much of the {noun} is in the picture",
               ("whole in frame", "partly in frame", "close-up", "several in frame"), important=True),
    ],
})

CARS = parse_subject({
    "preset": "cars", "noun": "car", "trigger": "",
    "levels": [{"label": "Make", "required": True}, {"label": "Model", "required": True},
               {"label": "Generation"}, {"label": "Years"}],
    "fields": [
        _field("View", "which side of the car faces the camera",
               ("front", "front three-quarter", "side", "rear three-quarter", "rear", "top-down", "interior",
                "detail"), important=True, phrase="{value} view"),
        _field("Colour", "main body colour in one or two words, e.g. \"dark blue\""),
        _field("Body style", "the body style", ("sedan", "coupe", "convertible", "hatchback", "wagon", "suv",
                                               "pickup", "van", "roadster"), important=True, consistent=True),
        _field("Setting", "where the car is, in a few words, e.g. \"wet city street\""),
        _field("Lighting", "the light", ("daylight", "overcast", "golden hour light", "at night", "studio lighting",
                                         "indoors")),
        _field("Framing", "how much of the car is in the picture",
               ("whole car in frame", "partly in frame", "close-up detail", "several cars in frame"), important=True),
        _field("Occlusion", "how much of the car is hidden by something", ("not hidden", "partly hidden",
                                                                          "mostly hidden")),
    ],
})

PRESETS = {"general": GENERAL, "cars": CARS}


def from_project(meta: dict) -> Subject:
    """A project's subject. Projects made before subjects existed were all about cars."""
    if "subject" in meta:
        return parse_subject(meta["subject"])
    return CARS if meta.get("schema") == "cars" else GENERAL
