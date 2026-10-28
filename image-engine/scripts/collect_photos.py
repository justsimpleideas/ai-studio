"""Collect openly licensed photos for a training project (docs/training.md, "Collecting photos").

    python collect_photos.py --plan plan.json --dest <folder> [--json]
    python collect_photos.py --plan plan.json --check [--json]

--check downloads no photos: it asks each source how many files a target would
find and, from a sample of them, how many have a licence and size it would keep.

Searches Wikimedia Commons first (a category and its subcategories, or a search)
and Openverse second, and keeps only photos whose
licence allows reuse with at most attribution: CC0, public domain, and CC BY; the
plan may also allow share-alike (CC BY-SA). Non-commercial, no-derivatives and
unknown licences are always skipped, as are photos
Commons marks with personality rights (identifiable people). Every saved photo is
recorded with its author, licence and source page in ATTRIBUTION.json and CREDITS.md,
which is what CC BY asks for.

Photos land in the layout the Train tab reads: one folder level per level of the
project's subject (<dest>/<brand>/<model>/ for a subject of Brand / Model), with a
labels.json holding the exact names. Limits: photos per target, bytes per
photo, and bytes in total. Running again with the same plan and folder resumes: what
is already there is kept and counted.

Everything is fetched over HTTPS (plain http only to this machine, for the tests),
with no cookies and no referrer: a service learns the search words, which files
are fetched, this computer's internet address and the User-Agent below.

Standard library only, plus Pillow to check that each file really is a picture.
Not a part of the engine's server, which never reaches the network: the studio runs
this script on the user's request, the way it runs fetch_model.py.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import io
import itertools
import json
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterator

COMMONS_API = "https://commons.wikimedia.org/w/api.php"
OPENVERSE_API = "https://api.openverse.org/v1/images/"
# Wikimedia asks tools to name themselves and give a contact, or they may be blocked
# (foundation.wikimedia.org, "User-Agent policy"). The contact is the user's if they
# gave one, else the project's page: the same for everyone who runs the app.
PROJECT_URL = "https://github.com/justsimpleideas/ai-studio"
CONTACT = re.compile(r"[A-Za-z0-9@.:/_+~ -]{0,100}")


def user_agent(contact: str = "") -> str:
    return f"LocalAIStudio-collect/1.0 ({contact or PROJECT_URL}; dataset tool for personal training)"


USER_AGENT = user_agent()
MANIFEST = "collection.json"
ATTRIBUTION = "ATTRIBUTION.json"
CREDITS = "CREDITS.md"
IMAGE_TYPES = {"image/jpeg": ".jpg", "image/png": ".png", "image/webp": ".webp"}
ALLOWED_LICENSES = ("cc0", "pd", "cc-by")
SHARE_ALIKE = "cc-by-sa"
OPENVERSE_LICENSES = {"cc0": "cc0", "pd": "pdm", "cc-by": "by", SHARE_ALIKE: "by-sa"}
DEFAULTS = {"max_total_bytes": 2 * 1024**3, "max_file_bytes": 8 * 1024**2, "width": 1600, "min_side": 640,
            "sources": ["commons", "openverse"]}
LIMITS = {"targets": 10, "levels": 4, "count": 1000, "name": 60, "max_total_bytes": 50 * 1024**3, "max_file_bytes": 50 * 1024**2}
REQUEST_GAP = 0.25   # seconds between requests to the same API: polite, sequential use
# A Commons category for a subject mostly holds subcategories (by generation, year,
# colour...) and few files of its own, so collecting walks down this far.
CATEGORY_DEPTH = 3
CATEGORY_LIMIT = 150
# Subcategories that hold pictures about the subject, or copies of it, rather than
# photos of the real thing. Words, not topics: they apply to any subject.
SKIP_CATEGORY = re.compile(
    r"\b(logos?|emblems?|badges?|drawings?|illustrations?|paintings?|artworks?|in art|diagrams?|maps?|plans?|"
    r"videos?|audio|sounds?|advertisements?|brochures?|posters?|postcards?|stamps?|coins?|documents?|patents?|"
    r"screenshots?|icons?|coats of arms|toys?|model cars?|scale models?|models of|die-?cast|miniatures?|lego|"
    r"replicas?|in fiction|in popular culture|in film|in video games)\b", re.I)
# A walked category tree is kept this long, so a check and the collection after it walk once.
TREE_CACHE_SECONDS = 24 * 3600
# Openverse answers at most this many pages of results to an anonymous client.
OPENVERSE_PAGES = 10
OPENVERSE_PAGE_SIZE = 20
CHECK_SAMPLE = 50
TIMEOUT = 45
RETRIES = 1  # a timed-out or dropped request is tried once more, after a pause
WINDOWS_RESERVED = {"con", "prn", "aux", "nul", *(f"com{i}" for i in range(1, 10)), *(f"lpt{i}" for i in range(1, 10))}

_json_events = False


def say(message: str) -> None:
    print(message, file=sys.stderr if _json_events else sys.stdout, flush=True)


def emit(event: str, **fields: object) -> None:
    if _json_events:
        print(json.dumps({"event": event, **fields}), flush=True)


class PlanError(ValueError):
    pass


# ---------------------------------------------------------------------------- plan


@dataclass
class Level:
    #: The subject level's key, e.g. "brand"; labels.json uses it.
    key: str
    required: bool


@dataclass
class Target:
    #: One name per level of the subject, in order; "" for an optional level left out.
    values: tuple[str, ...]
    count: int = 200
    #: A Commons category such as "Category:<Brand> <Model>": the most precise source.
    category: str = ""
    #: Search words; empty means the names joined.
    query: str = ""

    @property
    def label(self) -> str:
        return " ".join(v for v in self.values if v)

    def search_text(self) -> str:
        return self.query or self.label


@dataclass
class Plan:
    levels: list[Level]
    targets: list[Target]
    max_total_bytes: int
    max_file_bytes: int
    width: int
    min_side: int
    sources: list[str]
    #: Licences kept (normalised); share-alike only when the plan asks for it.
    licences: frozenset[str] = frozenset(ALLOWED_LICENSES)
    #: The user's contact for the User-Agent; "" means the project's page.
    contact: str = ""


def _name(value: object, what: str, required: bool) -> str:
    if value is None:
        value = ""
    if not isinstance(value, str):
        raise PlanError(f"{what} must be text")
    value = " ".join(value.split())
    if required and not value:
        raise PlanError(f"{what} is required")
    if len(value) > LIMITS["name"]:
        raise PlanError(f"{what} is longer than {LIMITS['name']} characters")
    return value


def _int(data: dict, key: str, lo: int, hi: int) -> int:
    value = data.get(key, DEFAULTS.get(key))
    if isinstance(value, bool) or not isinstance(value, int) or not lo <= value <= hi:
        raise PlanError(f"{key} must be an integer from {lo} to {hi}")
    return value


def parse_levels(raw: object) -> list[Level]:
    if not isinstance(raw, list) or not 1 <= len(raw) <= LIMITS["levels"]:
        raise PlanError(f"give 1 to {LIMITS['levels']} levels")
    levels = []
    for lv in raw:
        if not isinstance(lv, dict) or not isinstance(lv.get("key"), str) or not isinstance(lv.get("required"), bool):
            raise PlanError("each level must be {\"key\": text, \"required\": true or false}")
        if not re.fullmatch(r"\w{1,30}", lv["key"]) or lv["key"].startswith("_"):
            raise PlanError(f"level key {lv['key'][:40]!r} is not a key")
        levels.append(Level(lv["key"], lv["required"]))
    if len({lv.key for lv in levels}) != len(levels):
        raise PlanError("two levels have the same key")
    if not any(lv.required for lv in levels):
        raise PlanError("at least one level must be required")
    return levels


def parse_plan(data: object) -> Plan:
    if not isinstance(data, dict):
        raise PlanError("the plan must be a JSON object")
    levels = parse_levels(data.get("levels"))
    raw = data.get("targets")
    if not isinstance(raw, list) or not 1 <= len(raw) <= LIMITS["targets"]:
        raise PlanError(f"give 1 to {LIMITS['targets']} targets")
    targets = []
    for t in raw:
        if not isinstance(t, dict):
            raise PlanError("each target must be an object")
        values = t.get("values")
        if not isinstance(values, list) or len(values) != len(levels):
            raise PlanError("each target needs one name per level")
        target = Target(values=tuple(_name(v, lv.key, lv.required) for v, lv in zip(values, levels)),
                        count=_int(t, "count", 1, LIMITS["count"]) if "count" in t else 200,
                        category=_name(t.get("category"), "category", False),
                        query=_name(t.get("query"), "query", False))
        if target.category and not target.category.startswith("Category:"):
            target.category = f"Category:{target.category}"
        targets.append(target)
    labels = [t.label.lower() for t in targets]
    if len(set(labels)) != len(labels):
        raise PlanError("two targets have the same names")
    sources = data.get("sources", DEFAULTS["sources"])
    if not isinstance(sources, list) or not sources or any(s not in ("commons", "openverse") for s in sources):
        raise PlanError("sources must list commons and/or openverse")
    return Plan(levels=levels, targets=targets,
                max_total_bytes=_int(data, "max_total_bytes", 10 * 1024**2, LIMITS["max_total_bytes"]),
                max_file_bytes=_int(data, "max_file_bytes", 100 * 1024, LIMITS["max_file_bytes"]),
                width=_int(data, "width", 512, 4096), min_side=_int(data, "min_side", 256, 4096),
                sources=list(dict.fromkeys(sources)),
                licences=frozenset(ALLOWED_LICENSES) | ({SHARE_ALIKE} if _flag(data, "share_alike") else set()),
                contact=_contact(data.get("contact")))


def _flag(data: dict, key: str) -> bool:
    value = data.get(key, False)
    if not isinstance(value, bool):
        raise PlanError(f"{key} must be true or false")
    return value


def _contact(value: object) -> str:
    value = "" if value is None else value
    if not isinstance(value, str) or not CONTACT.fullmatch(value.strip()):
        raise PlanError("contact may hold up to 100 letters, digits, spaces and @ . : / _ + ~ -")
    return value.strip()


# ---------------------------------------------------------------------------- names and licences


def folder_name(value: str) -> str:
    """A safe folder name for one level's name: letters, digits, spaces become underscores."""
    cleaned = re.sub(r"[^\w\-. ]+", "", value, flags=re.UNICODE).strip(" .")
    cleaned = re.sub(r"\s+", "_", cleaned)[:60] or "unnamed"
    if cleaned.lower() in WINDOWS_RESERVED or set(cleaned) <= {"."}:
        cleaned = f"_{cleaned}"
    return cleaned


def file_stem(title: str, ident: str) -> str:
    """A file name from a source title, ASCII only, made unique by the source's id."""
    base = re.sub(r"^File:", "", title)
    base = re.sub(r"\.[A-Za-z0-9]{2,5}$", "", base)
    slug = re.sub(r"[^a-z0-9]+", "-", base.lower()).strip("-")[:60] or "photo"
    return f"{slug}-{re.sub(r'[^A-Za-z0-9]', '', ident)[:24]}"


def normalise_license(code: str, short: str = "") -> str | None:
    """'cc0', 'pd', 'cc-by' or 'cc-by-sa'; None for anything else (non-commercial,
    no-derivatives, unknown). Whether share-alike is kept is the plan's choice.

    Accepts Commons codes ("cc-by-4.0", "cc0", "pd") and Openverse ones ("by", "cc0", "pdm")."""
    for raw in (code, short):
        text = (raw or "").strip().lower().replace("_", "-").replace(" ", "-")
        if not text:
            continue
        if text in ("cc0", "cc0-1.0", "cc-zero") or text.startswith("cc0"):
            return "cc0"
        if text in ("pd", "pdm", "public-domain", "publicdomain") or text.startswith("pd-"):
            return "pd"
        if re.fullmatch(r"(cc-)?by-sa(-\d(\.\d)?)?", text):
            return SHARE_ALIKE
        if re.fullmatch(r"(cc-)?by(-\d(\.\d)?)?", text):
            return "cc-by"
        if text.startswith(("cc-by-sa", "cc-by-nc", "cc-by-nd", "by-sa", "by-nc", "by-nd")):
            return None
    return None


def strip_html(value: str) -> str:
    text = re.sub(r"<[^>]+>", " ", value or "")
    return " ".join(html.unescape(text).split())[:200]


# ---------------------------------------------------------------------------- network


class Http:
    """Sequential GETs with a polite gap per host and a byte limit on every response."""

    def __init__(self, contact: str = "") -> None:
        self._last: dict[str, float] = {}
        self.user_agent = user_agent(contact)

    def _wait(self, url: str) -> None:
        host = urllib.parse.urlsplit(url).netloc
        gap = time.monotonic() - self._last.get(host, 0.0)
        if gap < REQUEST_GAP:
            time.sleep(REQUEST_GAP - gap)
        self._last[host] = time.monotonic()

    def get(self, url: str, limit: int) -> tuple[bytes, str]:
        parts = urllib.parse.urlsplit(url)
        local = parts.hostname in ("127.0.0.1", "localhost", "::1")
        if parts.scheme != "https" and not (parts.scheme == "http" and local):
            # Plain http would show the address and the picture to anyone on the way.
            raise NotHttps(f"refusing a non-HTTPS address: {url[:80]}")
        for attempt in range(RETRIES + 1):
            self._wait(url)
            try:
                return self._get(url, limit)
            except urllib.error.HTTPError:
                raise
            except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
                if attempt == RETRIES:
                    raise
                say(f"  retrying after: {exc}")
                time.sleep(2.0)
        raise AssertionError("unreachable")

    def _get(self, url: str, limit: int) -> tuple[bytes, str]:
        req = urllib.request.Request(url, headers={"User-Agent": self.user_agent})
        with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
            length = resp.headers.get("Content-Length")
            if length is not None and length.isdigit() and int(length) > limit:
                raise TooLarge(int(length))
            data = resp.read(limit + 1)
            if len(data) > limit:
                raise TooLarge(len(data))
            return data, resp.headers.get("Content-Type", "")

    def json(self, url: str) -> Any:
        data, _ = self.get(url, 20 * 1024**2)
        return json.loads(data)


class TooLarge(Exception):
    pass


class NotHttps(OSError):
    pass


@dataclass
class Candidate:
    source: str          # "commons" | "openverse"
    ident: str           # stable id at the source
    title: str
    url: str             # what is downloaded (a scaled copy where the source offers one)
    page: str            # the human page with the licence
    license: str         # normalised
    license_text: str    # as the source states it
    author: str
    width: int
    height: int
    mime: str


@dataclass
class CategoryTree:
    #: (title, files directly in it), the root first, then breadth first.
    categories: list[tuple[str, int]]
    #: Subcategories left out because their names say they hold logos, drawings, toys...
    skipped: list[str]
    #: CATEGORY_LIMIT was reached before the whole tree was seen.
    truncated: bool = False
    missing: bool = False
    #: Parent -> the subcategories first reached through it.
    children: dict[str, list[str]] = field(default_factory=dict)

    @property
    def files(self) -> int:
        return sum(n for _, n in self.categories)

    def totals(self) -> dict[str, int]:
        """Files in each category and everything below it."""
        own = dict(self.categories)
        out: dict[str, int] = {}
        for title, _ in reversed(self.categories):  # children come after their parent
            out[title] = own[title] + sum(out.get(c, 0) for c in self.children.get(title, []))
        return out

    def to_json(self) -> dict:
        return {"categories": self.categories, "skipped": self.skipped, "truncated": self.truncated,
                "missing": self.missing, "children": self.children}

    @classmethod
    def from_json(cls, data: dict) -> "CategoryTree":
        return cls(categories=[(str(t), int(n)) for t, n in data["categories"]], skipped=list(data["skipped"]),
                   truncated=bool(data["truncated"]), missing=bool(data["missing"]),
                   children={str(k): [str(c) for c in v] for k, v in data["children"].items()})


class TreeCache:
    """Walked category trees on disk for a day: a check and the collection after it walk once."""

    def __init__(self, folder: Path | None) -> None:
        self.folder = folder

    def _path(self, api: str, root: str) -> Path | None:
        if self.folder is None:
            return None
        return self.folder / f"tree-{hashlib.sha256(f'{api} {root}'.encode()).hexdigest()[:20]}.json"

    def get(self, api: str, root: str) -> CategoryTree | None:
        path = self._path(api, root)
        try:
            if path is None or time.time() - path.stat().st_mtime > TREE_CACHE_SECONDS:
                return None
            return CategoryTree.from_json(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, ValueError, KeyError, TypeError):
            return None

    def put(self, api: str, root: str, tree: CategoryTree) -> None:
        path = self._path(api, root)
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            _write(path, json.dumps(tree.to_json()))


def _query(api: str, **params: str) -> str:
    return f"{api}?{urllib.parse.urlencode({'action': 'query', 'format': 'json', 'formatversion': '2', **params})}"


def category_tree(http: Http, api: str, root: str, on_progress: Callable[[int], None] | None = None,
                  cache: TreeCache | None = None) -> CategoryTree:
    """A category and its subcategories, CATEGORY_DEPTH levels down, at most CATEGORY_LIMIT.
    `on_progress` hears how many categories have been found so far."""
    cached = cache.get(api, root) if cache else None
    if cached is not None:
        return cached
    tree = _walk(http, api, root, on_progress)
    if cache:
        cache.put(api, root, tree)
    return tree


def _walk(http: Http, api: str, root: str, on_progress: Callable[[int], None] | None) -> CategoryTree:
    pages = (http.json(_query(api, prop="categoryinfo", titles=root)).get("query") or {}).get("pages") or [{}]
    if pages[0].get("missing"):
        return CategoryTree([], [], missing=True)
    tree = CategoryTree([(root, int((pages[0].get("categoryinfo") or {}).get("files") or 0))], [])
    seen, frontier = {root}, [root]
    for _ in range(CATEGORY_DEPTH):
        below: list[str] = []
        for parent in frontier:
            cont: dict[str, str] = {}
            while True:
                data = http.json(_query(api, generator="categorymembers", gcmtitle=parent, gcmtype="subcat",
                                        gcmlimit="max", prop="categoryinfo", **cont))
                for page in (data.get("query") or {}).get("pages") or []:
                    title = page.get("title", "")
                    if not title or title in seen:
                        continue
                    seen.add(title)
                    if SKIP_CATEGORY.search(title.removeprefix("Category:")):
                        tree.skipped.append(title)
                        continue
                    if len(tree.categories) >= CATEGORY_LIMIT:
                        tree.truncated = True
                        return tree
                    tree.categories.append((title, int((page.get("categoryinfo") or {}).get("files") or 0)))
                    tree.children.setdefault(parent, []).append(title)
                    below.append(title)
                if "continue" not in data:
                    break
                cont = {k: str(v) for k, v in data["continue"].items()}
            if on_progress is not None:
                on_progress(len(tree.categories))
        frontier = below
        if not frontier:
            break
    return tree


def commons_candidates(http: Http, api: str, target: Target, width: int, tree: CategoryTree | None = None,
                       licences: frozenset[str] = frozenset(ALLOWED_LICENSES)) -> Iterator[Candidate | str]:
    """Photos from a Commons category and its subcategories (if given) or a File-namespace
    search. Yields a reason string for each file skipped before download, so the totals
    can say why."""
    if not target.category:
        yield from _commons_files(http, api, width, licences, generator="search", gsrnamespace="6", gsrlimit="50",
                                  gsrsearch=f"{target.search_text()} filetype:bitmap")
        return
    tree = tree or category_tree(http, api, target.category)
    if tree.categories:
        yield from _spread(http, api, width, licences, tree, tree.categories[0][0], tree.totals())


def _spread(http: Http, api: str, width: int, licences: frozenset[str], tree: CategoryTree, title: str,
            totals: dict[str, int]) -> Iterator[Candidate | str]:
    """A category's photos taken in turns from its own files and each subcategory's, all
    the way down, so a quota is spread over generations, years... instead of being filled
    by whichever subcategory comes first."""
    streams: list[Iterator[Candidate | str]] = []
    if dict(tree.categories).get(title):
        streams.append(_commons_files(http, api, width, licences, generator="categorymembers", gcmtitle=title,
                                      gcmtype="file", gcmlimit="50"))
    streams += [_spread(http, api, width, licences, tree, child, totals) for child in tree.children.get(title, [])
                if totals.get(child)]
    yield from interleave(streams)


def interleave(streams: list[Iterator[Candidate | str]]) -> Iterator[Candidate | str]:
    """Round robin: one candidate from each stream in turn; skip reasons pass straight through."""
    active = list(streams)
    while active:
        for stream in list(active):
            for item in stream:
                yield item
                if not isinstance(item, str):
                    break
            else:
                active.remove(stream)


def _commons_files(http: Http, api: str, width: int, licences: frozenset[str], **source: str) -> Iterator[Candidate | str]:
    base = {"action": "query", "format": "json", "formatversion": "2", "prop": "imageinfo",
            "iiprop": "url|size|mime|sha1|extmetadata", "iiurlwidth": str(width),
            "iiextmetadatafilter": "License|LicenseShortName|Artist|Restrictions", **source}
    cont: dict[str, str] = {}
    while True:
        data = http.json(f"{api}?{urllib.parse.urlencode({**base, **cont})}")
        for page in (data.get("query") or {}).get("pages") or []:
            info = (page.get("imageinfo") or [{}])[0]
            meta = info.get("extmetadata") or {}
            value = lambda k: str((meta.get(k) or {}).get("value") or "")  # noqa: E731
            if "personality" in value("Restrictions").lower():
                yield "personality rights"
                continue
            lic = normalise_license(value("License"), value("LicenseShortName"))
            if lic not in licences:
                yield "licence"
                continue
            mime = info.get("mime", "")
            if mime not in IMAGE_TYPES:
                yield "not a photo"
                continue
            yield Candidate(source="commons", ident=str(page.get("pageid", "")), title=page.get("title", ""),
                            url=info.get("thumburl") or info.get("url", ""), page=info.get("descriptionurl", ""),
                            license=lic, license_text=value("LicenseShortName") or value("License"),
                            author=strip_html(value("Artist")) or "unknown",
                            width=int(info.get("width") or 0), height=int(info.get("height") or 0), mime=mime)
        if "continue" not in data:
            return
        cont = {k: str(v) for k, v in data["continue"].items()}


def _openverse_query(target: Target, licences: frozenset[str], page: int) -> str:
    return urllib.parse.urlencode({"q": target.search_text(),
                                   "license": ",".join(OPENVERSE_LICENSES[lic] for lic in sorted(licences)),
                                   "page_size": str(OPENVERSE_PAGE_SIZE), "page": str(page), "mature": "false"})


def openverse_candidates(http: Http, api: str, target: Target,
                         licences: frozenset[str] = frozenset(ALLOWED_LICENSES)) -> Iterator[Candidate | str]:
    page = 1
    while page <= OPENVERSE_PAGES:
        try:
            data = http.json(f"{api}?{_openverse_query(target, licences, page)}")
        except urllib.error.HTTPError as exc:
            exc.close()
            if exc.code in (401, 429):
                yield "openverse limit reached"
                return
            raise
        yield from openverse_candidates_from(data, licences)
        if page >= int(data.get("page_count") or 1):
            return
        page += 1


def openverse_candidates_from(data: dict, licences: frozenset[str] = frozenset(ALLOWED_LICENSES)) -> Iterator[Candidate | str]:
    """The candidates on one page of Openverse results."""
    for r in data.get("results") or []:
        if (r.get("source") or "").lower() in ("wikimedia", "wikimedia_commons"):
            yield "already from Commons"
            continue
        lic = normalise_license(str(r.get("license") or ""))
        if lic not in licences:
            yield "licence"
            continue
        ftype = (r.get("filetype") or "").lower()
        mime = {"jpg": "image/jpeg", "jpeg": "image/jpeg", "png": "image/png", "webp": "image/webp"}.get(ftype, "")
        if not mime and ftype:
            yield "not a photo"
            continue
        yield Candidate(source="openverse", ident=str(r.get("id", "")), title=str(r.get("title") or "photo"),
                        url=str(r.get("url") or ""), page=str(r.get("foreign_landing_url") or ""), license=lic,
                        license_text=f"{r.get('license', '')} {r.get('license_version') or ''}".strip(),
                        author=strip_html(str(r.get("creator") or "")) or "unknown",
                        width=int(r.get("width") or 0), height=int(r.get("height") or 0),
                        mime=mime or "image/jpeg")


# ---------------------------------------------------------------------------- checking


def _kept(cand: Candidate | str, min_side: int) -> bool:
    return isinstance(cand, Candidate) and not 0 < min(cand.width, cand.height) < min_side


def _walking(index: int, target: Target) -> Callable[[int], None]:
    return lambda n: emit("walking", index=index, label=target.label, categories=n)


def commons_check(http: Http, api: str, target: Target, plan: Plan, index: int = 0,
                  cache: TreeCache | None = None) -> dict:
    out: dict = {"mode": "category" if target.category else "search"}
    if target.category:
        tree = category_tree(http, api, target.category, _walking(index, target), cache)
        if tree.missing:
            return {**out, "missing": True, "found": 0, "sampled": 0, "kept": 0, "estimate": 0}
        out.update(found=tree.files, categories=len(tree.categories), skipped=tree.skipped[:20],
                   truncated=tree.truncated)
    else:
        data = http.json(_query(api, list="search", srsearch=f"{target.search_text()} filetype:bitmap",
                                srnamespace="6", srinfo="totalhits", srlimit="1"))
        out["found"] = int(((data.get("query") or {}).get("searchinfo") or {}).get("totalhits") or 0)
        tree = None
    sample = list(itertools.islice(commons_candidates(http, api, target, plan.width, tree, plan.licences),
                                   CHECK_SAMPLE))
    kept = sum(1 for c in sample if _kept(c, plan.min_side))
    return {**out, "sampled": len(sample), "kept": kept,
            "estimate": round(out["found"] * kept / len(sample)) if sample else 0}


def openverse_check(http: Http, api: str, target: Target, plan: Plan, index: int = 0,
                    cache: TreeCache | None = None) -> dict:
    try:
        data = http.json(f"{api}?{_openverse_query(target, plan.licences, 1)}")
    except urllib.error.HTTPError as exc:
        exc.close()
        if exc.code in (401, 429):
            return {"error": "Openverse asked to slow down; try again later."}
        raise
    sample = list(openverse_candidates_from(data, plan.licences))
    kept = sum(1 for c in sample if _kept(c, plan.min_side))
    found = int(data.get("result_count") or 0)
    # The collector can only page through what Openverse shows an anonymous client.
    reachable = min(found, OPENVERSE_PAGES * OPENVERSE_PAGE_SIZE)
    return {"found": found, "reachable": reachable, "sampled": len(sample), "kept": kept,
            "estimate": round(reachable * kept / len(sample)) if sample else 0}


def check(plan: Plan, *, commons_api: str = COMMONS_API, openverse_api: str = OPENVERSE_API,
          http: Http | None = None, cache: TreeCache | None = None) -> list[dict]:
    """How many photos each target would likely get, without downloading any."""
    http = http or Http(plan.contact)
    results = []
    for index, target in enumerate(plan.targets):
        result: dict = {"index": index, "label": target.label, "want": target.count}
        for source, fn, api in (("commons", commons_check, commons_api), ("openverse", openverse_check, openverse_api)):
            if source not in plan.sources:
                continue
            try:
                result[source] = fn(http, api, target, plan, index, cache)
            except (urllib.error.URLError, TimeoutError, ConnectionError, OSError, ValueError) as exc:
                result[source] = {"error": f"{source} did not answer ({exc})"[:200]}
        emit("check", **result)
        say(f"{target.label}: " + "; ".join(
            f"{s} about {r['estimate']} of {r['found']} found" if "estimate" in r else f"{s}: {r.get('error')}"
            for s, r in ((s, result[s]) for s in ("commons", "openverse") if s in result)))
        results.append(result)
    emit("checked", targets=len(results))
    return results


# ---------------------------------------------------------------------------- collection


@dataclass
class Saved:
    target: str
    file: str            # relative to the destination, forward slashes
    sha256: str
    bytes: int
    source: str
    ident: str
    title: str
    page: str
    license: str
    license_text: str
    author: str


@dataclass
class Manifest:
    plan: dict
    items: list[Saved] = field(default_factory=list)
    skipped: dict[str, int] = field(default_factory=dict)

    @classmethod
    def load(cls, dest: Path, plan: dict) -> "Manifest":
        path = dest / MANIFEST
        if path.is_file():
            data = json.loads(path.read_text(encoding="utf-8"))
            items = [Saved(**i) for i in data.get("items", []) if (dest / i["file"]).is_file()]
            return cls(plan=plan, items=items, skipped={})
        return cls(plan=plan)

    def save(self, dest: Path) -> None:
        _write(dest / MANIFEST, json.dumps({"plan": self.plan, "items": [asdict(i) for i in self.items],
                                            "skipped": self.skipped}, indent=1))
        credits = [asdict(i) for i in self.items]
        _write(dest / ATTRIBUTION, json.dumps(credits, indent=1))
        lines = ["# Photo credits", "",
                 "Photos collected for training, with the licence each was published under.", "",
                 "| File | Title | Author | Licence | Source |", "|---|---|---|---|---|"]
        for i in self.items:
            cells = [i.file, i.title, i.author, i.license_text or i.license, i.page]
            lines.append("| " + " | ".join(c.replace("|", "\\|") for c in cells) + " |")
        _write(dest / CREDITS, "\n".join(lines) + "\n")


def _write(path: Path, text: str) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def check_picture(data: bytes, min_side: int) -> tuple[int, int]:
    """Size of a real picture of at least min_side on its shorter side, or ValueError."""
    from PIL import Image, UnidentifiedImageError

    try:
        with Image.open(io.BytesIO(data)) as img:
            img.verify()
        with Image.open(io.BytesIO(data)) as img:
            width, height = img.size
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
        raise ValueError("not a readable picture") from exc
    if min(width, height) < min_side:
        raise ValueError("too small")
    return width, height


def target_folder(dest: Path, target: Target) -> Path:
    """One folder per level, down to the first level left out; labels.json carries the rest."""
    parts = []
    for value in target.values:
        if not value:
            break
        parts.append(folder_name(value))
    folder = dest.joinpath(*parts)
    if not folder.resolve().is_relative_to(dest.resolve()):
        raise PlanError(f"unsafe folder for {target.label}")
    return folder


def write_labels(dest: Path, levels: list[Level], target: Target) -> None:
    labels = {lv.key: value for lv, value in zip(levels, target.values) if value}
    folder = target_folder(dest, target)
    folder.mkdir(parents=True, exist_ok=True)
    _write(folder / "labels.json", json.dumps(labels, indent=1))


def collect(plan: Plan, raw_plan: dict, dest: Path, *, commons_api: str = COMMONS_API,
            openverse_api: str = OPENVERSE_API, http: Http | None = None, cache: TreeCache | None = None) -> Manifest:
    http = http or Http(plan.contact)
    dest.mkdir(parents=True, exist_ok=True)
    manifest = Manifest.load(dest, raw_plan)
    seen = {(i.source, i.ident) for i in manifest.items}
    hashes = {i.sha256 for i in manifest.items}
    total = sum(i.bytes for i in manifest.items)
    emit("start", targets=[t.label for t in plan.targets], have=len(manifest.items), bytes=total,
         max_total_bytes=plan.max_total_bytes)

    def skip(reason: str) -> None:
        manifest.skipped[reason] = manifest.skipped.get(reason, 0) + 1

    stopped_by_cap = False
    for index, target in enumerate(plan.targets):
        write_labels(dest, plan.levels, target)
        folder = target_folder(dest, target)
        have = sum(1 for i in manifest.items if i.target == target.label)
        emit("target", index=index, label=target.label, have=have, want=target.count)
        say(f"{target.label}: {have} of {target.count} already here")
        sources = []
        if "commons" in plan.sources:
            tree = category_tree(http, commons_api, target.category, _walking(index, target), cache) \
                if target.category else None
            sources.append(commons_candidates(http, commons_api, target, plan.width, tree, plan.licences))
        if "openverse" in plan.sources:
            sources.append(openverse_candidates(http, openverse_api, target, plan.licences))
        for stream in sources:
            if have >= target.count or stopped_by_cap:
                break
            try:
                for cand in stream:
                    if have >= target.count:
                        break
                    if isinstance(cand, str):
                        skip(cand)
                        continue
                    if (cand.source, cand.ident) in seen:
                        continue
                    if 0 < min(cand.width, cand.height) < plan.min_side:
                        skip("too small")
                        continue
                    room = plan.max_total_bytes - total
                    if room <= 0:
                        stopped_by_cap = True
                        break
                    try:
                        data, ctype = http.get(cand.url, min(plan.max_file_bytes, room))
                        check_picture(data, plan.min_side)
                    except TooLarge:
                        if room > plan.max_file_bytes:
                            skip("file too large")
                            continue
                        stopped_by_cap = True  # what is left of the total would not fit a typical photo
                        break
                    except ValueError as exc:
                        skip(str(exc))
                        continue
                    except NotHttps:
                        skip("not https")
                        continue
                    except (urllib.error.URLError, TimeoutError, ConnectionError, OSError):
                        skip("download failed")
                        continue
                    digest = hashlib.sha256(data).hexdigest()
                    seen.add((cand.source, cand.ident))
                    if digest in hashes:
                        skip("duplicate")
                        continue
                    ext = IMAGE_TYPES.get(ctype.split(";")[0].strip().lower()) or IMAGE_TYPES.get(cand.mime, ".jpg")
                    path = folder / f"{file_stem(cand.title, cand.ident)}{ext}"
                    part = path.with_name(path.name + ".part")
                    part.write_bytes(data)
                    os.replace(part, path)
                    hashes.add(digest)
                    total += len(data)
                    have += 1
                    manifest.items.append(Saved(target=target.label, file=path.relative_to(dest).as_posix(),
                                                sha256=digest, bytes=len(data), source=cand.source,
                                                ident=cand.ident, title=re.sub(r"^File:", "", cand.title)[:200],
                                                page=cand.page, license=cand.license,
                                                license_text=cand.license_text, author=cand.author))
                    emit("saved", index=index, label=target.label, have=have, want=target.count, bytes=total,
                         source=cand.source)
                    if have % 10 == 0:
                        manifest.save(dest)  # a stop or crash loses at most a few photos' records
            except (urllib.error.URLError, TimeoutError, ConnectionError, OSError, ValueError) as exc:
                say(f"{target.label}: a source stopped answering ({exc}); trying the next one")
                skip("source unavailable")
        manifest.save(dest)
        emit("target_done", index=index, label=target.label, have=have, want=target.count)
        if stopped_by_cap:
            say(f"Stopped: the size limit of {plan.max_total_bytes / 1e9:.1f} GB is reached")
            break
    manifest.save(dest)
    emit("done", photos=len(manifest.items), bytes=total, skipped=manifest.skipped, size_limit_reached=stopped_by_cap)
    return manifest


def main(argv: list[str] | None = None) -> int:
    global _json_events
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--plan", type=Path, required=True, help="a JSON file with the targets and limits")
    ap.add_argument("--dest", type=Path, help="where the photos go (not needed with --check)")
    ap.add_argument("--check", action="store_true", help="only count what each source would give; download nothing")
    ap.add_argument("--cache", type=Path, default=None, help="where walked category trees are kept for a day")
    ap.add_argument("--json", action="store_true", help="progress as JSON lines on stdout (for the studio)")
    ap.add_argument("--commons-api", default=COMMONS_API, help=argparse.SUPPRESS)
    ap.add_argument("--openverse-api", default=OPENVERSE_API, help=argparse.SUPPRESS)
    args = ap.parse_args(argv)
    _json_events = args.json
    try:
        raw = json.loads(args.plan.read_text(encoding="utf-8"))
        plan = parse_plan(raw)
    except (OSError, ValueError) as exc:
        emit("error", message=f"Invalid plan: {exc}")
        print(f"Invalid plan: {exc}", file=sys.stderr)
        return 2
    if args.check:
        try:
            check(plan, commons_api=args.commons_api, openverse_api=args.openverse_api, cache=TreeCache(args.cache))
        except KeyboardInterrupt:
            return 130
        return 0
    if args.dest is None:
        ap.error("--dest is required unless --check is given")
    try:
        collect(plan, raw, args.dest.resolve(), commons_api=args.commons_api, openverse_api=args.openverse_api,
                cache=TreeCache(args.cache))
    except PlanError as exc:
        emit("error", message=str(exc))
        print(f"FAILED: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        return 130
    return 0


if __name__ == "__main__":
    sys.exit(main())
