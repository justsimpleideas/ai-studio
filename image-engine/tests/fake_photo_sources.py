"""A local stand-in for Wikimedia Commons, Openverse and the hosts their photos live on.

Used by the photo-collection tests and the end-to-end test, so collecting photos can
be tested without reaching the internet. It answers the few API shapes
scripts/collect_photos.py uses, from a fixed catalogue:

* Commons category "Category:Example Coupe": 6 files over two pages (continue token):
  CC BY, CC0, public domain, CC BY-SA (refused), personality rights (refused) and an
  SVG (refused). Its subcategory "Category:Example Coupe (Mk2)" holds one more CC0
  photo, which has a subcategory of its own with one more; "Category:Example Coupe
  logos" holds a badge that must be left out. Searching Commons returns the six.
* Openverse: three results — CC BY from another host, CC BY-SA (refused), and one
  that came from Commons (skipped, already covered).
* /img/<name>.jpg: a distinct generated JPEG per name; "big-*" are oversized,
  "tiny-*" are too small, "broken-*" are not pictures, "same-*" repeat one picture.

Run it alone with `python -m tests.fake_photo_sources <port>` (the end-to-end test does).
"""

from __future__ import annotations

import io
import json
import random
import sys
import threading
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from PIL import Image, ImageDraw


def picture(name: str, size=(900, 700)) -> bytes:
    rng = random.Random(name if not name.startswith("same-") else "same")
    img = Image.new("RGB", size, tuple(rng.randrange(256) for _ in range(3)))
    draw = ImageDraw.Draw(img)
    for _ in range(20):
        x, y = rng.randrange(size[0]), rng.randrange(size[1])
        draw.rectangle((x, y, x + rng.randrange(40, 300), y + rng.randrange(40, 200)),
                       fill=tuple(rng.randrange(256) for _ in range(3)))
    out = io.BytesIO()
    img.save(out, "JPEG", quality=85)
    return out.getvalue()


def _page(pid, name, license_code, short, *, mime="image/jpeg", restrictions="", width=900, height=700):
    return {"pageid": pid, "ns": 6, "title": f"File:{name}",
            "imageinfo": [{"width": width, "height": height, "mime": mime, "url": f"/img/{name}",
                           "thumburl": f"/img/{name}", "descriptionurl": f"/wiki/File:{name}",
                           "extmetadata": {"License": {"value": license_code},
                                           "LicenseShortName": {"value": short},
                                           "Artist": {"value": f'<a href="/user">Photographer {pid}</a>'},
                                           "Restrictions": {"value": restrictions}}}]}


COMMONS_PAGES = [
    [_page(101, "coupe-front.jpg", "cc-by-4.0", "CC BY 4.0"),
     _page(102, "coupe-side.jpg", "cc0", "CC0"),
     _page(103, "coupe-sa.jpg", "cc-by-sa-4.0", "CC BY-SA 4.0")],
    [_page(104, "coupe-rear.jpg", "pd", "Public domain"),
     _page(105, "coupe-people.jpg", "cc-by-4.0", "CC BY 4.0", restrictions="personality"),
     _page(106, "coupe-logo.svg", "cc0", "CC0", mime="image/svg+xml")],
]

SUBCATEGORIES = {
    "Category:Example Coupe": ["Category:Example Coupe (Mk2)", "Category:Example Coupe logos"],
    "Category:Example Coupe (Mk2)": ["Category:Example Coupe (Mk2) in red"],
}
CATEGORY_FILES = {
    "Category:Example Coupe (Mk2)": [_page(110, "coupe-mk2.jpg", "cc0", "CC0")],
    "Category:Example Coupe (Mk2) in red": [_page(112, "coupe-mk2-red.jpg", "cc0", "CC0")],
    "Category:Example Coupe logos": [_page(111, "coupe-badge.jpg", "cc0", "CC0")],
}

OPENVERSE_RESULTS = [
    {"id": "ov-1", "title": "Coupe at a show", "url": "/img/ov-coupe.jpg", "foreign_landing_url": "/photo/ov-1",
     "creator": "Someone", "license": "by", "license_version": "2.0", "source": "flickr", "filetype": "jpg",
     "width": 900, "height": 700},
    {"id": "ov-2", "title": "Share-alike coupe", "url": "/img/ov-sa.jpg", "foreign_landing_url": "/photo/ov-2",
     "creator": "Someone", "license": "by-sa", "license_version": "2.0", "source": "flickr", "filetype": "jpg",
     "width": 900, "height": 700},
    {"id": "ov-3", "title": "From Commons", "url": "/img/coupe-front.jpg", "foreign_landing_url": "/photo/ov-3",
     "creator": "Someone", "license": "by", "license_version": "4.0", "source": "wikimedia", "filetype": "jpg",
     "width": 900, "height": 700},
]


class FakeSources:
    """Serves the catalogue. `extra_commons` adds pages; `openverse_status` forces an error code."""

    def __init__(self) -> None:
        self.hits: list[str] = []
        #: The query of every request, parsed.
        self.queries: list[dict] = []
        #: What was searched for, at either service.
        self.searches: list[str] = []
        self.extra_commons: list[dict] = []
        self.openverse_status = 200
        #: File name -> the address the API gives for it instead of this server's.
        self.absolute_override: dict[str, str] = {}
        handler = self._handler()
        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), handler)
        self.port = self.httpd.server_address[1]
        self.base = f"http://127.0.0.1:{self.port}"

    def start(self) -> "FakeSources":
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()
        return self

    def stop(self) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()

    @property
    def commons_api(self) -> str:
        return f"{self.base}/commons/api.php"

    @property
    def openverse_api(self) -> str:
        return f"{self.base}/openverse/v1/images/"

    def _absolute(self, page: dict) -> dict:
        page = json.loads(json.dumps(page))
        info = page["imageinfo"][0]
        for key in ("url", "thumburl", "descriptionurl"):
            info[key] = self.base + info[key]
        name = page["title"].removeprefix("File:")
        if name in self.absolute_override:
            info["url"] = info["thumburl"] = self.absolute_override[name]
        return page

    def _handler(self):
        sources = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):  # quiet
                pass

            def _send(self, status, body, ctype):
                self.send_response(status)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):
                url = urllib.parse.urlsplit(self.path)
                q = dict(urllib.parse.parse_qsl(url.query))
                sources.hits.append(url.path)
                sources.queries.append(q)
                if q.get("gsrsearch") or q.get("q"):
                    sources.searches.append(q.get("gsrsearch") or q["q"])
                if url.path == "/commons/api.php" and q.get("prop") == "categoryinfo" and "titles" in q:
                    title = q["titles"]
                    page = {"title": title, "ns": 14}
                    if title == "Category:Example Coupe":
                        page["categoryinfo"] = {"files": sum(len(p) for p in COMMONS_PAGES), "subcats": 2}
                    else:
                        page["missing"] = True
                    return self._send(200, json.dumps({"query": {"pages": [page]}}).encode(), "application/json")
                if url.path == "/commons/api.php" and q.get("gcmtype") == "subcat":
                    pages = [{"title": t, "ns": 14, "categoryinfo": {"files": len(CATEGORY_FILES.get(t, [])),
                                                                     "subcats": len(SUBCATEGORIES.get(t, []))}}
                             for t in SUBCATEGORIES.get(q.get("gcmtitle", ""), [])]
                    return self._send(200, json.dumps({"query": {"pages": pages}}).encode(), "application/json")
                if url.path == "/commons/api.php" and q.get("list") == "search":
                    body = {"query": {"searchinfo": {"totalhits": sum(len(p) for p in COMMONS_PAGES)}, "search": []}}
                    return self._send(200, json.dumps(body).encode(), "application/json")
                if url.path == "/commons/api.php" and q.get("gcmtitle") in CATEGORY_FILES:
                    pages = [sources._absolute(p) for p in CATEGORY_FILES[q["gcmtitle"]]]
                    return self._send(200, json.dumps({"query": {"pages": pages}}).encode(), "application/json")
                if url.path == "/commons/api.php":
                    pages = COMMONS_PAGES + ([sources.extra_commons] if sources.extra_commons else [])
                    index = int(q.get("gcmcontinue") or q.get("gsroffset") or 0)
                    body = {"query": {"pages": [sources._absolute(p) for p in pages[index]]}}
                    if index + 1 < len(pages):
                        key = "gcmcontinue" if "gcmtitle" in q else "gsroffset"
                        body["continue"] = {key: str(index + 1), "continue": f"{key}||"}
                    return self._send(200, json.dumps(body).encode(), "application/json")
                if url.path == "/openverse/v1/images/":
                    if sources.openverse_status != 200:
                        return self._send(sources.openverse_status, b'{"detail": "throttled"}', "application/json")
                    results = []
                    for r in OPENVERSE_RESULTS:
                        r = dict(r)
                        r["url"] = sources.base + r["url"]
                        r["foreign_landing_url"] = sources.base + r["foreign_landing_url"]
                        results.append(r)
                    body = {"result_count": len(results), "page_count": 1, "page": 1, "results": results}
                    return self._send(200, json.dumps(body).encode(), "application/json")
                if url.path.startswith("/img/"):
                    name = url.path[len("/img/"):]
                    if name.startswith("big-"):
                        return self._send(200, picture(name) + b"\0" * (3 * 1024 * 1024), "image/jpeg")
                    if name.startswith("tiny-"):
                        return self._send(200, picture(name, (200, 150)), "image/jpeg")
                    if name.startswith("broken-"):
                        return self._send(200, b"<html>not a picture</html>", "image/jpeg")
                    if name.endswith(".svg"):
                        return self._send(200, b"<svg/>", "image/svg+xml")
                    return self._send(200, picture(name), "image/jpeg")
                self._send(404, b"not found", "text/plain")

        return Handler


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    fake = FakeSources()
    if port:
        fake.httpd.server_close()
        fake.httpd = ThreadingHTTPServer(("127.0.0.1", port), fake._handler())
        fake.port, fake.base = port, f"http://127.0.0.1:{port}"
    print(f"fake photo sources on {fake.base}", flush=True)
    fake.httpd.serve_forever()
