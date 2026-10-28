import contextlib
import http.client
import json
import tempfile
import threading
import unittest
from pathlib import Path

from image_engine.backends import BackendError
from image_engine.backends.image import ImageBackend
from image_engine.config import Settings, SettingsStore
from image_engine.events import EventHub
from image_engine.gallery import Gallery
from image_engine.image_models import BASE_6B, LARGE, LARGE_FAST, TURBO, UnknownImageModel, profile_of, try_profile_of
from image_engine.jobs import JobQueue
from image_engine.memory import GIB, LARGE_PLANS, choose_large_plan
from image_engine.server import App, serve
from tests.fakes import FakeBackend

TOKEN = "t" * 40


def write_index(folder: Path, **index) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "model_index.json").write_text(json.dumps(index), encoding="utf-8")
    return folder


class ProfileTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_each_pipeline_has_its_profile(self):
        self.assertIs(profile_of(write_index(self.root / "a", _class_name="ZImagePipeline")), TURBO)
        self.assertIs(profile_of(write_index(self.root / "b", _class_name="Krea2Pipeline")), LARGE)
        self.assertIs(profile_of(write_index(self.root / "c", _class_name="Krea2Pipeline", is_distilled=True)), LARGE_FAST)
        # An imported 6B folder says which it is; the downloaded one does not and is the distilled model.
        self.assertIs(profile_of(write_index(self.root / "d", _class_name="ZImagePipeline", is_distilled=False)), BASE_6B)
        self.assertIs(profile_of(write_index(self.root / "e", _class_name="ZImagePipeline", is_distilled=True)), TURBO)
        self.assertTrue(BASE_6B.loras and not BASE_6B.training)

    def test_unknown_or_unreadable_folders(self):
        with self.assertRaises(UnknownImageModel):
            profile_of(write_index(self.root / "a", _class_name="SomethingElsePipeline"))
        with self.assertRaises(UnknownImageModel):
            profile_of(write_index(self.root / "b", _class_name="Krea2Pipeline", is_distilled="yes"))
        with self.assertRaises(UnknownImageModel):
            profile_of(self.root / "missing")
        self.assertIsNone(try_profile_of(""))
        self.assertIsNone(try_profile_of(self.root / "missing"))

    def test_what_each_model_can_do(self):
        self.assertTrue(TURBO.loras and TURBO.training)
        self.assertFalse(LARGE.loras or LARGE.training)
        self.assertEqual((LARGE.steps, LARGE.guidance), (28, 4.5))
        self.assertEqual((LARGE_FAST.steps, LARGE_FAST.guidance), (8, 0.0))
        self.assertEqual(len({p.model_id for p in (TURBO, LARGE, LARGE_FAST)}), 3)


class LargeMemoryPlanTests(unittest.TestCase):
    def test_auto_picks_by_vram(self):
        # 11.94 GiB is what a 12 GB laptop card reports.
        cases = [(48, "full"), (32, "full"), (24, "quality"), (16, "balanced"), (11.94, "balanced"),
                 (10, "low"), (8, "low")]
        for gib, expected in cases:
            with self.subTest(gib=gib):
                self.assertEqual(choose_large_plan("auto", int(gib * GIB)).name, expected)

    def test_explicit_mode_wins_over_vram(self):
        self.assertEqual(choose_large_plan("low", 48 * GIB).name, "low")

    def test_streaming_plans_never_keep_the_text_encoder_on_the_gpu(self):
        for plan in LARGE_PLANS.values():
            self.assertIn(plan.text_encoder, ("gpu", "swap", "cpu"))
            if plan.stream_transformer:
                self.assertNotEqual(plan.text_encoder, "gpu")


class FakeTurbo(FakeBackend):
    model_id = "turbo-6b"

    @contextlib.contextmanager
    def training_session(self):
        self.calls.append(("training_session",))
        yield "parts"


class FakeLarge(FakeBackend):
    model_id = "dit-12b"
    profile = LARGE


class RouterTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.turbo_dir = write_index(self.root / "turbo", _class_name="ZImagePipeline")
        self.large_dir = write_index(self.root / "large", _class_name="Krea2Pipeline")
        self.settings = Settings(model_dir=str(self.turbo_dir))
        self.made = {}

        def factory(cls):
            def make(get_settings):
                backend = cls()
                self.made[cls.__name__] = backend
                return backend
            return make

        self.router = ImageBackend(lambda: self.settings, {"diffusion": factory(FakeTurbo), "large": factory(FakeLarge)})

    def tearDown(self):
        self.tmp.cleanup()

    def test_the_folder_decides_until_a_model_loads(self):
        self.assertEqual((self.router.model_id, self.router.profile), ("turbo-6b", TURBO))
        self.settings.model_dir = str(self.large_dir)
        self.assertEqual((self.router.model_id, self.router.profile), ("dit-12b", LARGE))
        self.settings.model_dir = str(self.root / "not-downloaded-yet")
        self.assertEqual(self.router.profile, TURBO)  # what the download button would put there
        self.settings.model_dir = str(write_index(self.root / "odd", _class_name="OtherPipeline"))
        self.assertIsNone(self.router.profile)

    def test_loading_another_family_unloads_the_first(self):
        self.router.load(lambda _msg: None)
        turbo = self.made["FakeTurbo"]
        self.assertTrue(self.router.loaded and turbo.loaded)
        self.settings.model_dir = str(self.large_dir)
        self.assertEqual(self.router.model_id, "turbo-6b")  # still the loaded model's
        self.router.load(lambda _msg: None)
        self.assertFalse(turbo.loaded)
        self.assertTrue(self.made["FakeLarge"].loaded)
        self.assertEqual(self.router.model_id, "dit-12b")
        self.router.unload()
        self.assertFalse(self.router.loaded)

    def test_a_folder_holding_a_bare_model_file_points_to_the_import(self):
        bare = self.root / "downloads"
        bare.mkdir()
        (bare / "model.safetensors").write_bytes(b"x")
        self.settings.model_dir = str(bare)
        with self.assertRaises(BackendError) as ctx:
            self.router.load(lambda _msg: None)
        self.assertIn("Import a model file", str(ctx.exception))

    def test_an_unsupported_folder_fails_to_load_with_its_name(self):
        self.settings.model_dir = str(write_index(self.root / "odd", _class_name="OtherPipeline"))
        with self.assertRaises(BackendError) as ctx:
            self.router.load(lambda _msg: None)
        self.assertIn("OtherPipeline", str(ctx.exception))

    def test_stages_go_to_the_loaded_model(self):
        with self.assertRaises(BackendError):
            self.router.encode("a", "", 0.0)
        self.router.load(lambda _msg: None)
        self.assertEqual(self.router.encode("a", "", 0.0), {"prompt": "a"})
        self.assertIn(("encode", "a"), self.made["FakeTurbo"].calls)

    def test_loras_and_training_only_where_the_model_takes_them(self):
        self.router.set_loras([])  # nothing loaded, nothing asked: fine
        self.router.load(lambda _msg: None)
        self.router.set_loras([("abc", "path", 1.0)])
        with self.router.training_session() as parts:
            self.assertEqual(parts, "parts")
        self.settings.model_dir = str(self.large_dir)
        self.router.load(lambda _msg: None)
        self.router.set_loras([])
        with self.assertRaises(BackendError):
            self.router.set_loras([("abc", "path", 1.0)])
        with self.assertRaises(BackendError):
            with self.router.training_session():
                pass


class StubLoras:
    def get(self, lora_id):
        return {"id": lora_id, "name": "x", "sha256": "0" * 64}


class StatusTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        root = Path(cls.tmp.name)
        cls.large_dir = write_index(root / "large", _class_name="Krea2Pipeline")
        cls.settings = SettingsStore(root / "data")
        cls.settings.update({"model_dir": str(cls.large_dir)})
        cls.backend = ImageBackend(cls.settings.get, {"diffusion": lambda g: FakeTurbo(), "large": lambda g: FakeLarge()})
        cls.events = EventHub()
        cls.gallery = Gallery(root / "outputs")
        cls.jobs = JobQueue(cls.backend, cls.gallery, cls.events)
        cls.app = App(settings=cls.settings, backend=cls.backend, gallery=cls.gallery, events=cls.events,
                      jobs=cls.jobs, port=0, token=TOKEN, loras=StubLoras())
        cls.httpd = serve(cls.app, "127.0.0.1")
        cls.port = cls.httpd.server_address[1]
        threading.Thread(target=cls.httpd.serve_forever, daemon=True).start()

    @classmethod
    def tearDownClass(cls):
        cls.app.stopping.set()
        cls.httpd.shutdown()
        cls.httpd.server_close()
        cls.gallery.close()
        cls.tmp.cleanup()

    def request(self, method, path, body=None):
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=10)
        data = json.dumps(body).encode() if body is not None else None
        conn.request(method, path, body=data, headers={"Host": f"127.0.0.1:{self.port}", "X-Studio-Token": TOKEN,
                                                       "Content-Type": "application/json"})
        resp = conn.getresponse()
        out = json.loads(resp.read() or b"null")
        conn.close()
        return resp.status, out

    def test_status_reports_the_folders_model_and_its_defaults(self):
        status, body = self.request("GET", "/api/status")
        self.assertEqual(status, 200)
        self.assertEqual(body["image"]["profile"]["name"], "large")
        self.assertFalse(body["image"]["profile"]["loras"])
        self.assertEqual((body["defaults"]["steps"], body["defaults"]["guidance"]), (28, 4.5))
        self.assertEqual(body["model"]["id"], "dit-12b")

    def test_a_lora_request_for_a_model_without_loras_is_refused_up_front(self):
        status, body = self.request("POST", "/api/jobs", {"prompt": "a", "loras": [{"id": "0" * 16, "strength": 1}]})
        self.assertEqual(status, 400)
        self.assertIn("does not take LoRAs", json.dumps(body))


if __name__ == "__main__":
    unittest.main()
