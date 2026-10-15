"""Saved-results viewer checks using tiny synthetic PNGs and loopback HTTP."""

import hashlib
from http.server import ThreadingHTTPServer
import io
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import ProxyHandler, build_opener

try:
    from PIL import Image
except ImportError:
    Image = None

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
if Image is not None:
    import results_viewer as viewer


@unittest.skipUnless(Image is not None, "Install optional dependency: bash setup.sh --viewer")
class ViewerTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.directory = Path(self.temporary.name)
        self.results = self.directory / "results"
        self.data = self.directory / "data"
        self.batch = self.results / "run_fixture"
        self.artifact = self.batch / "cases/00001"
        self.artifact.mkdir(parents=True)
        train, masks = self.data / "extracted/kits23/train", self.data / "extracted/kits23/train_mask"
        train.mkdir(parents=True)
        masks.mkdir()
        image_path = train / "case_fixture.png"
        Image.new("RGB", (3, 3), (100, 100, 100)).save(image_path)
        targets = []
        for name, label, positions in [
            ("tumor", "Kidney tumor", [(0, 0)]),
            ("cyst", "Kidney cyst", [(1, 0)]),
            ("kidney", "Kidney", [(0, 0), (0, 1)]),
        ]:
            mask = Image.new("L", (3, 3))
            for position in positions:
                mask.putpixel(position, 255)
            path = masks / (name + ".png")
            mask.save(path)
            targets.append({"label": label, "mask": "/missing-server/kits23/train_mask/" + path.name,
                            "mask_sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
        self.case = {"sample_id": "fixture", "case_id": "kits23:case_fixture",
                     "image": "/missing-server/kits23/train/" + image_path.name,
                     "image_sha256": hashlib.sha256(image_path.read_bytes()).hexdigest(),
                     "diagnoses": ["Kidney tumor", "Kidney cyst"], "targets": targets[:2],
                     "kidney": targets[2], "requires_negative_source_review": False}
        self.annotation = {"findings": "Synthetic observation", "assessment": "Synthetic explanation", "label": "both"}
        self.result = {"sample_id": "fixture", "status": "ok", "correct": True,
                       "expected_label": "both", "predicted_label": "both", "annotation": self.annotation}
        self.raw = json.dumps({"choices": [{"message": {"content": json.dumps(self.annotation),
                   "reasoning_content": "Synthetic complete reasoning"}, "finish_reason": "stop"}]}).encode()
        self.write_batch()
        self.store = viewer.Store(self.results, self.data)
        self.opener = build_opener(ProxyHandler({}))

    def tearDown(self):
        self.temporary.cleanup()

    def write_batch(self):
        (self.batch / "selected_cases.jsonl").write_text(json.dumps(self.case) + "\n")
        (self.batch / "results.jsonl").write_text(json.dumps(self.result) + "\n")
        (self.artifact / "result.json").write_text(json.dumps(self.result))
        (self.artifact / "response.txt").write_bytes(self.raw)
        (self.batch / "exit.json").write_text('{"exit_code":0}')

    def test_relocated_images_and_pixel_exact_overlays(self):
        raw = self.store.asset("run_fixture", 1, "image", "all", False)
        self.assertEqual(hashlib.sha256(raw).hexdigest(), self.case["image_sha256"])
        with Image.open(io.BytesIO(self.store.asset("run_fixture", 1, "overlay", "all", True))) as image:
            self.assertEqual(image.getpixel((0, 0)), (*viewer.COLORS["Kidney tumor"], 255))
            self.assertEqual(image.getpixel((1, 0)), (*viewer.COLORS["Kidney cyst"], 255))
            self.assertEqual(image.getpixel((0, 1)), (*viewer.COLORS["Kidney"], 255))
            self.assertEqual(image.getpixel((2, 2)), (0, 0, 0, 0))
        with Image.open(io.BytesIO(self.store.asset("run_fixture", 1, "overlay", "Kidney cyst", False))) as image:
            self.assertEqual(image.getpixel((0, 0))[3], 0)
            self.assertEqual(image.getpixel((1, 0))[3], 255)
        (self.data / "extracted/kits23/train/case_fixture.png").write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "校验值"):
            self.store.asset("run_fixture", 1, "image", "all", False)

    def test_catalog_pending_and_refresh(self):
        self.assertEqual(self.store.catalog()["runs"][0]["accuracy"], 1)
        self.assertEqual(self.store.detail("run_fixture", 1)["generated_json"], self.annotation)
        (self.artifact / "result.json").unlink()
        (self.batch / "results.jsonl").write_text('')
        self.assertEqual(self.store.samples("run_fixture")["samples"][0]["status"], "pending")
        self.assertEqual(self.store.catalog()["runs"][0]["completed"], 0)
        self.write_batch()
        self.assertEqual(self.store.catalog()["runs"][0]["completed"], 1)
        self.assertEqual(viewer.Store(self.directory / "empty", self.data).catalog()["runs"], [])

    def test_http_contract_raw_response_and_routes(self):
        class Quiet(viewer.Handler):
            def log_message(self, *args):
                pass
        server = ThreadingHTTPServer(("127.0.0.1", 0), Quiet)
        server.store = self.store
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        base = f"http://127.0.0.1:{server.server_port}"
        def get(path):
            with self.opener.open(base + path, timeout=3) as response:
                return response.read()
        try:
            self.assertEqual(json.loads(get('/health'))['runs'], 1)
            self.assertEqual(json.loads(get('/api/runs'))['runs'][0]['id'], 'run_fixture')
            query = urlencode({'run': 'run_fixture', 'index': 1})
            self.assertEqual(json.loads(get('/api/sample?' + query))['generated_json'], self.annotation)
            self.assertEqual(get('/api/raw?' + query), self.raw)
            self.assertTrue(get('/asset?' + query).startswith(b'\x89PNG'))
            html = get('/').decode()
            self.assertIn('GT mask 叠加图', html)
            self.assertIn('<details id="raw-details"', html)
            self.assertNotIn('<details id="raw-details" open', html)
            for path in ['/config.redacted.json', '/api/sample?' + urlencode({'run': '../', 'index': 1}),
                         '/api/raw?' + urlencode({'run': 'run_fixture', 'index': 99})]:
                with self.assertRaises(HTTPError):
                    get(path)
        finally:
            server.shutdown()
            server.server_close()
            thread.join()

    def test_missing_masks_and_invalid_run_do_not_expose_files(self):
        with self.assertRaisesRegex(ValueError, '--data-dir'):
            viewer.Store(self.results, self.directory / 'no_data').asset('run_fixture', 1, 'image', 'all', False)
        with self.assertRaises(KeyError):
            self.store.run_path('../outside')
        (self.results / 'run_link').symlink_to(self.batch, target_is_directory=True)
        with self.assertRaises(KeyError):
            self.store.run_path('run_link')

    def test_detached_launcher_and_existing_name_preserved(self):
        command = [sys.executable, str(ROOT / 'scripts/start_results_viewer.py'), '--name', 'fixture', '--port', '0',
                   '--root', str(self.results), '--data-dir', str(self.data),
                   '--output-root', str(self.directory / 'sessions')]
        pid = None
        try:
            result = subprocess.run(command, cwd='/tmp', text=True, capture_output=True, timeout=15)
            self.assertEqual(result.returncode, 0, result.stderr)
            launched = json.loads(result.stdout)
            pid = launched['pid']
            self.assertEqual(launched['state'], 'running')
            with self.opener.open(launched['url'] + '/health', timeout=3) as response:
                self.assertEqual(json.load(response)['runs'], 1)
            repeated = subprocess.run(command, cwd='/tmp', text=True, capture_output=True, timeout=10)
            self.assertEqual(repeated.returncode, 2)
            self.assertIn('existing logs', repeated.stderr)
            with self.opener.open(launched['url'] + '/health', timeout=3) as response:
                self.assertTrue(json.load(response)['ok'])
        finally:
            if pid is not None:
                os.kill(pid, signal.SIGTERM)


if __name__ == '__main__':
    unittest.main()
