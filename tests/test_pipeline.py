"""Offline integration tests: synthetic PNGs, ZIPs and loopback HTTP only."""

import base64
from collections import Counter
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import importlib.util
import json
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch
import zipfile
import zlib

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import prepare_data as data
import test_api as api


def png(value):
    def chunk(kind, body):
        return struct.pack("!I", len(body)) + kind + body + struct.pack("!I", zlib.crc32(kind + body))
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack("!2I5B", 1, 1, 8, 0, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(bytes((0, value)))) + chunk(b"IEND", b""))


def save_json(path, value):
    path.write_text(json.dumps(value) + "\n")


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.directory = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def fixture(self):
        archive = self.directory / "kits23.zip"
        cases = []
        with zipfile.ZipFile(archive, "w") as zipped:
            for index, diagnoses in enumerate((["Kidney tumor"], ["Kidney cyst"],
                                                ["Kidney tumor", "Kidney cyst"], [])):
                raw = png(index)
                image = f"kits23/train/synthetic_{index}.png"
                zipped.writestr(image, raw)
                cases.append({"sample_id": f"synthetic-{index}", "case_id": f"case-{index}",
                              "split": "train", "group": api.LABELS[index], "image": image,
                              "image_sha256": hashlib.sha256(raw).hexdigest(),
                              "diagnoses": diagnoses, "targets": [], "kidney": None})
        manifest = self.directory / "manifest.jsonl"
        manifest.write_text("".join(json.dumps(row) + "\n" for row in cases))
        source = self.directory / "source.json"
        lock = {"repo_id": "synthetic/no-network", "repo_type": "dataset", "revision": "fixture",
                "filename": "kits23.zip", "size_bytes": archive.stat().st_size,
                "sha256": data.sha256_file(archive), "cases_manifest_sha256": data.sha256_file(manifest)}
        save_json(source, lock)
        return archive, source, manifest

    def test_fixed_manifest_and_original_script(self):
        source, cases, assets = data.load_inputs()
        self.assertEqual(len(cases), 200)
        self.assertEqual(len({row['case_id'] for row in cases}), 160)
        self.assertEqual(Counter(api.gt_label(row) for row in cases), dict.fromkeys(api.LABELS, 50))
        self.assertEqual(len(assets), 594)
        self.assertEqual(source['size_bytes'], 16178227112)
        self.assertEqual(len(api.select_cases(cases, 200, 42)), 200)
        self.assertEqual(api.select_cases(cases, 20, 42), api.select_cases(cases, 20, 42))
        self.assertEqual(Counter(api.gt_label(row) for row in api.select_cases(cases, 20, 42)),
                         dict.fromkeys(api.LABELS, 5))
        provenance = json.loads((ROOT / "PROVENANCE.json").read_text())
        self.assertEqual(data.sha256_file(ROOT / "scripts/test_api.py"), provenance['source_api_script_sha256'])
        self.assertNotIn("/opt/", (ROOT / "manifests/cases_200.jsonl").read_text())
        config = json.loads((ROOT / "configs/api.example.json").read_text())
        self.assertEqual(config['test']['num_cases'], 200)
        self.assertFalse(config['api'].get('api_key'))

    def test_prepare_resume_repair_and_checksum(self):
        archive, source, manifest = self.fixture()
        output = self.directory / "prepared"
        jobs, metadata = data.prepare(output, archive, source_path=source, manifest_path=manifest)
        self.assertEqual(metadata['assets_verified'], 4)
        self.assertTrue(metadata['archive_verified_this_invocation'])
        row = json.loads(jobs.read_text().splitlines()[0])
        image = jobs.parent / row['image']
        self.assertEqual(data.sha256_file(image), row['image_sha256'])
        with patch.object(data, 'download_archive', side_effect=AssertionError('must not download')):
            _, cached = data.prepare(output, verify_only=True, source_path=source, manifest_path=manifest)
        self.assertFalse(cached['archive_verified_this_invocation'])
        image.write_bytes(b"corrupt")
        with self.assertRaisesRegex(ValueError, 'missing/corrupt'):
            data.prepare(output, verify_only=True, source_path=source, manifest_path=manifest)
        data.prepare(output, archive, source_path=source, manifest_path=manifest)
        self.assertEqual(data.sha256_file(image), row['image_sha256'])
        invalid = self.directory / "invalid.zip"
        raw = bytearray(archive.read_bytes())
        raw[0] ^= 1
        invalid.write_bytes(raw)
        with self.assertRaisesRegex(ValueError, 'Archive SHA-256'):
            data.prepare(self.directory / "reject", invalid, source_path=source, manifest_path=manifest)

    def test_download_path_and_traversal_rejection(self):
        archive, source, manifest = self.fixture()
        with patch.object(data, 'download_archive', return_value=archive) as mocked:
            data.prepare(self.directory / 'download', source_path=source, manifest_path=manifest)
            mocked.assert_called_once()
            self.assertEqual(mocked.call_args.args[0]['revision'], 'fixture')
        rows = [json.loads(line) for line in manifest.read_text().splitlines()]
        rows[0]['image'] = 'kits23/train/../../escape.png'
        manifest.write_text(''.join(json.dumps(row) + '\n' for row in rows))
        lock = json.loads(source.read_text())
        lock['cases_manifest_sha256'] = data.sha256_file(manifest)
        save_json(source, lock)
        with self.assertRaisesRegex(ValueError, 'ZIP member'):
            data.load_inputs(source, manifest)
        target = self.directory / 'symlink'
        target.mkdir()
        (target / 'kits23').symlink_to('/tmp', target_is_directory=True)
        with self.assertRaisesRegex(ValueError, 'escapes'):
            data.destination(target, 'kits23/train/image.png')

    def test_bad_config_fails_before_download(self):
        command = [sys.executable, str(ROOT / 'run.py'), '--num-cases', '201',
                   '--data-dir', str(self.directory / 'absent'), '--output-dir', str(self.directory / 'out')]
        result = subprocess.run(command, cwd='/tmp', text=True, capture_output=True, timeout=15)
        self.assertEqual(result.returncode, 2)
        self.assertIn('num_cases must be between', result.stderr)
        self.assertFalse((self.directory / 'absent').exists())

    def test_http_scoring_and_exact_raw_responses(self):
        archive, source, manifest = self.fixture()
        jobs, _ = data.prepare(self.directory / 'http_data', archive,
                               source_path=source, manifest_path=manifest)
        rows = [json.loads(line) for line in jobs.read_text().splitlines()]
        fixture = {}
        for index, row in enumerate(rows):
            annotation = {'findings': 'Synthetic observation.', 'assessment': 'Synthetic explanation.',
                          'label': api.gt_label(row)}
            if index == 1:
                annotation['label'] = 'both'  # valid but wrong
            if index == 2:
                annotation['label'] = 'invalid'
            raw = json.dumps({'choices': [{'message': {'content': json.dumps(annotation),
                              'reasoning_content': 'Synthetic reasoning; never use for scoring.'},
                              'finish_reason': 'stop'}]}).encode()
            if index == 3:
                raw = b'{"error":{"message":"synthetic HTTP failure"}}'
            fixture[row['image_sha256']] = (503 if index == 3 else 200, raw)
        received = []
        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                payload = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                image = base64.b64decode(payload['messages'][0]['content'][1]['image_url']['url'].split(',')[1])
                checksum = hashlib.sha256(image).hexdigest()
                received.append((self.path, self.headers.get('Authorization'), payload, checksum))
                status, raw = fixture[checksum]
                self.send_response(status)
                self.end_headers()
                self.wfile.write(raw)
            def log_message(self, *args):
                pass
        server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        config = {'api': {'base_url': f'http://127.0.0.1:{server.server_port}/v1', 'model': 'synthetic',
                          'api_key_env': 'KITS23_TEST_KEY', 'timeout_seconds': 5},
                  'test': {'num_cases': 4, 'workers': 2, 'seed': 42, 'cases_file': str(jobs)},
                  'output_dir': str(self.directory / 'http_results')}
        config_path = self.directory / 'http.json'
        save_json(config_path, config)
        env = dict(os.environ, KITS23_TEST_KEY='synthetic-secret', NO_PROXY='127.0.0.1', no_proxy='127.0.0.1')
        try:
            result = subprocess.run([sys.executable, str(ROOT / 'scripts/test_api.py'), '--config', str(config_path)],
                                    cwd='/tmp', env=env, text=True, capture_output=True, timeout=20)
        finally:
            server.shutdown()
            server.server_close()
            thread.join()
        self.assertEqual(result.returncode, 0, result.stderr)
        run = next((self.directory / 'http_results').iterdir())
        summary = json.loads((run / 'summary.json').read_text())
        self.assertEqual((summary['total'], summary['correct'], summary['valid_predictions']), (4, 1, 2))
        self.assertEqual(summary['accuracy'], 0.25)
        self.assertEqual(summary['accuracy_on_valid'], 0.5)
        self.assertEqual(len(received), 4)
        for path, auth, payload, _ in received:
            self.assertEqual(path, '/v1/chat/completions')
            self.assertEqual(auth, 'Bearer synthetic-secret')
            self.assertEqual(payload['messages'][0]['content'][0]['text'], api.PROMPT)
            self.assertEqual(len(payload['messages'][0]['content']), 2)
        for row in (json.loads(line) for line in (run / 'results.jsonl').read_text().splitlines()):
            self.assertEqual((run / row['artifact_dir'] / 'response.txt').read_bytes(), fixture[row['image_sha256']][1])
        for path in run.rglob('*.json'):
            self.assertNotIn('synthetic-secret', path.read_text())

    def test_truncation_and_thinking_final_parser(self):
        def response(text, finish='stop'):
            return {'choices': [{'message': {'content': text, 'reasoning_content': '{"label":"both"}'},
                                 'finish_reason': finish}]}
        for text in ['```json\n{"label":"neither"}\n```', '<think>fixture</think>\n{"label":"neither"}']:
            self.assertEqual(api.final_label(response(text))['label'], 'neither')
        for text, finish in [('{"label":"both"}', 'length'), ('not json', 'stop'), ('', 'stop')]:
            with self.assertRaises(ValueError):
                api.final_label(response(text, finish))


if __name__ == '__main__':
    unittest.main()
