#!/usr/bin/env python3
"""Read-only local browser for saved API runs, images, GT masks and responses."""

import argparse
from datetime import datetime
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
import json
import os
from pathlib import Path
import re
import sys
from urllib.parse import parse_qs, urlsplit
try:
    from PIL import Image
except ImportError:
    print("Viewer dependency missing. Run: bash setup.sh --viewer", file=sys.stderr)
    sys.exit(2)

ROOT = Path(__file__).resolve().parents[1]
COLORS = {"Kidney tumor": (245, 116, 86), "Kidney cyst": (62, 168, 244), "Kidney": (84, 189, 142)}
LABEL_NAMES = {"tumor_only": "仅肿瘤", "cyst_only": "仅囊肿", "both": "肿瘤＋囊肿", "neither": "两类均无"}


def read_json(path, default=None):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return default


def rows(path):
    if not path.exists():
        return []
    values = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            try:
                values.append(json.loads(line))
            except ValueError:
                pass  # A running writer may not have finished its last line yet.
    return values


def checked(path, expected=None):
    raw = Path(path).read_bytes()
    if expected and hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError("图像或mask校验值与清单不一致")
    return raw


def content_json(content):
    if not isinstance(content, str):
        return None
    text = re.sub(r"^<think>.*?</think>\s*", "", content.strip(), count=1, flags=re.S)
    if text.startswith("```"):
        match = re.fullmatch(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.S)
        if match:
            text = match[1].strip()
    try:
        return json.loads(text)
    except ValueError:
        return None


class Store:
    def __init__(self, root, data_dir=None):
        self.root = Path(root).resolve()
        self.data_dir = Path(data_dir or ROOT / "data").resolve()

    def asset_path(self, recorded, folder):
        """Resolve saved paths locally, or relocate missing paths to prepared data."""
        path = Path(recorded)
        candidates = [path] if path.is_absolute() else [self.data_dir / path, self.root / path]
        candidates.append(self.data_dir / "extracted/kits23" / folder / path.name)
        for candidate in candidates:
            if candidate.is_file():
                return candidate
        raise ValueError("Image/mask unavailable; prepare data and set --data-dir: " + path.name)

    def run_path(self, run_id):
        if not re.fullmatch(r"[A-Za-z0-9_-]+", run_id):
            raise KeyError("Unknown run")
        path = self.root / run_id
        if path.is_symlink() or not (path / "selected_cases.jsonl").is_file():
            raise KeyError("Unknown run")
        return path

    def load(self, run_id):
        path = self.run_path(run_id)
        cases = rows(path / "selected_cases.jsonl")
        result_map = {row.get("sample_id"): row for row in rows(path / "results.jsonl")}
        records = []
        for index, case in enumerate(cases, 1):
            local = read_json(path / "cases" / f"{index:05d}" / "result.json")
            result = local if local and local.get("sample_id") == case["sample_id"] else result_map.get(case["sample_id"])
            records.append((case, result))
        return path, records

    def catalog(self):
        runs = []
        if self.root.exists():
            for path in sorted(self.root.iterdir(), reverse=True):
                try:
                    _, records = self.load(path.name)
                except (KeyError, OSError, ValueError):
                    continue
                done = sum(result is not None for _, result in records)
                correct = sum(bool(result and result.get("correct")) for _, result in records)
                runs.append({"id": path.name, "total": len(records), "completed": done, "correct": correct,
                             "accuracy": correct / len(records) if records else None,
                             "finished": (path / "exit.json").exists()})
        return {"runs": runs, "labels": LABEL_NAMES, "colors": COLORS}

    def record(self, run_id, index):
        path, records = self.load(run_id)
        if not 1 <= index <= len(records):
            raise KeyError("Unknown case")
        case, result = records[index - 1]
        return path, case, result

    @staticmethod
    def listing(index, case, result):
        diagnoses = set(case.get("diagnoses", []))
        expected = "both" if len(diagnoses) == 2 else ("tumor_only" if "Kidney tumor" in diagnoses
                    else "cyst_only" if "Kidney cyst" in diagnoses else "neither")
        return {"index": index, "sample_id": case["sample_id"], "case_id": case.get("case_id", ""),
                "expected_label": result.get("expected_label", expected) if result else expected,
                "predicted_label": result.get("predicted_label") if result else None,
                "status": result.get("status", "pending") if result else "pending",
                "correct": bool(result.get("correct")) if result else False}

    def samples(self, run_id):
        _, records = self.load(run_id)
        return {"run_id": run_id, "samples": [self.listing(i, case, result)
                for i, (case, result) in enumerate(records, 1)]}

    def detail(self, run_id, index):
        path, case, result = self.record(run_id, index)
        result = result or {}
        response_path = path / "cases" / f"{index:05d}" / "response.txt"
        envelope = read_json(response_path, {})
        choices = envelope.get("choices", []) if isinstance(envelope, dict) else []
        choice = choices[0] if choices and isinstance(choices[0], dict) else {}
        message = choice.get("message", {})
        content = message.get("content") if isinstance(message, dict) else None
        annotation = result.get("annotation")
        generated = content_json(content) if annotation is None else annotation
        return {**self.listing(index, case, result), "run_id": run_id,
                "image_name": Path(case["image"]).name, "generated_json": generated,
                "unparsed_content": content if generated is None else None,
                "raw_available": response_path.is_file(), "finish_reason": choice.get("finish_reason"),
                "error": result.get("error"), "elapsed_seconds": result.get("elapsed_seconds"),
                "usage": result.get("usage"), "targets": [target["label"] for target in case.get("targets", [])],
                "kidney_available": bool(case.get("kidney")),
                "negative_pending": bool(case.get("requires_negative_source_review"))}

    def asset(self, run_id, index, kind, target, organ):
        _, case, _ = self.record(run_id, index)
        raw = checked(self.asset_path(case["image"], "train"), case.get("image_sha256"))
        if kind == "image":
            return raw
        if kind != "overlay" or target not in ("all", "Kidney tumor", "Kidney cyst"):
            raise ValueError("Unknown image type or mask class")
        with Image.open(io.BytesIO(raw)) as original:
            width, height = original.size
        pixels = Image.new("RGBA", (width, height), (0, 0, 0, 0))
        assets = []
        if organ and case.get("kidney"):
            assets.append(case["kidney"])
        assets.extend(t for t in case.get("targets", []) if target == "all" or t["label"] == target)
        for asset in assets:
            data = checked(self.asset_path(asset["mask"], "train_mask"), asset.get("mask_sha256"))
            with Image.open(io.BytesIO(data)) as image:
                if image.size != (width, height):
                    raise ValueError("mask与原图尺寸不一致")
                mask = image.convert("L").point(lambda value: 255 if value else 0)
            pixels.paste((*COLORS[asset["label"]], 255), (0, 0, width, height), mask)
        buffer = io.BytesIO()
        pixels.save(buffer, format="PNG")
        return buffer.getvalue()


class Handler(BaseHTTPRequestHandler):
    def send(self, body, mime="application/json; charset=utf-8", status=200):
        if not isinstance(body, bytes):
            body = json.dumps(body, ensure_ascii=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", mime)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        try:
            url = urlsplit(self.path)
            query = parse_qs(url.query)
            store = self.server.store
            if url.path == "/":
                return self.send(Path(__file__).with_suffix(".html").read_bytes(), "text/html; charset=utf-8")
            if url.path == "/favicon.ico":
                return self.send(b"", "image/x-icon", 204)
            if url.path == "/health":
                return self.send({"ok": True, "service": "kits23-api-results-viewer", "runs": len(store.catalog()["runs"])})
            if url.path == "/api/runs":
                return self.send(store.catalog())
            run_id = query.get("run", [""])[0]
            if url.path == "/api/samples":
                return self.send(store.samples(run_id))
            index = int(query.get("index", ["0"])[0])
            if url.path == "/api/sample":
                return self.send(store.detail(run_id, index))
            if url.path == "/api/raw":
                path, _, _ = store.record(run_id, index)
                raw = path / "cases" / f"{index:05d}" / "response.txt"
                if not raw.is_file():
                    raise KeyError("No raw response")
                return self.send(raw.read_bytes(), "text/plain; charset=utf-8")
            if url.path == "/asset":
                return self.send(store.asset(run_id, index, query.get("kind", ["image"])[0],
                      query.get("target", ["all"])[0], query.get("organ", ["0"])[0] == "1"), "image/png")
            return self.send({"error": "Not found"}, status=404)
        except KeyError:
            return self.send({"error": "未找到该批次、样本或原始返回"}, status=404)
        except (BrokenPipeError, ConnectionResetError):
            pass
        except (ValueError, OSError) as error:
            return self.send({"error": str(error)}, status=400)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT / "results", help="Parent directory of saved run_* batches")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data", help="Prepared data directory, including relocated images/masks")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8507)
    parser.add_argument("--output", type=Path, help="New directory for server metadata; default results/viewer_sessions/server_<time>")
    args = parser.parse_args()
    if not 0 <= args.port <= 65535:
        parser.error("port must be in 0..65535")
    args.root = args.root.resolve()
    args.data_dir = args.data_dir.resolve()
    args.output = (args.output or ROOT / "results/viewer_sessions" /
                   datetime.now().astimezone().strftime("server_%Y%m%d_%H%M%S_%f")).resolve()
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    args.output.mkdir(parents=True, exist_ok=False)
    server.store = Store(args.root, args.data_dir)
    record = {"started_at": datetime.now().astimezone().isoformat(), "pid": os.getpid(),
              "host": args.host, "port": server.server_port, "root": str(args.root.resolve()),
              "data_dir": str(args.data_dir), "local_url": f"http://127.0.0.1:{server.server_port}",
              "command": [sys.executable, *sys.argv], "cwd": os.getcwd(),
              "code_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                              for p in (Path(__file__), Path(__file__).with_suffix(".html"))}}
    (args.output / "server.json").write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(record, ensure_ascii=False), flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
