#!/usr/bin/env python3
"""Test a vision Chat Completions endpoint and score the final KiTS23 label.

Usage: python3 test_api.py --config api_config.json
Standard library only. One request per case, no automatic retries.
"""

import argparse
import base64
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import random
import re
import sys
import time
from urllib.error import HTTPError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen


LABELS = ("tumor_only", "cyst_only", "both", "neither")
PROMPT = """Inspect this axial abdominal CT display for renal tumors and renal cysts.
Check all visible renal regions. Briefly describe visible findings and explain
how they support your final target decision. Do not infer HU, enhancement,
histology, patient laterality or unseen slices. Limit absence to this slice.
Return ONLY a JSON object with these fields:
{"findings": "brief visible observations", "assessment": "brief evidence-based explanation", "label": "tumor_only|cyst_only|both|neither"}
The label must be exactly one of the four strings: tumor_only (tumor without
cyst), cyst_only (cyst without tumor), both, neither (neither target on this slice).
Do not output segmentation tokens."""


def dump(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def digest(data):
    return hashlib.sha256(data).hexdigest()


def gt_label(case):
    diagnoses = case["diagnoses"]
    if (not isinstance(diagnoses, list) or not all(isinstance(x, str) for x in diagnoses)
            or len(set(diagnoses)) != len(diagnoses)
            or not set(diagnoses) <= {"Kidney tumor", "Kidney cyst"}):
        raise ValueError("Invalid GT diagnoses; expected a unique tumor/cyst list")
    if len(diagnoses) == 2:
        return "both"
    return {"Kidney tumor": "tumor_only", "Kidney cyst": "cyst_only"}.get(
        diagnoses[0], "neither") if diagnoses else "neither"


def select_cases(cases, count, seed):
    if type(count) is not int or not 1 <= count <= len(cases):
        raise ValueError(f"num_cases must be between 1 and {len(cases)}")
    if len({case["sample_id"] for case in cases}) != len(cases):
        raise ValueError("Duplicate sample_id in cases_file")
    rng = random.Random(seed)
    groups = {label: [] for label in LABELS}
    for case in cases:
        groups[gt_label(case)].append(case)
    for group in groups.values():
        rng.shuffle(group)
    order = list(LABELS)
    rng.shuffle(order)
    selected = []
    while len(selected) < count:
        for label in order:
            if groups[label] and len(selected) < count:
                selected.append(groups[label].pop())
    return selected


def final_label(response):
    choice = response["choices"][0]
    content = choice["message"].get("content")
    if choice.get("finish_reason") != "stop":
        raise ValueError("Completion did not finish with stop (possibly truncated/refused)")
    if not isinstance(content, str) or not content.strip():
        raise ValueError("Missing final message.content")
    text = content.strip()
    # Some compatible thinking servers put a closed think block before final JSON.
    text = re.sub(r"^<think>.*?</think>\s*", "", text, count=1, flags=re.S)
    if text.startswith("```"):
        match = re.fullmatch(r"```(?:json)?\s*\n?(.*?)\n?```", text, flags=re.S)
        if match:
            text = match[1].strip()
    value = json.loads(text)
    if not isinstance(value, dict) or value.get("label") not in LABELS:
        raise ValueError("Final JSON label must be tumor_only/cyst_only/both/neither")
    return value


def run_case(index, case, config, endpoint, api_key, run_dir):
    case_dir = run_dir / "cases" / f"{index:05d}"
    case_dir.mkdir(parents=True)
    started = time.monotonic()
    result = {"index": index, "sample_id": case["sample_id"], "image": case["image"],
              "expected_label": gt_label(case), "predicted_label": None,
              "negative_source_review_required": bool(case.get("requires_negative_source_review")),
              "status": "input_error", "correct": False, "http_status": None,
              "api_called": False, "artifact_dir": str(case_dir.relative_to(run_dir))}
    try:
        image_path = Path(case["image"])
        image_bytes = image_path.read_bytes()
        actual_hash = digest(image_bytes)
        if case.get("image_sha256") and actual_hash != case["image_sha256"]:
            raise ValueError("Image SHA-256 differs from manifest")
        if not image_bytes.startswith(b"\x89PNG\r\n\x1a\n"):
            raise ValueError("Expected original PNG image")
        result["image_sha256"] = actual_hash
        payload = {**config["api"].get("request", {}), "model": config["api"]["model"],
                   "stream": False, "n": 1, "messages": [{"role": "user", "content": [
                       {"type": "text", "text": config.get("prompt", PROMPT)},
                       {"type": "image_url", "image_url": {
                           "url": "data:image/png;base64," + base64.b64encode(image_bytes).decode("ascii")}}
                   ]}]}
        # The actual request has no GT, case identifiers, file paths or masks.
        dump(case_dir / "request.json", payload)
        result["request_sha256"] = digest(json.dumps(payload, sort_keys=True).encode())
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = "Bearer " + api_key
        request = Request(endpoint, data=json.dumps(payload).encode(), headers=headers, method="POST")
        result["status"] = "api_error"
        result["api_called"] = True
        try:
            with urlopen(request, timeout=config["api"].get("timeout_seconds", 300)) as response:
                raw = response.read()
                result["http_status"] = response.status
        except HTTPError as error:
            result["http_status"] = error.code
            (case_dir / "response.txt").write_bytes(error.read())
            raise ValueError(f"HTTP {error.code}") from None
        # Persist exact bytes before decoding or parsing, including malformed output.
        (case_dir / "response.txt").write_bytes(raw)
        result["status"] = "parse_error"
        envelope = json.loads(raw)
        if isinstance(envelope, dict):
            result["usage"] = envelope.get("usage")
        annotation = final_label(envelope)
        result.update(status="ok", annotation=annotation, predicted_label=annotation["label"],
                      correct=annotation["label"] == result["expected_label"])
    except Exception as error:
        message = str(error)
        if api_key:
            message = message.replace(api_key, "<redacted>")
        result.update(error_type=type(error).__name__, error=message)
    result["elapsed_seconds"] = round(time.monotonic() - started, 3)
    dump(case_dir / "result.json", result)
    return result


def score(results):
    total = len(results)
    correct = sum(row["correct"] for row in results)
    valid = sum(row["status"] == "ok" for row in results)
    per_label = {}
    confusion = {label: {pred: 0 for pred in (*LABELS, "invalid")} for label in LABELS}
    for row in results:
        confusion[row["expected_label"]][row["predicted_label"] or "invalid"] += 1
    for label in LABELS:
        group = [row for row in results if row["expected_label"] == label]
        matches = sum(row["correct"] for row in group)
        per_label[label] = {"total": len(group), "correct": matches,
                            "accuracy": matches / len(group) if group else None}
    observed = [group["accuracy"] for group in per_label.values() if group["total"]]
    return {"total": total, "correct": correct, "accuracy": correct / total if total else None,
            "score_percent": round(100 * correct / total, 2) if total else None,
            "valid_predictions": valid, "accuracy_on_valid": correct / valid if valid else None,
            "macro_accuracy_observed_labels": sum(observed) / len(observed) if observed else None,
            "status_counts": dict(Counter(row["status"] for row in results)),
            "per_label": per_label, "confusion_matrix": confusion,
            "scoring": "Exact four-state label match; API/input/parse errors count as incorrect.",
            "reference": "Existing exported annotations; neither candidates are not source-GT verified."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("api_config.json"))
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    api = config["api"]
    base_url = api["base_url"].rstrip("/")
    if (urlsplit(base_url).scheme not in ("http", "https") or not urlsplit(base_url).netloc
            or "YOUR_" in base_url or not api.get("model") or "YOUR_" in api["model"]):
        raise ValueError("Fill api.base_url and api.model in the config first")
    endpoint = base_url if base_url.endswith("/chat/completions") else base_url + "/chat/completions"
    api_key = api.get("api_key") or os.environ.get(api.get("api_key_env", "OPENAI_API_KEY"), "")
    settings = config["test"]
    workers = settings.get("workers", 1)
    if type(workers) is not int or not 1 <= workers <= 16:
        raise ValueError("workers must be an integer in 1..16")
    if not isinstance(api.get("request", {}), dict):
        raise ValueError("api.request must be an object of Chat Completions parameters")
    if any(key in api.get("request", {}) for key in ("model", "messages", "stream", "n")):
        raise ValueError("api.request cannot override model/messages/stream/n")
    timeout = api.get("timeout_seconds", 300)
    if not isinstance(timeout, (int, float)) or not 0 < timeout < float("inf"):
        raise ValueError("timeout_seconds must be a positive finite number")
    if not isinstance(config.get("prompt", PROMPT), str) or not config.get("prompt", PROMPT).strip():
        raise ValueError("prompt must be nonempty text")
    manifest = (config_path.parent / settings["cases_file"]).resolve()
    manifest_bytes = manifest.read_bytes()
    cases = [json.loads(line) for line in manifest_bytes.decode().splitlines() if line.strip()]
    selected = select_cases(cases, settings["num_cases"], settings.get("seed", 42))
    # Resolve relative image paths against the manifest, independent of launch cwd.
    selected = [dict(case, image=str((manifest.parent / case["image"]).resolve())) for case in selected]
    output_root = (config_path.parent / config.get("output_dir", "../results/api_test")).resolve()
    run_dir = output_root / datetime.now().astimezone().strftime("run_%Y%m%d_%H%M%S_%f")
    run_dir.mkdir(parents=True, exist_ok=False)
    safe_config = json.loads(json.dumps(config))
    if "api_key" in safe_config["api"]:
        safe_config["api"]["api_key"] = "<redacted>" if api_key else ""
    dump(run_dir / "config.redacted.json", safe_config)
    dump(run_dir / "metadata.json", {"started_at": datetime.now().astimezone().isoformat(),
         "command": [sys.executable, *sys.argv], "cwd": str(Path.cwd()), "python": sys.version,
         "cases_file": str(manifest), "cases_sha256": digest(manifest_bytes),
         "script_sha256": digest(Path(__file__).read_bytes()),
         "sampling": "seeded shuffled round-robin over four GT states", "seed": settings.get("seed", 42)})
    with (run_dir / "selected_cases.jsonl").open("w", encoding="utf-8") as stream:
        for case in selected:
            stream.write(json.dumps(case, ensure_ascii=False) + "\n")
    print(f"Results: {run_dir}", flush=True)
    results = []
    with (run_dir / "run.log").open("w", encoding="utf-8") as log, \
            (run_dir / "results.jsonl").open("w", encoding="utf-8") as stream, \
            ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(run_case, index, case, config, endpoint, api_key, run_dir)
                   for index, case in enumerate(selected, 1)]
        for future in futures:
            result = future.result()
            results.append(result)
            stream.write(json.dumps(result, ensure_ascii=False) + "\n")
            stream.flush()
            line = (f"[{len(results)}/{len(selected)}] {result['status']} "
                    f"GT={result['expected_label']} pred={result['predicted_label']} correct={result['correct']}")
            print(line, flush=True)
            log.write(line + "\n")
            log.flush()
    summary = score(results)
    summary["finished_at"] = datetime.now().astimezone().isoformat()
    dump(run_dir / "summary.json", summary)
    exit_code = 0 if summary["valid_predictions"] else 1
    dump(run_dir / "exit.json", {"exit_code": exit_code, "batch_completed": True,
                                "api_calls": sum(row["api_called"] for row in results)})
    print(f"Label accuracy: {summary['correct']}/{summary['total']} = {summary['score_percent']:.2f}%", flush=True)
    return exit_code


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (ValueError, KeyError, OSError) as error:
        print(f"Configuration/run error: {error}", file=sys.stderr)
        sys.exit(2)
