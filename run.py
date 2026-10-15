#!/usr/bin/env python3
"""Prepare fixed KiTS23 data and invoke the original single-call API evaluator."""

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "scripts"))
from prepare_data import load_inputs, prepare, sha256_file, write_json
from test_api import select_cases


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/api.example.json")
    parser.add_argument("--base-url", help="Overrides API_BASE_URL and config api.base_url")
    parser.add_argument("--model", help="Overrides API_MODEL and config api.model; match server model ID")
    parser.add_argument("--num-cases", type=int, help="1..200; default 200 in example config")
    parser.add_argument("--workers", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--output-dir", type=Path, help="Overrides config output_dir (relative to launch cwd)")
    parser.add_argument("--archive", type=Path, help="Existing original ZIP; avoids network download")
    parser.add_argument("--skip-download", action="store_true", help="Only use verified prepared assets")
    parser.add_argument("--dry-run", action="store_true", help="Prepare/verify data and save plan; zero model calls")
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    api, test = config["api"], config["test"]
    api["base_url"] = args.base_url or os.environ.get("API_BASE_URL") or api["base_url"]
    api["model"] = args.model or os.environ.get("API_MODEL") or api.get("model", "")
    if api.get("api_key"):
        raise ValueError("Use the api_key_env environment variable for credentials, not an API key in JSON")
    url = urlsplit(api["base_url"])
    if url.scheme not in ("http", "https") or not url.netloc or url.username or url.password or url.query or url.fragment:
        raise ValueError("api.base_url must be an HTTP(S) endpoint without credentials/query/fragment")
    if not isinstance(api["model"], str) or not api["model"].strip() or "YOUR_" in api["model"]:
        raise ValueError("No default model: set --model, API_MODEL, or config api.model to the server's visual model ID")
    for key, value in (("num_cases", args.num_cases), ("workers", args.workers), ("seed", args.seed)):
        if value is not None:
            test[key] = value
    if type(test["workers"]) is not int or not 1 <= test["workers"] <= 16:
        raise ValueError("workers must be an integer in 1..16")
    timeout = api.get("timeout_seconds", 300)
    if type(timeout) not in (int, float) or not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("timeout_seconds must be positive and finite")
    request = api.get("request", {})
    if not isinstance(request, dict) or {"model", "messages", "stream", "n"} & request.keys():
        raise ValueError("api.request must be an object without model/messages/stream/n overrides")
    # Validate selection before downloading 16 GB or contacting any model endpoint.
    _, source_cases, _ = load_inputs()
    select_cases(source_cases, test["num_cases"], test["seed"])
    jobs, data_metadata = prepare(args.data_dir, args.archive, args.skip_download)
    test["cases_file"] = str(jobs)
    output = (args.output_dir.resolve() if args.output_dir else
              (config_path.parent / config.get("output_dir", "../results")).resolve())
    output.mkdir(parents=True, exist_ok=True)
    config["output_dir"] = str(output)
    config["data_provenance"] = data_metadata
    config["client_provenance"] = {
        "wrapper_sha256": sha256_file(__file__),
        "evaluator_sha256": sha256_file(ROOT / "scripts/test_api.py"),
        "config_source_sha256": sha256_file(config_path),
    }
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    if args.dry_run:
        plan_dir = output / ("plan_" + stamp)
        plan_dir.mkdir(exist_ok=False)
        selected = select_cases(source_cases, test["num_cases"], test["seed"])
        write_json(plan_dir / "config.json", config)
        write_json(plan_dir / "plan.json", {"num_cases": len(selected), "api_calls": 0,
                   "max_calls_if_executed": len(selected), "selected_sample_ids": [r['sample_id'] for r in selected],
                   "data_provenance": data_metadata})
        print(f"Dry run complete: {plan_dir}; zero model API calls", flush=True)
        return 0
    configurations = output / "configurations"
    configurations.mkdir(exist_ok=True)
    actual_config = configurations / (stamp + ".json")
    write_json(actual_config, config)
    command = [sys.executable, str(ROOT / "scripts/test_api.py"), "--config", str(actual_config)]
    return subprocess.run(command, cwd=ROOT, check=False).returncode


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (ValueError, KeyError, OSError) as error:
        print(f"Evaluation setup error: {error}", file=sys.stderr)
        sys.exit(2)
