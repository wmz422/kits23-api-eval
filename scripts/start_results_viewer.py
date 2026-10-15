#!/usr/bin/env python3
"""Start the saved-results viewer in a detached process with logs and a PID."""
import argparse
from datetime import datetime
import json
from pathlib import Path
import re
import socket
import subprocess
import sys
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", default=datetime.now().astimezone().strftime("viewer_%Y%m%d_%H%M%S_%f"))
    parser.add_argument("--port", type=int, default=8507)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--root", type=Path, default=ROOT / "results")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--output-root", type=Path, default=ROOT / "results/viewer_sessions")
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.name) or not (args.port == 0 or 1024 <= args.port <= 65535):
        parser.error("Choose a valid name and port (0 or 1024..65535)")
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    output = output_root / args.name
    log_path = output_root / (args.name + ".log")
    launch_path = output_root / (args.name + ".launch.json")
    if output.exists() or log_path.exists() or launch_path.exists():
        parser.error("Choose a new --name; existing logs and metadata are preserved")
    with socket.socket() as probe:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        probe.bind((args.host, args.port))
        port = probe.getsockname()[1]
    command = [sys.executable, "-u", str(ROOT / "scripts/results_viewer.py"),
               "--host", args.host, "--port", str(port), "--root", str(args.root.resolve()),
               "--data-dir", str(args.data_dir.resolve()), "--output", str(output)]
    with log_path.open("x") as log:
        process = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL,
                                   stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    record = {"started_at": datetime.now().astimezone().isoformat(), "pid": process.pid,
              "command": command, "cwd": str(ROOT), "log": str(log_path), "state": "starting"}
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    connect_host = "127.0.0.1" if args.host == "0.0.0.0" else args.host
    url = f"http://{connect_host}:{port}"
    for _ in range(50):
        if process.poll() is not None:
            record.update(state="failed", exit_code=process.returncode)
            break
        try:
            with opener.open(url + "/health", timeout=1) as response:
                health = json.loads(response.read())
            if health.get("service") == "kits23-api-results-viewer":
                record.update(state="running", health=health, url=url)
                break
        except OSError:
            time.sleep(.2)
    else:
        process.terminate()
        record.update(state="readiness_timeout", exit_code=process.wait(timeout=5))
    launch_path.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(record, ensure_ascii=False, indent=2), flush=True)
    return 0 if record["state"] == "running" else 1


if __name__ == "__main__":
    sys.exit(main())
