import json
import os
import time
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory
from datetime import datetime, timedelta


def create_mock_session(path, date_str, session_id):
    data = {
        "sessionId": session_id,
        "startTime": f"{date_str}T12:00:00Z",
        "messages": [
            {
                "type": "gemini",
                "model": "gemini-3-flash",
                "tokens": {"input": 100, "output": 50},
            }
        ],
    }
    with open(path, "w") as f:
        json.dump(data, f)


def run_benchmark():
    script_path = "scripts/token_usage.py"

    with TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        chats_dir = tmp_path / "bench-project" / "chats"
        chats_dir.mkdir(parents=True)

        print(f"Generating 5,000 historical files and 5 today's files in {tmpdir}...")

        # 5,000 old files (last month)
        old_date = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
        for i in range(5000):
            fpath = chats_dir / f"session-old-{i}.json"
            create_mock_session(fpath, old_date, f"old-{i}")
            # Set mtime to the past
            past_ts = (datetime.now() - timedelta(days=30)).timestamp()
            os.utime(fpath, (past_ts, past_ts))
        # 5 new files (today)
        today_date = datetime.now().strftime("%Y-%m-%d")
        for i in range(5):
            fpath = chats_dir / f"session-today-{i}.json"
            create_mock_session(fpath, today_date, f"today-{i}")

        print("\n--- Benchmark: Optimized Path (--today --raw) ---")
        start = time.perf_counter()
        subprocess.run(["python3", script_path, tmpdir, "--today", "--raw"], check=True)
        end = time.perf_counter()
        opt_time = end - start
        print(f"Execution Time: {opt_time:.4f} seconds")

        print("\n--- Benchmark: Full Path (no filters) ---")
        start = time.perf_counter()
        subprocess.run(["python3", script_path, tmpdir, "--raw"], check=True)
        end = time.perf_counter()
        full_time = end - start
        print(f"Execution Time: {full_time:.4f} seconds")

        improvement = (full_time - opt_time) / full_time * 100
        print(
            f"\nOptimization Result: {opt_time / full_time:.2%} of full execution time."
        )
        print(f"Performance Gain: {improvement:.1f}% faster")


if __name__ == "__main__":
    run_benchmark()
