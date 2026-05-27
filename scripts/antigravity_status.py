#!/usr/bin/env python3
"""Antigravity provider: local LSP probing, port discovery, and quota parsing."""

import json
import os
import re
import ssl
import subprocess
import urllib.request
from typing import Any, Dict, List, Optional

# Local HTTPS uses a self-signed cert; allow insecure TLS for local communication.
# This is necessary because the Antigravity language server generates its own certs.
SSL_CONTEXT = ssl._create_unverified_context()

# Mapping Priority and Display Labels
MODEL_MAPPING = [
    {"pattern": "claude", "exclude": "thinking", "label": "Claude", "priority": 1},
    {"pattern": "pro", "include": "low", "label": "Gemini Pro", "priority": 2},
    {"pattern": "gemini", "include": "flash", "label": "Gemini Flash", "priority": 3},
]


def _find_processes() -> List[Dict[str, Any]]:
    """Detects all Antigravity/Jetski processes and returns candidates sorted by priority."""
    candidates = []

    def get_priority(cmd: str) -> int:
        # Prioritize specific CLI instances over the general Hub server
        if "/cli" in cmd or "cmd/cli" in cmd or "antigravity-cli" in cmd:
            return 1
        if "language_server" in cmd:
            return 2
        if "jetski-hub-server" in cmd or "antigravity-hub-server" in cmd:
            return 3
        return 4

    # Try native /proc on Linux for "Standard Library First" compliance
    if os.path.exists("/proc"):
        for pid_dir in os.listdir("/proc"):
            if not pid_dir.isdigit():
                continue
            try:
                with open(os.path.join("/proc", pid_dir, "cmdline"), "rb") as f:
                    cmdline = f.read().replace(b"\x00", b" ").decode("utf-8")

                # Match process name and Antigravity markers
                if (
                    "language_server" in cmdline
                    or "/cmd/cli/cli" in cmdline
                    or "jetski-hub-server" in cmdline
                    or "antigravity-hub-server" in cmdline
                    or "antigravity-cli" in cmdline
                    or "antigravity" in cmdline
                ) and (
                    "--app_data_dir antigravity" in cmdline
                    or "/antigravity/" in cmdline
                    or "--app_data_dir jetski" in cmdline
                    or "--app_data_dir=jetski" in cmdline
                    or "/jetski/" in cmdline
                ):
                    csrf_match = re.search(r"--csrf_token\s+([^\s]+)", cmdline)
                    csrf_token = csrf_match.group(1) if csrf_match else None

                    port_match = re.search(r"--extension_server_port\s+(\d+)", cmdline)
                    port = port_match.group(1) if port_match else None

                    candidates.append({
                        "pid": pid_dir,
                        "csrf_token": csrf_token,
                        "port": port,
                        "cmdline": cmdline
                    })
            except (IOError, OSError, UnicodeDecodeError):
                continue

    # Fallback to ps for macOS or if /proc is unavailable
    try:
        output = subprocess.check_output(
            ["ps", "-ax", "-o", "pid=,command="], stderr=subprocess.DEVNULL
        ).decode("utf-8")

        for line in output.splitlines():
            if (
                "language_server" in line
                or "/cmd/cli/cli" in line
                or "jetski-hub-server" in line
                or "antigravity-hub-server" in line
                or "antigravity-cli" in line
                or "antigravity" in line
            ) and (
                "--app_data_dir antigravity" in line
                or "/antigravity/" in line
                or "--app_data_dir jetski" in line
                or "--app_data_dir=jetski" in line
                or "/jetski/" in line
            ):
                pid_match = re.search(r"^\s*(\d+)", line)
                if not pid_match:
                    continue
                pid = pid_match.group(1)

                csrf_match = re.search(r"--csrf_token\s+([^\s]+)", line)
                csrf_token = csrf_match.group(1) if csrf_match else None

                port_match = re.search(r"--extension_server_port\s+(\d+)", line)
                port = port_match.group(1) if port_match else None

                # Avoid duplicates if we already scanned via /proc
                if not any(c["pid"] == pid for c in candidates):
                    candidates.append({
                        "pid": pid,
                        "csrf_token": csrf_token,
                        "port": port,
                        "cmdline": line
                    })
    except (subprocess.SubprocessError, FileNotFoundError):
        pass

    # Sort candidates so CLI instances are queried first
    candidates.sort(key=lambda x: get_priority(x["cmdline"]))
    return candidates


def _find_ports(pid: str) -> List[int]:
    """Finds all listening TCP ports for a given PID."""
    ports = []

    # Try native /proc on Linux for "Standard Library First" compliance
    # We look into /proc/<pid>/net/tcp and /proc/<pid>/fd
    if os.path.exists(f"/proc/{pid}/net/tcp"):
        try:
            with open(f"/proc/{pid}/net/tcp", "r") as f:
                lines = f.readlines()[1:]  # skip header
                for line in lines:
                    parts = line.split()
                    local_addr = parts[1]
                    state = parts[3]
                    # State 0A is LISTEN
                    if state == "0A":
                        port_hex = local_addr.split(":")[1]
                        ports.append(int(port_hex, 16))
            if ports:
                return sorted(list(set(ports)))
        except (IOError, OSError, IndexError, ValueError):
            pass

    # Fallback to lsof
    try:
        output = subprocess.check_output(
            ["lsof", "-nP", "-iTCP", "-sTCP:LISTEN", "-p", pid],
            stderr=subprocess.DEVNULL,
        ).decode("utf-8")
        for line in output.splitlines():
            match = re.search(r":(\d+)\s+\(LISTEN\)", line)
            if match:
                ports.append(int(match.group(1)))
    except (subprocess.SubprocessError, FileNotFoundError):
        pass
    return sorted(list(set(ports)))


def _probe_connect_port(
    ports: List[int], csrf_token: Optional[str], preferred_port: Optional[str] = None
) -> Optional[int]:
    """Probes ports to find the gRPC/Connect API port."""
    # csrf_token is now optional as some local server versions do not require it

    # Prioritize the preferred port if available
    if preferred_port:
        try:
            p = int(preferred_port)
            if p not in ports:
                ports = [p] + ports
        except ValueError:
            pass

    for port in ports:
        url = f"https://127.0.0.1:{port}/exa.language_server_pb.LanguageServerService/GetUnleashData"
        req = urllib.request.Request(
            url,
            data=json.dumps({}).encode("utf-8"),
            headers={
                "X-Codeium-Csrf-Token": csrf_token or "",
                "Connect-Protocol-Version": "1",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(
                req, context=SSL_CONTEXT, timeout=1
            ) as response:
                if response.status == 200:
                    return port
        except Exception:
            continue
    return None


def get_status() -> Dict[str, Any]:
    """Fetches and parses the Antigravity quota status trying all candidate processes."""
    candidates = _find_processes()
    if not candidates:
        return {"running": False}

    last_running_pid = None
    last_status_error = False

    for proc in candidates:
        pid = proc["pid"]
        last_running_pid = pid
        csrf_token = proc["csrf_token"]
        ext_port = proc["port"]

        ports = _find_ports(pid)
        connect_port = _probe_connect_port(ports, csrf_token, preferred_port=ext_port)

        if not connect_port:
            continue

        status_data = None

        # Primary: GetUserStatus
        try:
            url = f"https://127.0.0.1:{connect_port}/exa.language_server_pb.LanguageServerService/GetUserStatus"
            req = urllib.request.Request(
                url,
                data=json.dumps(
                    {
                        "ideName": "antigravity",
                        "extensionName": "antigravity",
                        "locale": "en",
                        "ideVersion": "unknown",
                    }
                ).encode("utf-8"),
                headers={
                    "X-Codeium-Csrf-Token": csrf_token or "",
                    "Connect-Protocol-Version": "1",
                    "Content-Type": "application/json",
                },
                method="POST",
            )
            with urllib.request.urlopen(req, context=SSL_CONTEXT, timeout=2) as response:
                if response.status == 200:
                    status_data = json.loads(response.read().decode("utf-8"))
        except Exception:
            # Fallback: GetCommandModelConfigs
            try:
                url = f"https://127.0.0.1:{connect_port}/exa.language_server_pb.LanguageServerService/GetCommandModelConfigs"
                req = urllib.request.Request(
                    url,
                    data=json.dumps({}).encode("utf-8"),
                    headers={
                        "X-Codeium-Csrf-Token": csrf_token or "",
                        "Connect-Protocol-Version": "1",
                        "Content-Type": "application/json",
                    },
                    method="POST",
                )
                with urllib.request.urlopen(
                    req, context=SSL_CONTEXT, timeout=2
                ) as response:
                    if response.status == 200:
                        status_data = json.loads(response.read().decode("utf-8"))
            except Exception:
                pass

        # Retry over HTTP on preferred port if HTTPS failed
        if not status_data and ext_port:
            try:
                url = f"http://127.0.0.1:{ext_port}/exa.language_server_pb.LanguageServerService/GetUserStatus"
                req = urllib.request.Request(
                    url,
                    data=json.dumps({}).encode("utf-8"),
                    headers={
                        "X-Codeium-Csrf-Token": csrf_token or "",
                        "Connect-Protocol-Version": "1",
                        "Content-Type": "application/json",
                    },
                    method="POST",
                )
                with urllib.request.urlopen(req, timeout=2) as response:
                    if response.status == 200:
                        status_data = json.loads(response.read().decode("utf-8"))
            except Exception:
                pass

        if not status_data:
            last_status_error = True
            continue

        # Parsing and model mapping
        models = []
        configs = (
            status_data.get("userStatus", {})
            .get("cascadeModelConfigData", {})
            .get("clientModelConfigs", [])
        )

        if not configs:
            configs = status_data.get("clientModelConfigs", [])

        for cfg in configs:
            model_name = cfg.get("modelName", "unknown")
            label = cfg.get("label", model_name)
            quota = cfg.get("quotaInfo", {})
            remaining = quota.get("remainingFraction", 0.0)
            reset_time = quota.get("resetTime")

            priority = 99
            display_label = label

            label_lower = label.lower()
            for mapping in MODEL_MAPPING:
                if mapping["pattern"] in label_lower:
                    if "include" in mapping and mapping["include"] not in label_lower:
                        continue
                    if "exclude" in mapping and mapping["exclude"] in label_lower:
                        continue
                    priority = mapping["priority"]
                    display_label = mapping["label"]
                    break

            models.append(
                {
                    "label": display_label,
                    "raw_label": label,
                    "remaining": remaining,
                    "reset_time": reset_time,
                    "priority": priority,
                }
            )

        # Sort by priority, then by remaining fraction (lowest first)
        models.sort(key=lambda x: (x["priority"], x["remaining"]))

        return {
            "running": True,
            "connected": True,
            "pid": pid,
            "models": models,
            "email": status_data.get("userStatus", {}).get("accountEmail"),
            "plan": status_data.get("userStatus", {}).get("planName"),
        }

    # If we found processes but none could be successfully queried
    return {
        "running": True,
        "connected": False,
        "pid": last_running_pid,
        "status_error": last_status_error
    }


if __name__ == "__main__":
    import sys

    status = get_status()
    if not status.get("running"):
        print("Antigravity is not running.")
        sys.exit(0)

    print(f"Antigravity running (PID: {status.get('pid')})")
    if not status.get("connected"):
        print("Could not connect to Antigravity API.")
        sys.exit(1)

    if status.get("status_error"):
        print("Error fetching status from API.")
        sys.exit(1)

    print(f"Account: {status.get('email', 'N/A')} ({status.get('plan', 'N/A')})")
    print("-" * 40)
    print(f"{'Model':<20} {'Remaining':>10} {'Reset Time'}")
    print("-" * 40)
    for m in status.get("models", []):
        rem_pct = f"{m['remaining'] * 100:.1f}%"
        print(f"{m['label']:<20} {rem_pct:>10} {m['reset_time']}")
