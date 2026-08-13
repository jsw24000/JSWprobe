#!/usr/bin/env python3
"""Run a command behind a resource watchdog.

The watchdog is intentionally local and conservative: it only terminates the
wrapped command/process tree, never unrelated system services. It uses only
standard library modules plus optional nvidia-smi when available.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple


GIB = 1024**3
MIB = 1024**2
PAGE_SIZE = os.sysconf("SC_PAGE_SIZE")


@dataclass
class GpuStat:
    index: str
    uuid: str
    used_bytes: int
    total_bytes: int
    temp_c: Optional[float]
    util_pct: Optional[float]


def now_iso() -> str:
    return dt.datetime.now().astimezone().isoformat(timespec="seconds")


def parse_float(value: str) -> Optional[float]:
    value = value.strip()
    if not value or value in {"N/A", "[N/A]"}:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def parse_int(value: str) -> Optional[int]:
    value = value.strip()
    if not value or value in {"N/A", "[N/A]"}:
        return None
    try:
        return int(float(value))
    except ValueError:
        return None


def gib(value: int) -> float:
    return value / GIB


def read_meminfo() -> Dict[str, int]:
    values: Dict[str, int] = {}
    with open("/proc/meminfo", "r", encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            if len(parts) >= 2:
                values[parts[0].rstrip(":")] = int(parts[1]) * 1024
    return values


def read_pressure(path: str) -> Dict[str, float]:
    result: Dict[str, float] = {}
    try:
        text = Path(path).read_text(encoding="utf-8")
    except OSError:
        return result
    for line in text.splitlines():
        fields = line.split()
        if not fields:
            continue
        prefix = fields[0]
        for field in fields[1:]:
            if "=" not in field:
                continue
            key, value = field.split("=", 1)
            try:
                result[f"{prefix}_{key}"] = float(value)
            except ValueError:
                pass
    return result


def process_exists(pid: int) -> bool:
    return Path(f"/proc/{pid}").exists()


def proc_children(pid: int) -> List[int]:
    children: Set[int] = set()
    task_dir = Path(f"/proc/{pid}/task")
    try:
        tids = [p.name for p in task_dir.iterdir() if p.name.isdigit()]
    except OSError:
        return []
    for tid in tids:
        try:
            text = Path(f"/proc/{pid}/task/{tid}/children").read_text(encoding="utf-8")
        except OSError:
            continue
        for item in text.split():
            try:
                children.add(int(item))
            except ValueError:
                pass
    return sorted(children)


def process_tree(root_pid: int) -> Set[int]:
    seen: Set[int] = set()
    stack = [root_pid]
    while stack:
        pid = stack.pop()
        if pid in seen or not process_exists(pid):
            continue
        seen.add(pid)
        stack.extend(proc_children(pid))
    return seen


def rss_bytes(pid: int) -> int:
    try:
        statm = Path(f"/proc/{pid}/statm").read_text(encoding="utf-8").split()
    except OSError:
        return 0
    if len(statm) < 2:
        return 0
    try:
        return int(statm[1]) * PAGE_SIZE
    except ValueError:
        return 0


def cmdline(pid: int) -> str:
    try:
        raw = Path(f"/proc/{pid}/cmdline").read_bytes()
    except OSError:
        return ""
    text = raw.replace(b"\0", b" ").decode("utf-8", errors="replace").strip()
    return text


def proc_comm(pid: int) -> str:
    try:
        return Path(f"/proc/{pid}/comm").read_text(encoding="utf-8").strip()
    except OSError:
        return ""


def ancestor_pids(pid: int) -> Set[int]:
    ancestors: Set[int] = set()
    current = pid
    while current > 1:
        status_path = Path(f"/proc/{current}/status")
        try:
            lines = status_path.read_text(encoding="utf-8").splitlines()
        except OSError:
            break
        parent = 0
        for line in lines:
            if line.startswith("PPid:"):
                try:
                    parent = int(line.split()[1])
                except (ValueError, IndexError):
                    parent = 0
                break
        if parent <= 1 or parent in ancestors:
            break
        ancestors.add(parent)
        current = parent
    return ancestors


def process_pattern_present(pattern: str, exclude_pids: Set[int]) -> bool:
    proc_dir = Path("/proc")
    try:
        entries = list(proc_dir.iterdir())
    except OSError:
        return False
    for entry in entries:
        if not entry.name.isdigit():
            continue
        pid = int(entry.name)
        if pid in exclude_pids:
            continue
        haystack = f"{proc_comm(pid)} {cmdline(pid)}"
        if pattern in haystack:
            return True
    return False


def read_loadavg() -> Tuple[float, float, float]:
    try:
        first = Path("/proc/loadavg").read_text(encoding="utf-8").split()[:3]
        return float(first[0]), float(first[1]), float(first[2])
    except (OSError, ValueError, IndexError):
        return 0.0, 0.0, 0.0


def run_nvidia_smi(args: Sequence[str]) -> str:
    try:
        completed = subprocess.run(
            ["nvidia-smi", *args],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=5,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return ""
    if completed.returncode != 0:
        return ""
    return completed.stdout.strip()


def read_gpu_stats() -> List[GpuStat]:
    output = run_nvidia_smi(
        [
            "--query-gpu=index,uuid,memory.used,memory.total,temperature.gpu,utilization.gpu",
            "--format=csv,noheader,nounits",
        ]
    )
    if not output:
        return []

    stats: List[GpuStat] = []
    for row in csv.reader(output.splitlines()):
        if len(row) < 6:
            continue
        used_mib = parse_int(row[2]) or 0
        total_mib = parse_int(row[3]) or 0
        stats.append(
            GpuStat(
                index=row[0].strip(),
                uuid=row[1].strip(),
                used_bytes=used_mib * MIB,
                total_bytes=total_mib * MIB,
                temp_c=parse_float(row[4]),
                util_pct=parse_float(row[5]),
            )
        )
    return stats


def read_compute_gpu_memory(target_pids: Set[int]) -> Dict[str, int]:
    if not target_pids:
        return {}
    output = run_nvidia_smi(
        [
            "--query-compute-apps=pid,gpu_uuid,used_memory",
            "--format=csv,noheader,nounits",
        ]
    )
    if not output:
        return {}

    by_uuid: Dict[str, int] = {}
    for row in csv.reader(output.splitlines()):
        if len(row) < 3:
            continue
        pid = parse_int(row[0])
        if pid not in target_pids:
            continue
        uuid = row[1].strip()
        used_mib = parse_int(row[2]) or 0
        by_uuid[uuid] = by_uuid.get(uuid, 0) + used_mib * MIB
    return by_uuid


def signal_process_tree(root_pid: int, sig: signal.Signals) -> None:
    pids = list(process_tree(root_pid))
    pids.sort(reverse=True)
    for pid in pids:
        try:
            os.kill(pid, sig)
        except ProcessLookupError:
            pass
        except PermissionError:
            print(f"[safe-run] no permission to signal pid {pid}", file=sys.stderr)


def terminate_target(
    *,
    pid: int,
    reason: str,
    use_process_group: bool,
    grace_seconds: float,
) -> None:
    print(f"[safe-run] threshold crossed: {reason}", file=sys.stderr)
    if use_process_group:
        try:
            pgid = os.getpgid(pid)
            print(f"[safe-run] sending SIGTERM to process group {pgid}", file=sys.stderr)
            os.killpg(pgid, signal.SIGTERM)
        except ProcessLookupError:
            return
        except PermissionError:
            print("[safe-run] no permission for process group SIGTERM; falling back to tree", file=sys.stderr)
            signal_process_tree(pid, signal.SIGTERM)
    else:
        print(f"[safe-run] sending SIGTERM to process tree rooted at {pid}", file=sys.stderr)
        signal_process_tree(pid, signal.SIGTERM)

    deadline = time.monotonic() + grace_seconds
    while time.monotonic() < deadline:
        if not process_exists(pid):
            return
        time.sleep(0.2)

    if use_process_group:
        try:
            pgid = os.getpgid(pid)
            print(f"[safe-run] sending SIGKILL to process group {pgid}", file=sys.stderr)
            os.killpg(pgid, signal.SIGKILL)
            return
        except ProcessLookupError:
            return
        except PermissionError:
            print("[safe-run] no permission for process group SIGKILL; falling back to tree", file=sys.stderr)

    print(f"[safe-run] sending SIGKILL to process tree rooted at {pid}", file=sys.stderr)
    signal_process_tree(pid, signal.SIGKILL)


def make_log_path(log_dir: Path, label: str) -> Path:
    log_dir.mkdir(parents=True, exist_ok=True)
    safe_label = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in label)[:80]
    timestamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    return log_dir / f"{timestamp}-{safe_label}.jsonl"


def collect_metrics(root_pid: int, required_process_names: Sequence[str]) -> Dict[str, object]:
    pids = process_tree(root_pid)
    rss_total = sum(rss_bytes(pid) for pid in pids)
    meminfo = read_meminfo()
    gpu_stats = read_gpu_stats()
    target_gpu_by_uuid = read_compute_gpu_memory(pids)
    load1, load5, load15 = read_loadavg()
    memory_pressure = read_pressure("/proc/pressure/memory")
    io_pressure = read_pressure("/proc/pressure/io")
    exclude_pids = set(pids)
    exclude_pids.add(os.getpid())
    exclude_pids.update(ancestor_pids(os.getpid()))
    required_processes = {
        pattern: process_pattern_present(pattern, exclude_pids)
        for pattern in required_process_names
    }

    gpu_payload = []
    uuid_to_index = {g.uuid: g.index for g in gpu_stats}
    for gpu in gpu_stats:
        target_used = target_gpu_by_uuid.get(gpu.uuid, 0)
        gpu_payload.append(
            {
                "index": gpu.index,
                "uuid": gpu.uuid,
                "total_used_gib": round(gib(gpu.used_bytes), 3),
                "total_memory_gib": round(gib(gpu.total_bytes), 3),
                "target_used_gib": round(gib(target_used), 3),
                "temperature_c": gpu.temp_c,
                "utilization_pct": gpu.util_pct,
            }
        )
    for uuid, used in target_gpu_by_uuid.items():
        if uuid not in uuid_to_index:
            gpu_payload.append(
                {
                    "index": "unknown",
                    "uuid": uuid,
                    "total_used_gib": None,
                    "total_memory_gib": None,
                    "target_used_gib": round(gib(used), 3),
                    "temperature_c": None,
                    "utilization_pct": None,
                }
            )

    swap_total = meminfo.get("SwapTotal", 0)
    swap_free = meminfo.get("SwapFree", 0)
    return {
        "time": now_iso(),
        "pid": root_pid,
        "process_count": len(pids),
        "target_rss_gib": round(gib(rss_total), 3),
        "mem_available_gib": round(gib(meminfo.get("MemAvailable", 0)), 3),
        "mem_total_gib": round(gib(meminfo.get("MemTotal", 0)), 3),
        "swap_used_gib": round(gib(max(0, swap_total - swap_free)), 3),
        "swap_total_gib": round(gib(swap_total), 3),
        "load1": load1,
        "load5": load5,
        "load15": load15,
        "memory_pressure_some_avg10": memory_pressure.get("some_avg10"),
        "memory_pressure_full_avg10": memory_pressure.get("full_avg10"),
        "io_pressure_some_avg10": io_pressure.get("some_avg10"),
        "io_pressure_full_avg10": io_pressure.get("full_avg10"),
        "cpu_pressure_some_avg10": read_pressure("/proc/pressure/cpu").get("some_avg10"),
        "required_processes": required_processes,
        "gpus": gpu_payload,
    }


def threshold_reason(args: argparse.Namespace, metrics: Dict[str, object]) -> Optional[str]:
    mem_available_gib = float(metrics["mem_available_gib"])
    target_rss_gib = float(metrics["target_rss_gib"])
    swap_used_gib = float(metrics["swap_used_gib"])
    load1 = float(metrics["load1"])
    required_processes = metrics.get("required_processes", {})

    if isinstance(required_processes, dict):
        for pattern, present in required_processes.items():
            if not present:
                return f"required process pattern is missing: {pattern}"

    if args.min_avail_gb is not None and mem_available_gib < args.min_avail_gb:
        return f"system available memory {mem_available_gib:.1f} GiB < {args.min_avail_gb:.1f} GiB"
    if args.max_target_rss_gb is not None and target_rss_gib > args.max_target_rss_gb:
        return f"target RSS {target_rss_gib:.1f} GiB > {args.max_target_rss_gb:.1f} GiB"
    if args.max_swap_used_gb is not None and swap_used_gib > args.max_swap_used_gb:
        return f"swap used {swap_used_gib:.1f} GiB > {args.max_swap_used_gb:.1f} GiB"
    if args.max_load1 is not None and load1 > args.max_load1:
        return f"system load1 {load1:.2f} > {args.max_load1:.2f}"

    pressure_checks = [
        ("cpu some avg10", metrics.get("cpu_pressure_some_avg10"), args.max_cpu_pressure_some_avg10),
        ("memory some avg10", metrics.get("memory_pressure_some_avg10"), args.max_memory_pressure_some_avg10),
        ("memory full avg10", metrics.get("memory_pressure_full_avg10"), args.max_memory_pressure_full_avg10),
        ("io some avg10", metrics.get("io_pressure_some_avg10"), args.max_io_pressure_some_avg10),
        ("io full avg10", metrics.get("io_pressure_full_avg10"), args.max_io_pressure_full_avg10),
    ]
    for label, value, limit in pressure_checks:
        if limit is not None and value is not None and float(value) > limit:
            return f"{label} pressure {float(value):.2f}% > {limit:.2f}%"

    for gpu in metrics.get("gpus", []):
        if not isinstance(gpu, dict):
            continue
        index = gpu.get("index", "unknown")
        temp = gpu.get("temperature_c")
        total_used = gpu.get("total_used_gib")
        target_used = gpu.get("target_used_gib")
        if args.max_gpu_temp_c is not None and temp is not None and float(temp) > args.max_gpu_temp_c:
            return f"GPU {index} temperature {float(temp):.1f} C > {args.max_gpu_temp_c:.1f} C"
        if (
            args.max_total_gpu_mem_gb is not None
            and total_used is not None
            and float(total_used) > args.max_total_gpu_mem_gb
        ):
            return f"GPU {index} total memory {float(total_used):.1f} GiB > {args.max_total_gpu_mem_gb:.1f} GiB"
        if (
            args.max_target_gpu_mem_gb is not None
            and target_used is not None
            and float(target_used) > args.max_target_gpu_mem_gb
        ):
            return f"GPU {index} target memory {float(target_used):.1f} GiB > {args.max_target_gpu_mem_gb:.1f} GiB"

    return None


def print_metrics(metrics: Dict[str, object]) -> None:
    gpu_bits = []
    for gpu in metrics.get("gpus", []):
        if not isinstance(gpu, dict):
            continue
        gpu_bits.append(
            "gpu{idx}:target={target}GiB,total={total}GiB,temp={temp}C".format(
                idx=gpu.get("index"),
                target=gpu.get("target_used_gib"),
                total=gpu.get("total_used_gib"),
                temp=gpu.get("temperature_c"),
            )
        )
    gpu_text = " ".join(gpu_bits) if gpu_bits else "gpu=n/a"
    print(
        "[safe-run] rss={rss}GiB avail={avail}GiB swap={swap}GiB load1={load1:.2f} cpu_psi={cpu_psi} mem_psi={mem_psi} io_psi={io_psi} {gpu}".format(
            rss=metrics["target_rss_gib"],
            avail=metrics["mem_available_gib"],
            swap=metrics["swap_used_gib"],
            load1=float(metrics["load1"]),
            cpu_psi=metrics.get("cpu_pressure_some_avg10"),
            mem_psi=metrics.get("memory_pressure_some_avg10"),
            io_psi=metrics.get("io_pressure_full_avg10"),
            gpu=gpu_text,
        ),
        file=sys.stderr,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run or monitor a process and terminate it if resource thresholds are crossed.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--pid", type=int, help="monitor an existing process tree by PID")
    target.add_argument("command", nargs=argparse.REMAINDER, help="command to run after --")

    parser.add_argument("--interval", type=float, default=5.0, help="monitor interval in seconds")
    parser.add_argument("--grace-seconds", type=float, default=20.0, help="time between SIGTERM and SIGKILL")
    parser.add_argument("--min-avail-gb", type=float, default=32.0, help="kill when system MemAvailable is below this GiB")
    parser.add_argument("--max-swap-used-gb", type=float, default=1.0, help="kill when swap usage exceeds this GiB")
    parser.add_argument("--max-target-rss-gb", type=float, help="kill when target process tree RSS exceeds this GiB")
    parser.add_argument("--max-target-gpu-mem-gb", type=float, help="kill when target GPU memory on any GPU exceeds this GiB")
    parser.add_argument("--max-total-gpu-mem-gb", type=float, help="kill when total memory used on any GPU exceeds this GiB")
    parser.add_argument("--max-gpu-temp-c", type=float, default=82.0, help="kill when any GPU exceeds this temperature")
    parser.add_argument("--max-load1", type=float, help="kill when system 1-minute load average exceeds this")
    parser.add_argument("--max-cpu-pressure-some-avg10", type=float, help="kill when CPU PSI some avg10 exceeds this percent")
    parser.add_argument("--max-memory-pressure-some-avg10", type=float, help="kill when memory PSI some avg10 exceeds this percent")
    parser.add_argument("--max-memory-pressure-full-avg10", type=float, help="kill when memory PSI full avg10 exceeds this percent")
    parser.add_argument("--max-io-pressure-some-avg10", type=float, help="kill when IO PSI some avg10 exceeds this percent")
    parser.add_argument("--max-io-pressure-full-avg10", type=float, help="kill when IO PSI full avg10 exceeds this percent")
    parser.add_argument(
        "--require-process-name",
        action="append",
        default=[],
        help="kill target if no running process command/comm contains this text; repeatable",
    )
    parser.add_argument("--log-dir", type=Path, default=Path("safe_run_logs"), help="directory for JSONL metric logs")
    parser.add_argument("--label", default="safe-run", help="label used in log file name")
    parser.add_argument("--quiet", action="store_true", help="do not print periodic metrics")
    parser.add_argument(
        "--kill-process-group",
        action="store_true",
        help="for --pid mode, signal the whole process group instead of the process tree",
    )
    return parser


def normalize_command(command: Sequence[str]) -> List[str]:
    command = list(command)
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        raise SystemExit("missing command; use: safe_run.py [limits] -- your_command ...")
    return command


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    child: Optional[subprocess.Popen[str]] = None
    use_process_group = False
    if args.pid is not None:
        root_pid = args.pid
        if not process_exists(root_pid):
            print(f"[safe-run] pid {root_pid} does not exist", file=sys.stderr)
            return 2
        use_process_group = args.kill_process_group
        label = args.label if args.label != "safe-run" else f"pid-{root_pid}"
    else:
        command = normalize_command(args.command)
        child = subprocess.Popen(command, start_new_session=True)
        root_pid = child.pid
        use_process_group = True
        label = args.label if args.label != "safe-run" else Path(command[0]).name
        print(f"[safe-run] started pid {root_pid}: {' '.join(command)}", file=sys.stderr)

    log_path = make_log_path(args.log_dir, label)
    print(f"[safe-run] logging metrics to {log_path}", file=sys.stderr)

    exit_code = 0
    killed_reason: Optional[str] = None
    with log_path.open("a", encoding="utf-8") as log:
        while True:
            if child is not None:
                polled = child.poll()
                if polled is not None:
                    exit_code = polled if polled >= 0 else 128 + abs(polled)
                    break
            elif not process_exists(root_pid):
                exit_code = 0
                break

            metrics = collect_metrics(root_pid, args.require_process_name)
            log.write(json.dumps(metrics, sort_keys=True) + "\n")
            log.flush()
            if not args.quiet:
                print_metrics(metrics)

            reason = threshold_reason(args, metrics)
            if reason:
                killed_reason = reason
                terminate_target(
                    pid=root_pid,
                    reason=reason,
                    use_process_group=use_process_group,
                    grace_seconds=args.grace_seconds,
                )
                if child is not None:
                    try:
                        child.wait(timeout=max(1.0, args.grace_seconds))
                    except subprocess.TimeoutExpired:
                        pass
                    polled = child.poll()
                    exit_code = polled if polled is not None and polled >= 0 else 137
                else:
                    exit_code = 137
                break

            time.sleep(max(0.5, args.interval))

    if killed_reason:
        print(f"[safe-run] stopped target: {killed_reason}", file=sys.stderr)
    else:
        print(f"[safe-run] target exited with code {exit_code}", file=sys.stderr)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
