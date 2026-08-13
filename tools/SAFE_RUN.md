# Safe Run Watchdog

`tools/safe_run.py` runs a command behind a local watchdog. It logs metrics to
`safe_run_logs/*.jsonl` and stops only the wrapped command/process tree when a
threshold is crossed.

## Recommended command

```bash
systemd-run --user --scope \
  -p MemoryMax=150G \
  -p MemorySwapMax=2G \
  python3 tools/safe_run.py \
    --min-avail-gb 32 \
    --max-swap-used-gb 1 \
    --max-target-rss-gb 145 \
    --max-target-gpu-mem-gb 92 \
    --max-gpu-temp-c 82 \
    --interval 5 \
    --label train \
    -- python3 your_train.py
```

`systemd-run` is the hard containment layer. `safe_run.py` is the observable
watchdog layer: it prints and records why a job was stopped.

## More conservative remote command

Use this when you are connected through ToDesk and want the job to stop if the
desktop/remote session starts looking unhealthy:

```bash
systemd-run --user --scope \
  -p MemoryMax=140G \
  -p MemorySwapMax=1G \
  -p CPUQuota=6400% \
  python3 tools/safe_run.py \
    --min-avail-gb 40 \
    --max-swap-used-gb 0.5 \
    --max-target-rss-gb 135 \
    --max-target-gpu-mem-gb 90 \
    --max-gpu-temp-c 80 \
    --max-cpu-pressure-some-avg10 30 \
    --max-memory-pressure-some-avg10 1 \
    --max-io-pressure-full-avg10 5 \
    --require-process-name ToDesk_Session \
    --interval 3 \
    --label remote-train \
    -- python3 your_train.py
```

`--require-process-name ToDesk_Session` means: if the active ToDesk video
session disappears while the job is running, stop the job. Remove that option
when you are running locally or before the remote session is established.

## Monitor an already running process

```bash
python3 tools/safe_run.py --pid 12345 --max-target-rss-gb 145 --min-avail-gb 32
```

For `--pid` mode, the script signals the process tree by default. Add
`--kill-process-group` only when you are sure the PID has its own process group.

## Useful thresholds for this host

- CPU cores: 128. A high load average is not automatically bad on this machine.
  Use `--max-load1` only when you know the job should not saturate the server.
- RAM: about 188 GiB. Keep at least 24-32 GiB free for the desktop, kernel, file
  cache, and recovery.
- Swap: currently 2 GiB. Treat swap usage as an early warning; heavy swap can
  make the desktop look frozen.
- GPU memory: each GPU reports about 95.6 GiB usable. Keeping a few GiB free is
  safer than filling it to the last MiB.
