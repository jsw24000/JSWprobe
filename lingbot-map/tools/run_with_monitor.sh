#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  tools/run_with_monitor.sh [options] -- command [args...]

Options:
  --interval SEC       Sampling interval. Default: 5
  --cpu-limit PCT      Warn when process tree CPU exceeds PCT.
                       100 means one logical CPU, 800 means eight logical CPUs.
  --mem-limit MB       Warn when process tree RSS memory exceeds MB.
  --kill-on-limit      Terminate the process tree if CPU or memory exceeds a limit.
  --log FILE           CSV log path. Default: auto-generated from command
                       and timestamp, e.g. monitor_python_demo_py_...csv

Example:
  tools/run_with_monitor.sh --interval 5 --cpu-limit 800 --mem-limit 64000 -- \
    python demo.py --model_path ./checkpoints/lingbot-map.pt \
      --image_folder example/church --mask_sky --use_sdpa
USAGE
}

interval=5
cpu_limit=""
mem_limit=""
kill_on_limit=0
log_file=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --interval)
      interval="$2"
      shift 2
      ;;
    --cpu-limit)
      cpu_limit="$2"
      shift 2
      ;;
    --mem-limit)
      mem_limit="$2"
      shift 2
      ;;
    --kill-on-limit)
      kill_on_limit=1
      shift
      ;;
    --log)
      log_file="$2"
      shift 2
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    --)
      shift
      break
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ $# -eq 0 ]]; then
  echo "Missing command. Use -- before the command to run." >&2
  usage >&2
  exit 2
fi

clock_hz="$(getconf CLK_TCK)"

collect_tree() {
  local root="$1"
  local queue=("$root")
  local out=()
  local pid child children

  while ((${#queue[@]} > 0)); do
    pid="${queue[0]}"
    queue=("${queue[@]:1}")
    [[ -d "/proc/$pid" ]] || continue
    out+=("$pid")
    children="$(pgrep -P "$pid" 2>/dev/null || true)"
    while read -r child; do
      [[ -n "$child" ]] && queue+=("$child")
    done <<< "$children"
  done

  printf '%s\n' "${out[@]}"
}

proc_ticks() {
  local pid="$1"
  local stat rest utime stime
  [[ -r "/proc/$pid/stat" ]] || {
    echo 0
    return
  }
  stat="$(<"/proc/$pid/stat")"
  rest="${stat##*) }"
  read -r _ _ _ _ _ _ _ _ _ _ _ utime stime _ <<< "$rest"
  echo "$((utime + stime))"
}

tree_ticks() {
  local total=0
  local pid
  while read -r pid; do
    [[ -n "$pid" ]] || continue
    total="$((total + $(proc_ticks "$pid")))"
  done < <(collect_tree "$1")
  echo "$total"
}

tree_rss_kb() {
  local total=0
  local pid rss
  while read -r pid; do
    [[ -n "$pid" ]] || continue
    if [[ -r "/proc/$pid/status" ]]; then
      rss="$(awk '/^VmRSS:/ {print $2}' "/proc/$pid/status")"
      total="$((total + ${rss:-0}))"
    fi
  done < <(collect_tree "$1")
  echo "$total"
}

gpu_snapshot() {
  local output
  if command -v nvidia-smi >/dev/null 2>&1; then
    output="$(
      nvidia-smi --query-gpu=utilization.gpu,memory.used,power.draw \
        --format=csv,noheader,nounits 2>/dev/null || true
    )"
    if [[ "$output" == *","* ]]; then
      printf '%s\n' "$output" | paste -sd ';' -
    fi
  fi
}

command_slug() {
  local raw="$*"
  local slug
  slug="$(
    printf '%s' "$raw" |
      tr -cs '[:alnum:]._=-' '_' |
      sed -e 's/^_*//' -e 's/_*$//' -e 's/__*/_/g' |
      cut -c 1-140
  )"
  printf '%s' "${slug:-command}"
}

"$@" &
child_pid="$!"

if [[ -z "$log_file" ]]; then
  log_file="monitor_$(command_slug "$@")_$(date +%Y%m%d_%H%M%S).csv"
fi

terminate_tree() {
  local root="$1"
  local pids
  pids="$(collect_tree "$root" | tac | tr '\n' ' ')"
  [[ -n "$pids" ]] || return 0
  kill -TERM $pids 2>/dev/null || true
  sleep 5
  kill -KILL $pids 2>/dev/null || true
}

echo "Started PID $child_pid"
echo "Logging to $log_file"
echo "timestamp,pid_count,cpu_percent,rss_mb,gpu_snapshot,cpu_limit,mem_limit,cpu_exceeded,mem_exceeded,limit_exceeded" > "$log_file"

prev_ticks="$(tree_ticks "$child_pid")"
prev_time="$(date +%s.%N)"

cleanup() {
  if kill -0 "$child_pid" 2>/dev/null; then
    terminate_tree "$child_pid"
  fi
}
trap cleanup INT TERM

while kill -0 "$child_pid" 2>/dev/null; do
  sleep "$interval"

  now_ticks="$(tree_ticks "$child_pid")"
  now_time="$(date +%s.%N)"
  rss_kb="$(tree_rss_kb "$child_pid")"
  pid_count="$(collect_tree "$child_pid" | wc -l)"

  cpu_percent="$(
    awk -v dticks="$((now_ticks - prev_ticks))" \
        -v hz="$clock_hz" \
        -v t0="$prev_time" \
        -v t1="$now_time" \
        'BEGIN { elapsed=t1-t0; if (elapsed <= 0) elapsed=1; printf "%.1f", dticks / hz / elapsed * 100 }'
  )"
  rss_mb="$((rss_kb / 1024))"
  exceeded=0
  cpu_exceeded=0
  mem_exceeded=0

  if [[ -n "$cpu_limit" ]]; then
    if awk -v value="$cpu_percent" -v limit="$cpu_limit" 'BEGIN { exit !(value > limit) }'; then
      cpu_exceeded=1
      exceeded=1
    fi
  fi
  if [[ -n "$mem_limit" && "$rss_mb" -gt "$mem_limit" ]]; then
    mem_exceeded=1
    exceeded=1
  fi

  printf '%s,%s,%s,%s,"%s",%s,%s,%s,%s,%s\n' \
    "$(date --iso-8601=seconds)" \
    "$pid_count" \
    "$cpu_percent" \
    "$rss_mb" \
    "$(gpu_snapshot)" \
    "${cpu_limit:-}" \
    "${mem_limit:-}" \
    "$cpu_exceeded" \
    "$mem_exceeded" \
    "$exceeded" | tee -a "$log_file"

  if [[ "$exceeded" -eq 1 && "$kill_on_limit" -eq 1 ]]; then
    echo "Limit exceeded; terminating process tree rooted at $child_pid" >&2
    terminate_tree "$child_pid"
    break
  fi

  prev_ticks="$now_ticks"
  prev_time="$now_time"
done

wait "$child_pid"
