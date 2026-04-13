#!/usr/bin/env python3
"""
Ziegler-Nichols PID autotuner for the mipurifier fan controller.

Connects to the ESPHome SSE event stream, sweeps Kp (with Ki=Kd=0) until
sustained oscillation is detected, measures the ultimate gain (Ku) and
oscillation period (Tu), then computes and applies PI parameters.

Usage:
    python3 pid_autotune.py [--host mipurifier-sypialnia.local] [--target-rpm 1000]
"""

import argparse
import json
import math
import statistics
import sys
import threading
import time
import urllib.parse
import urllib.request
from collections import deque
from typing import Optional


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------

def http_post(url: str) -> None:
    req = urllib.request.Request(url, data=b"", method="POST")
    with urllib.request.urlopen(req, timeout=10):
        pass


def set_number(base_url: str, name: str, value: float) -> None:
    encoded = urllib.parse.quote(name)
    url = f"{base_url}/number/{encoded}/set?value={value:.6g}"
    http_post(url)
    print(f"    SET {name} = {value:.4g}")


def fan_turn_on(base_url: str) -> None:
    http_post(f"{base_url}/fan/fan/turn_on")


def fan_turn_off(base_url: str) -> None:
    http_post(f"{base_url}/fan/fan/turn_off")


# ---------------------------------------------------------------------------
# SSE event reader (background thread → queue)
# ---------------------------------------------------------------------------

class EventReader(threading.Thread):
    """Reads ESPHome SSE events and puts (sensor_id, value) into a queue."""

    def __init__(self, base_url: str, queue: deque, stop_event: threading.Event):
        super().__init__(daemon=True)
        self.url = f"{base_url}/events"
        self.queue = queue
        self.stop_event = stop_event

    def run(self):
        while not self.stop_event.is_set():
            try:
                req = urllib.request.Request(self.url)
                with urllib.request.urlopen(req, timeout=90) as resp:
                    event_type = None
                    for raw_line in resp:
                        if self.stop_event.is_set():
                            return
                        line = raw_line.decode("utf-8", errors="replace").rstrip("\r\n")
                        if line.startswith("event:"):
                            event_type = line[6:].strip()
                        elif line.startswith("data:") and event_type == "state":
                            data_str = line[5:].strip()
                            try:
                                data = json.loads(data_str)
                                eid = data.get("id", "")
                                val = data.get("value")
                                if val is not None:
                                    self.queue.append((time.monotonic(), eid, float(val)))
                            except (json.JSONDecodeError, ValueError, KeyError):
                                pass
                            event_type = None
            except Exception as exc:
                if not self.stop_event.is_set():
                    print(f"    [SSE reconnect: {exc}]", file=sys.stderr)
                    time.sleep(2)


# ---------------------------------------------------------------------------
# Oscillation detection
# ---------------------------------------------------------------------------

def analyse_window(samples: list[tuple[float, float]], target: float) -> tuple[bool, float, float]:
    """
    Given a list of (timestamp, rpm) samples, decide whether the signal is
    showing sustained oscillation.

    Returns (oscillating, period_s, amplitude_rpm).
    """
    if len(samples) < 8:
        return False, 0.0, 0.0

    times = [t for t, _ in samples]
    values = [v for _, v in samples]
    mean = statistics.mean(values)

    # Find zero-crossings of (value - mean)
    crossings: list[float] = []
    for i in range(1, len(values)):
        a, b = values[i - 1] - mean, values[i] - mean
        if a * b < 0:
            # linear interpolation
            frac = abs(a) / (abs(a) + abs(b))
            crossings.append(times[i - 1] + frac * (times[i] - times[i - 1]))

    if len(crossings) < 4:
        return False, 0.0, 0.0

    # Half-periods between consecutive crossings
    half_periods = [crossings[i + 1] - crossings[i] for i in range(len(crossings) - 1)]
    mean_hp = statistics.mean(half_periods)
    if mean_hp <= 0:
        return False, 0.0, 0.0

    period = 2.0 * mean_hp

    # Reject if period is too variable (CV > 35 %)
    if len(half_periods) >= 3:
        cv = statistics.stdev(half_periods) / mean_hp
        if cv > 0.35:
            return False, period, 0.0

    amplitude = max(abs(v - mean) for v in values)

    # Must be at least 3 % of target to count as real oscillation
    if amplitude < 0.03 * target:
        return False, period, amplitude

    # Check amplitude stability: compare first half vs second half
    mid = len(values) // 2
    amp_first = max(abs(v - mean) for v in values[:mid])
    amp_second = max(abs(v - mean) for v in values[mid:])
    if amp_first > 0 and amp_second / amp_first > 3.0:
        # Growing — not yet at steady oscillation
        return False, period, amplitude

    return True, period, amplitude


# ---------------------------------------------------------------------------
# Main autotune routine
# ---------------------------------------------------------------------------

def autotune(base_url: str, target_rpm: float, obs_per_step: float,
             kp_steps: list[float]) -> None:

    print(f"\n=== Ziegler-Nichols Fan PID Autotuner ===")
    print(f"Host      : {base_url}")
    print(f"Target RPM: {target_rpm}")
    print(f"Kp steps  : {kp_steps}")
    print()

    # Shared event buffer
    event_buf: deque = deque(maxlen=4096)
    stop_evt = threading.Event()
    reader = EventReader(base_url, event_buf, stop_evt)
    reader.start()
    time.sleep(1)  # let SSE connect

    def drain_rpm(duration_s: float) -> list[tuple[float, float]]:
        """Collect (monotonic_time, rpm) samples for duration_s seconds."""
        event_buf.clear()
        deadline = time.monotonic() + duration_s
        samples: list[tuple[float, float]] = []
        while time.monotonic() < deadline:
            while event_buf:
                ts, eid, val = event_buf.popleft()
                if "fan_rpm" in eid:
                    samples.append((ts, val))
            time.sleep(0.05)
        return samples

    try:
        # ---- Initialise -------------------------------------------------------
        print("► Initialising: fan ON, Ki=0, Kd=0, target RPM set")
        fan_turn_on(base_url)
        set_number(base_url, "Target RPM", target_rpm)
        set_number(base_url, "PID Ki", 0.0)
        set_number(base_url, "PID Kd", 0.0)
        set_number(base_url, "PID Kp", 0.0)
        print(f"  Waiting 10 s for fan to spin up...")
        time.sleep(10)

        # ---- Kp sweep ---------------------------------------------------------
        ku: Optional[float] = None
        tu: Optional[float] = None

        for kp in kp_steps:
            print(f"\n► Kp = {kp:.4g}  (observing {obs_per_step:.0f} s)")
            set_number(base_url, "PID Kp", kp)

            # Wait for transient, then measure
            settle = min(obs_per_step * 0.4, 8.0)
            print(f"  Settling {settle:.0f} s...", end=" ", flush=True)
            time.sleep(settle)
            print("measuring...")

            measure_t = obs_per_step - settle
            samples = drain_rpm(measure_t)

            if not samples:
                print("  ⚠ No RPM data — check SSE stream.")
                continue

            vals = [v for _, v in samples]
            mean_rpm = statistics.mean(vals)
            std_rpm = statistics.stdev(vals) if len(vals) > 1 else 0.0
            print(f"  RPM  mean={mean_rpm:.0f}  std=±{std_rpm:.0f}  n={len(vals)}")

            oscillating, period, amplitude = analyse_window(samples, target_rpm)
            status = "OSCILLATING ✓" if oscillating else "stable"
            print(f"  Osc  {status}  period={period:.2f} s  amp=±{amplitude:.0f} RPM")

            if oscillating:
                ku = kp
                tu = period
                print(f"\n  ★ Ultimate gain found: Ku={ku:.4g}  Tu={tu:.3f} s")
                break

        # ---- Results ----------------------------------------------------------
        if ku is None or tu is None:
            print("\n✗ Could not find ultimate gain in the given Kp range.")
            print("  Try extending kp_steps or reducing noise (lower update_interval).")
            return

        # Ziegler-Nichols PI tuning (no derivative — RPM signal is noisy)
        kp_zn = 0.45 * ku
        ti_zn = tu / 1.2
        ki_zn = kp_zn / ti_zn
        kd_zn = 0.0

        # Some-overshoot variant (gentler than classic Z-N)
        kp_so = 0.33 * ku
        ki_so = kp_so / (tu / 2.0)
        kd_so = 0.0

        print(f"\n{'─'*50}")
        print(f"Ku = {ku:.4g}    Tu = {tu:.3f} s")
        print()
        print(f"  Classic Z-N PI :  Kp={kp_zn:.4g}  Ki={ki_zn:.4g}  Kd=0")
        print(f"  Some-overshoot :  Kp={kp_so:.4g}  Ki={ki_so:.4g}  Kd=0")
        print(f"{'─'*50}")

        choice = input("\nApply which? [1=Z-N PI, 2=some-overshoot, n=none]: ").strip().lower()
        if choice == "1":
            kp_f, ki_f, kd_f = kp_zn, ki_zn, kd_zn
        elif choice == "2":
            kp_f, ki_f, kd_f = kp_so, ki_so, kd_so
        else:
            print("Parameters not applied.")
            return

        set_number(base_url, "PID Kp", kp_f)
        set_number(base_url, "PID Ki", ki_f)
        set_number(base_url, "PID Kd", kd_f)
        print("✓ Parameters applied.")

    finally:
        stop_evt.set()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Z-N PID autotuner for mipurifier fan")
    parser.add_argument("--host", default="mipurifier-sypialnia.local",
                        help="ESPHome device hostname (default: mipurifier-sypialnia.local)")
    parser.add_argument("--target-rpm", type=float, default=1000,
                        help="RPM setpoint during tuning (default: 1000)")
    parser.add_argument("--obs", type=float, default=30,
                        help="Observation seconds per Kp step (default: 30)")
    parser.add_argument("--start-kp", type=float, default=0.01,
                        help="Skip all Kp steps below this value (default: 0.01)")
    args = parser.parse_args()

    base = f"http://{args.host}"

    # Kp values to try — geometric from 0.01 up to 3.0
    steps = [round(v, 4) for v in
             [0.01 * (10 ** (i * 0.2)) for i in range(28)]
             if v <= 3.01 and v >= args.start_kp - 1e-9]

    autotune(base_url=base, target_rpm=args.target_rpm,
             obs_per_step=args.obs, kp_steps=steps)
