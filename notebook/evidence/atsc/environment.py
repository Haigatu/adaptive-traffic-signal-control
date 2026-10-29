"""Actual microscopic SUMO environment, 0.5 s integration, 10 s decisions.

TTC is a same-lane rear-end surrogate only; events are threshold-entry episodes.
No fabricated collision labels, no oracle future state, no autonomous teleports.
"""

from collections import deque
import uuid
import numpy as np
import torch
import importlib
import os
import shutil
from pathlib import Path
import xml.etree.ElementTree as ET
from .data import ROOT
from .network import build, make_routes, DIRECTIONS, binary
from .models import Forecaster


class Environment:
    def __init__(
        self,
        predictor=None,
        with_forecast=True,
        w=1.0,
        duration=3600,
        warmup=3600,
        dscale=1000.0,
        cscale=1.0,
        record=False,
        backend="libsumo",
        gui=False,
        run_dir=None,
        gui_options=None,
        gui_name=None,
    ):
        if backend not in ("libsumo", "traci") or (gui and backend != "traci"):
            raise ValueError("GUI requires backend='traci'")
        self.sim = importlib.import_module(backend)
        self.gui = gui
        self.gui_options = gui_options or []
        self.gui_name = gui_name
        self.run_dir = Path(run_dir) if run_dir else None
        if self.run_dir:
            self.run_dir.mkdir(parents=True, exist_ok=True)
        self.with_forecast = with_forecast
        self.w = w
        self.duration = duration
        self.warmup = warmup
        self.dscale = dscale
        self.cscale = cscale
        self.record = record
        self.model = None
        self.adj = torch.eye(
            4
        ).to_sparse()  # isolated approach nodes; no invented upstream links
        if predictor:
            ck = torch.load(predictor, map_location="cpu", weights_only=False)
            self.model = Forecaster(ck["kind"], ck["hidden"])
            self.model.load_state_dict(ck["state_dict"])
            self.model.eval()
            self.mean = ck["mean"]
            self.std = ck["std"]
        self.net = ROOT / "data" / "network"
        if not (self.net / "intersection.net.xml").exists():
            build()
        self.inlanes = [f"{d}_in_{i}" for d in DIRECTIONS for i in range(3)]
        self.outlanes = [f"{d}_out_{i}" for d in DIRECTIONS for i in range(3)]
        self.running = False

    def reset(self, seed, saturation, split="train"):
        if self.running:
            self.close()
        route, self.meta = make_routes(
            seed, saturation, self.warmup + self.duration, split, output_dir=self.run_dir
        )
        self.trip = (
            (self.run_dir or ROOT / "data" / "runs")
            / f"trip_{split}_{seed}_{saturation:.2f}_{uuid.uuid4().hex}.xml"
        )
        detectors = self.net / "detectors.add.xml"
        network = self.net / "intersection.net.xml"
        if self.gui_name and self.run_dir:
            display_network = self.run_dir / f"{self.gui_name}.net.xml"
            shutil.copyfile(network, display_network)
            network = display_network
        if self.run_dir or os.name == "nt":
            tree = ET.parse(detectors)
            for detector in tree.getroot():
                detector.set("file", "NUL" if os.name == "nt" else "/dev/null")
            detectors = self.trip.with_suffix(".detectors.add.xml")
            tree.write(detectors)
        self.sim.start(
            [
                binary("sumo-gui" if self.gui else "sumo"),
                "-n",
                str(network),
                "-r",
                str(route),
                "-a",
                str(detectors),
                "--step-length",
                ".5",
                "--seed",
                str(seed),
                "--no-step-log",
                "true",
                "--no-warnings",
                "true",
                "--duration-log.disable",
                "true",
                "--time-to-teleport",
                "-1",
                "--tripinfo-output",
                str(self.trip),
                "--tripinfo-output.write-unfinished",
                "true",
                "--waiting-time-memory",
                "3600",
                "--xml-validation",
                "never",
            ] + (["--start", "--quit-on-end", "--delay", "0", *self.gui_options] if self.gui else [])
        )
        self.running = True
        self.phase = 0
        self.elapsed = 0
        self.total_steps = 0
        self.switches = 0
        self.history = deque(maxlen=12)
        self.all_counts = []
        self.bucket = np.zeros(4, np.float32)
        self.last_det = {}
        self.forecast = np.zeros((3, 4), np.float32)
        self.last_time = 0
        self.active_pairs = set()
        self.seen_ids = set()
        self.entered = {}
        self.arrival_times = {}
        self.trace = []
        self.delay = 0.0
        self.conflicts = 0
        self.tet = 0.0
        self.wait = 0.0
        self.qsum = 0.0
        self.co2 = 0.0
        self.arrived = 0
        self.teleports = 0
        self.collisions = 0
        # Warm-up supplies actual sensor history; metrics begin after warm-up.
        while self.sim.simulation.getTime() < self.warmup:
            t = self.sim.simulation.getTime()
            p = int(t // 30) % 4
            within = t % 30
            self._signal(p, yellow=within >= 27)
            self._tick(measure=False)
        self.phase = int((self.warmup // 30) % 4)
        self._signal(self.phase)
        self._refresh_forecast()
        self.measure_start = float(self.sim.simulation.getTime())
        return self.state()

    def _signal(self, phase, yellow=False):
        c = "y" if yellow else "G"
        state = "".join(c if i // 3 == phase else "r" for i in range(12))
        self.sim.trafficlight.setRedYellowGreenState("J", state)

    def _refresh_forecast(self):
        if len(self.history) < 12:
            return
        x = np.stack(self.history)
        if self.model is None:
            self.forecast = np.repeat(x[-1:], 3, axis=0)
        else:
            with torch.no_grad():
                self.forecast = np.maximum(
                    self.model(
                        torch.from_numpy((x - self.mean) / self.std)[None], self.adj
                    )[0].numpy()
                    * self.std
                    + self.mean,
                    0,
                )

    def _tick(self, measure=True):
        self.sim.simulationStep()
        t = self.sim.simulation.getTime()
        dt = 0.5
        for i, d in enumerate(DIRECTIONS):
            for l in range(3):
                det = f"{d}_{l}"
                ids = set(self.sim.inductionloop.getLastStepVehicleIDs(det))
                self.bucket[i] += len(ids - self.last_det.get(det, set()))
                self.last_det[det] = ids
        if abs(t % 300) < 1e-6:
            self.history.append(self.bucket.copy())
            self.all_counts.append(self.bucket.copy())
            self.bucket[:] = 0
            self._refresh_forecast()
        for v in self.sim.simulation.getDepartedIDList():
            self.entered[v] = t
        for v in self.sim.simulation.getArrivedIDList():
            self.arrival_times[v] = t
        if not measure:
            return
        self.total_steps += 1
        ids = self.sim.vehicle.getIDList()
        current = set()
        q = 0
        for v in ids:
            speed = self.sim.vehicle.getSpeed(v)
            allowed = max(self.sim.vehicle.getAllowedSpeed(v), 0.1)
            self.delay += max(0, 1 - speed / allowed) * dt
            if speed < 0.1:
                self.wait += dt
            self.co2 += self.sim.vehicle.getCO2Emission(v) * dt / 1e6  # mg to kg
            leader = self.sim.vehicle.getLeader(v, 100)
            if leader and leader[0]:
                lv, gap = leader
                # getLeader returns bumper gap minus follower's minGap.
                physical_gap = gap + self.sim.vehicle.getMinGap(v)
                closing = speed - self.sim.vehicle.getSpeed(lv)
                if (
                    self.sim.vehicle.getLaneID(v) == self.sim.vehicle.getLaneID(lv)
                    and closing > 0
                    and 0 < physical_gap / closing < 1.5
                ):
                    current.add((v, lv))
        self.conflicts += len(current - self.active_pairs)
        self.tet += len(current) * dt
        self.active_pairs = current
        for lane in self.inlanes:
            q += self.sim.lane.getLastStepHaltingNumber(lane)
        self.qsum += q
        self.arrived += self.sim.simulation.getArrivedNumber()
        self.teleports += self.sim.simulation.getStartingTeleportNumber()
        self.collisions += self.sim.simulation.getCollidingVehiclesNumber()
        # Account for demand held outside the network, so controllers cannot hide congestion.
        pending = self.sim.simulation.getPendingVehicles()
        self.delay += len(pending) * dt

    def state(self):
        q = (
            np.array(
                [self.sim.lane.getLastStepHaltingNumber(l) for l in self.inlanes], np.float32
            )
            / 80
        )
        wt = (
            np.array([self.sim.lane.getWaitingTime(l) for l in self.inlanes], np.float32)
            / 2400
        )
        count = (
            np.array(
                [self.sim.lane.getLastStepVehicleNumber(l) for l in self.inlanes], np.float32
            )
            / 80
        )
        st = np.r_[q, wt, count, np.eye(4, dtype=np.float32)[self.phase]]
        if self.with_forecast:
            st = np.r_[st, self.forecast.flatten() / 120]
        return st.astype("float32")

    def queue_by_approach(self):
        return np.array(
            [
                sum(self.sim.lane.getLastStepHaltingNumber(f"{d}_in_{i}") for i in range(3))
                for d in DIRECTIONS
            ]
        )

    def pressures(self):
        values = []
        for i, d in enumerate(DIRECTIONS):
            v = 0
            for lane, offset in enumerate([-1, 2, 1]):
                dest = DIRECTIONS[(i + offset) % 4]
                v += self.sim.lane.getLastStepHaltingNumber(
                    f"{d}_in_{lane}"
                ) - self.sim.lane.getLastStepHaltingNumber(f"{dest}_out_{lane}")
            values.append(v)
        return np.array(values)

    def step(self, action):
        action = int(action)
        d0 = self.delay
        c0 = self.conflicts
        t0 = self.sim.simulation.getTime()
        deadline = self.measure_start + self.duration
        if action != self.phase:
            self._signal(self.phase, True)
            for _ in range(6):
                if self.sim.simulation.getTime() < deadline:
                    self._tick()
            self.phase = action
            self.switches += 1
        self._signal(self.phase)
        # Ten full green seconds after every action; switching adds three yellow seconds.
        for _ in range(20):
            if self.sim.simulation.getTime() < deadline:
                self._tick()
        self.elapsed = self.sim.simulation.getTime() - self.measure_start
        delay = self.delay - d0
        conflicts = self.conflicts - c0
        reward = -(
            self.w * delay / self.dscale + (1 - self.w) * conflicts / self.cscale
        )
        done = self.elapsed >= self.duration
        if self.record:
            self.trace.append(
                {
                    "time": self.elapsed,
                    "phase": self.phase,
                    "queue": self.queue_by_approach().tolist(),
                    "forecast": self.forecast.mean(0).tolist(),
                    "vehicles": [
                        [
                            v,
                            *self.sim.vehicle.getPosition(v),
                            self.sim.vehicle.getAngle(v),
                            self.sim.vehicle.getSpeed(v),
                        ]
                        for v in self.sim.vehicle.getIDList()
                    ],
                }
            )
        return (
            self.state(),
            float(reward),
            done,
            {
                "interval_delay": delay,
                "interval_conflicts": conflicts,
                "duration": self.sim.simulation.getTime() - t0,
            },
        )

    def metrics(self):
        n_requested = sum(
            self.warmup <= t < self.warmup + self.duration
            for t in self.meta["departures"].values()
        )
        return {
            "integrated_delay_vehicle_s": self.delay,
            "delay_per_requested_vehicle_s": self.delay / max(n_requested, 1),
            "mean_queue_vehicles": self.qsum / max(self.total_steps, 1),
            "throughput": self.arrived,
            "conflicts": self.conflicts,
            "conflicts_per_1000_arrivals": 1000 * self.conflicts / max(self.arrived, 1),
            "tet_vehicle_s": self.tet,
            "waiting_vehicle_s": self.wait,
            "co2_kg": self.co2,
            "switches": self.switches,
            "teleports": self.teleports,
            "collision_vehicle_events": self.collisions,
            "requested_evaluation_vehicles": n_requested,
            "inserted_evaluation_vehicles": sum(
                t >= self.warmup for t in self.entered.values()
            ),
            "remaining_in_network": self.sim.vehicle.getIDCount(),
            "pending_insertion": len(self.sim.simulation.getPendingVehicles()),
            "elapsed_s": self.elapsed,
        }

    def close(self):
        if self.running:
            self.sim.close()
            self.running = False
