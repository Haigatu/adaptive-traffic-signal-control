"""Deterministic four-arm SUMO network with explicit non-conflicting phases."""

from pathlib import Path
import subprocess, xml.etree.ElementTree as ET, uuid
import numpy as np
import sumo
from .data import ROOT, load

DIRECTIONS = ["N", "E", "S", "W"]


def binary(name):
    candidate = Path(sumo.__file__).parent / "bin" / name
    if not candidate.exists():
        candidate = candidate.with_suffix(".exe")
    return str(candidate)


def build():
    out = ROOT / "data" / "network"
    out.mkdir(exist_ok=True)
    nodes = ET.Element("nodes")
    ET.SubElement(nodes, "node", id="J", x="0", y="0", type="traffic_light")
    positions = [(0, 600), (600, 0), (0, -600), (-600, 0)]
    edges = ET.Element("edges")
    con = ET.Element("connections")
    for d, (x, y) in zip(DIRECTIONS, positions):
        ET.SubElement(nodes, "node", id=d, x=str(x), y=str(y), type="priority")
        ET.SubElement(
            edges,
            "edge",
            {"id": d + "_in", "from": d, "to": "J", "numLanes": "3", "speed": "13.89"},
        )
        ET.SubElement(
            edges,
            "edge",
            {"id": d + "_out", "from": "J", "to": d, "numLanes": "3", "speed": "13.89"},
        )
    for i, d in enumerate(DIRECTIONS):
        # Right, straight, left: clockwise cardinal order viewed by arriving driver.
        for lane, offset in enumerate([-1, 2, 1]):
            dest = DIRECTIONS[(i + offset) % 4]
            ET.SubElement(
                con,
                "connection",
                {
                    "from": d + "_in",
                    "to": dest + "_out",
                    "fromLane": str(lane),
                    "toLane": str(lane),
                    "tl": "J",
                    "linkIndex": str(i * 3 + lane),
                },
            )
    tls = ET.Element("tlLogics")
    logic = ET.SubElement(
        tls, "tlLogic", id="J", type="static", programID="0", offset="0"
    )
    for i in range(4):
        state = "".join("G" if j // 3 == i else "r" for j in range(12))
        ET.SubElement(logic, "phase", duration="27", state=state)
        ET.SubElement(logic, "phase", duration="3", state=state.replace("G", "y"))
    for name, tree in [
        ("nodes.nod.xml", nodes),
        ("edges.edg.xml", edges),
        ("connections.con.xml", con),
        ("signals.tll.xml", tls),
    ]:
        ET.ElementTree(tree).write(out / name)
    subprocess.run(
        [
            binary("netconvert"),
            "--node-files",
            str(out / "nodes.nod.xml"),
            "--edge-files",
            str(out / "edges.edg.xml"),
            "--connection-files",
            str(out / "connections.con.xml"),
            "--tllogic-files",
            str(out / "signals.tll.xml"),
            "-o",
            str(out / "intersection.net.xml"),
            "--no-turnarounds",
            "true",
        ],
        check=True,
        capture_output=True,
    )
    adds = ET.Element("additional")
    for d in DIRECTIONS:
        for l in range(3):
            ET.SubElement(
                adds,
                "inductionLoop",
                id=f"{d}_{l}",
                lane=f"{d}_in_{l}",
                pos="-50",
                period="300",
                file="NUL" if __import__("os").name == "nt" else "/dev/null",
            )
    ET.ElementTree(adds).write(out / "detectors.add.xml")
    return out


def make_routes(seed, saturation, end=7200, split="train"):
    root = ROOT / "data" / "runs"
    root.mkdir(exist_ok=True)
    path = root / f"{split}_{seed}_{saturation:.2f}_{end}.rou.xml"
    rng = np.random.default_rng(seed)
    raw, _ = load("PEMS08")
    a = int(len(raw) * 0.6)
    b = int(len(raw) * 0.8)
    lo, hi = {"train": (0, a), "tune": (a, b), "test": (b, len(raw))}[split]
    bins = int(np.ceil(end / 300))
    t0 = int(rng.integers(lo, max(lo + 1, hi - bins)))
    sensors = rng.choice(raw.shape[1], 4, replace=False)
    profile = raw[t0 : t0 + bins, sensors].copy()
    # Scenario mean is predefined by requested saturation; only scenario demand,
    # never predictor fitting, sees the complete profile.
    profile /= np.maximum(profile.mean(0, keepdims=True), 1)
    profile = np.clip(profile, 0.15, 2.2)
    flows = profile * (saturation * 1350) / 3600
    arrivals = []
    for k in range(bins):
        for i, d in enumerate(DIRECTIONS):
            n = rng.poisson(flows[k, i] * min(300, end - k * 300))
            times = k * 300 + rng.uniform(0, min(300, end - k * 300), n)
            for t in times:
                lane = int(rng.integers(0, 3))
                dest = DIRECTIONS[(i + [-1, 2, 1][lane]) % 4]
                arrivals.append(
                    (float(t), i, lane, dest, "heavy" if rng.random() < 0.1 else "car")
                )
    arrivals.sort()
    r = ET.Element("routes")
    ET.SubElement(
        r,
        "vType",
        id="car",
        vClass="passenger",
        accel="2.6",
        decel="4.5",
        sigma=".5",
        length="5",
        minGap="2.5",
        maxSpeed="13.89",
        speedFactor="1",
        speedDev="0",
    )
    ET.SubElement(
        r,
        "vType",
        id="heavy",
        vClass="truck",
        accel="1.3",
        decel="4",
        sigma=".5",
        length="10",
        minGap="2.5",
        maxSpeed="11.11",
        speedFactor="1",
        speedDev="0",
    )
    for j, (t, i, l, dest, typ) in enumerate(arrivals):
        v = ET.SubElement(
            r,
            "vehicle",
            id=f"v{j}",
            type=typ,
            depart=f"{t:.2f}",
            departLane=str(l),
            departSpeed="max",
        )
        ET.SubElement(v, "route", edges=f"{DIRECTIONS[i]}_in {dest}_out")
    tmp = path.with_suffix("." + uuid.uuid4().hex + ".tmp")
    ET.ElementTree(r).write(tmp)
    tmp.replace(path)
    return path, {
        "requested": len(arrivals),
        "seed": seed,
        "saturation": saturation,
        "split": split,
        "profile_start": t0,
        "sensors": sensors.tolist(),
        "departures": {f"v{j}": t for j, (t, *_) in enumerate(arrivals)},
    }


if __name__ == "__main__":
    print(build())
