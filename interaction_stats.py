"""Offline calibration: python interaction_stats.py EXTRACTED_DR_V1_1 --out interaction.json"""
import argparse
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
from pyproj import Transformer
import shapely

CHINA = "DR_CHN_Merging_ZS"
USA = "DR_USA_Intersection_MA"
# Continuous lanes in the eastbound carriageway, traced through the ZS Lanelet2 map.
EAST = (
    (30008, 30005, 30002, 30000, 30039, 30029, 30042, 30019),
    (30007, 30004, 30040, 30016, 30022, 30023, 30037, 30018),
    (30006, 30003, 30001, 30017, 30021, 30028),
)
# Westbound: first lanelets after the two feeder lanes reach the main carriageway.
FEEDER = {30024, 30025}
MAIN = {30030, 30038, 30014, 30026, 30015, 30045}  # adjacent mainline lane


def polygons(path):
    tree = ET.parse(path)
    project = Transformer.from_crs(4326, 32631, always_xy=True).transform
    origin = np.array(project(0, 0))
    nodes = {n.get("id"): np.array(project(float(n.get("lon")), float(n.get("lat"))))-origin
             for n in tree.findall("node")}
    ways = {w.get("id"): np.array([nodes[n.get("ref")] for n in w.findall("nd")])
            for w in tree.findall("way")}
    result = {}
    for relation in tree.findall("relation"):
        sides = {m.get("role"): ways[m.get("ref")] for m in relation.findall("member")
                 if m.get("role") in ("left", "right")}
        if len(sides) != 2:
            continue
        left, right = sides["left"], sides["right"]
        if np.linalg.norm(left[0]-right[0]) < np.linalg.norm(left[0]-right[-1]):
            right = right[::-1]
        result[int(relation.get("id"))] = shapely.Polygon(np.vstack([left, right]))
    return result


def describe(values):
    a = np.asarray(values, dtype=float)
    if not a.size or not np.isfinite(a).all():
        raise ValueError("Insufficient or invalid calibration measurements")
    return {"n": int(a.size), **{f"p{q}": round(float(np.percentile(a, q)), 4)
                                for q in (10, 25, 50, 75, 90)}}


def analyze_site(root, site, lanes):
    files = sorted((root/"recorded_trackfiles"/site).glob("vehicle_tracks_*.csv"))
    stats = {k: [] for k in ("speed_mps", "neighbors_25m", "lane_flow_proxy_vph",
                              "change_front_center_m", "change_rear_center_m", "merge_main_lead_m")}
    recordings = []
    for path in files:
        df = pd.read_csv(path)
        cars = df[df.agent_type == "car"]
        recording = {"file": path.name, "cars": int(cars.track_id.nunique()), "rows": len(cars),
                     "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        recordings.append(recording)
        # Sample traffic every second. Use 0.2 s for map-based maneuver detection.
        df = cars[cars.timestamp_ms % 200 == 0].copy()
        df["lane"] = -1
        if lanes:
            x, y = df.x.to_numpy(), df.y.to_numpy()
            for label, poly in lanes.items():
                df.loc[shapely.contains_xy(poly, x, y), "lane"] = label
        corridor = {lane: i for i, group in enumerate(EAST) for lane in group}
        df["corridor"] = df.lane.map(corridor).fillna(-1).astype(int)
        df = df.sort_values(["track_id", "timestamp_ms"])
        by_track = df.groupby("track_id")
        prev = by_track.corridor.shift()
        valid = (by_track.timestamp_ms.diff() == 200) & (prev >= 0) & (df.corridor >= 0)
        # Persist on each side at least one more sample; reject boundary flicker.
        df["change"] = (valid & ((df.corridor-prev).abs() == 1)
                        & (by_track.corridor.shift(2) == prev)
                        & (by_track.corridor.shift(-1) == df.corridor))
        feeder = df.lane.isin(FEEDER)
        df["merge_entry"] = feeder & ~df.track_id.where(feeder).duplicated() & (df.vx < 0)
        # Only count entrants seen upstream of the junction in a feeder lane.
        df["merge_entry"] &= by_track.lane.shift().isin([30032, 30044])
        flow_groups = {}
        for t, frame in df.groupby("timestamp_ms"):
            sample = t % 1000 == 0
            if not sample and not (frame.change.any() or frame.merge_entry.any()):
                continue
            xy = frame[["x","y"]].to_numpy()
            heading = frame.psi_rad.to_numpy()
            delta = xy[None,:,:] - xy[:,None,:]
            forward = delta[:,:,0]*np.cos(heading[:,None]) + delta[:,:,1]*np.sin(heading[:,None])
            lateral = -delta[:,:,0]*np.sin(heading[:,None]) + delta[:,:,1]*np.cos(heading[:,None])
            same_dir = np.cos(heading[:,None]-heading) > .9
            nearby = np.linalg.norm(delta, axis=2) <= 25
            np.fill_diagonal(nearby, False)
            if sample:
                speed = np.hypot(frame.vx.to_numpy(), frame.vy.to_numpy())
                stats["speed_mps"].extend(speed)
                stats["neighbors_25m"].extend(nearby.sum(axis=1))
                # Density in a 50 m same-direction strip, anchored at an observed car.
                following = nearby & same_dir & (abs(lateral) <= 1.7)
                flow_groups.setdefault(t//10000, []).extend(following.sum(axis=1)*speed*3600/50)
            for i in np.flatnonzero(frame.change.to_numpy()):
                target = frame.corridor.to_numpy() == frame.corridor.iloc[i]
                offsets = forward[i, target & same_dir[i]]
                for side, distances in (("front", offsets), ("rear", -offsets)):
                    positive = distances[distances > 0]
                    if positive.size:
                        stats[f"change_{side}_center_m"].append(positive.min())
            for i in np.flatnonzero(frame.merge_entry.to_numpy()):
                offsets = forward[i, frame.lane.isin(MAIN).to_numpy() & same_dir[i]]
                if offsets.size:
                    stats["merge_main_lead_m"].append(offsets[np.argmin(abs(offsets))])
        stats["lane_flow_proxy_vph"].extend(np.mean(v) for v in flow_groups.values())
    if not files:
        raise FileNotFoundError(site)
    if lanes and any(not stats[k] for k in ("change_front_center_m","change_rear_center_m","merge_main_lead_m")):
        raise ValueError("Missing maneuver observations")
    return {"recordings": recordings, "metrics": {k: describe(v) for k,v in stats.items() if v}}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--out", type=Path, default=Path("interaction.json"))
    args = parser.parse_args()
    map_path = args.root/"maps"/f"{CHINA}.osm"
    result = {"release":"INTERACTION-Dataset-DR-v1_1",
              "revision":"3587152b529f26649ada0838a308eeaf77f5bdc2",
              "map_sha256":hashlib.sha256(map_path.read_bytes()).hexdigest(),
              CHINA:analyze_site(args.root, CHINA, polygons(map_path)),
              USA:analyze_site(args.root, USA, {})}
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False)+"\n", encoding="utf-8")

