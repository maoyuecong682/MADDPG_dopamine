"""Build portable SMARTS scenarios from small SUMO networks."""
from pathlib import Path
import json
import shutil
import sqlite3
import subprocess
from tempfile import TemporaryDirectory
from xml.etree import ElementTree as ET

from smarts.sstudio import gen_scenario, sstypes as t


ROOT = Path(__file__).resolve().parent / "scenarios"
TASKS = ("lane_change", "merge", "left_turn")
# Coordinates in metres. Keep the netconvert origin fixed for the conflict windows.
NODES = {
    "lane_change": {"w": (0, 50), "e": (140, 50)},
    "merge": {"w": (0, 50), "r": (8, 20), "j": (75, 50), "e": (155, 50)},
    "left_turn": {"w": (0, 80), "j": (80, 80), "n": (80, 160), "e": (160, 80)},
}
# edge id, endpoints, number of lanes, right-of-way priority
EDGES = {
    "lane_change": [("main","w","e",2,2)],
    "merge": [("main","w","j",1,2), ("ramp","r","j",1,1), ("out","j","e",1,2)],
    "left_turn": [("west","w","j",1,2), ("north","j","n",1,1),
                  ("east","e","j",1,2), ("through","j","w",1,2)],
}
# Two ego missions and the background traffic route for each task.
ROUTES = {
    "lane_change": [(("main",0,15),("main",1,112)), (("main",1,33),("main",0,128)),
                    (("main",0,1),("main",0,137))],
    "merge": [(("ramp",0,22),("out",0,62)), (("main",0,18),("out",0,72)),
              (("main",0,1),("out",0,74))],
    "left_turn": [(("west",0,22),("north",0,58)), (("east",0,26),("through",0,62)),
                  (("east",0,1),("through",0,72))],
}


def write_pedestrian(build_dir):
    """One SMARTS social pedestrian walking north along the east roadside."""
    # SMARTS 2.0.1 discovers .shf files in build/. Type 4 decodes to pedestrian.
    with TemporaryDirectory(dir=build_dir) as tmp:
        path = Path(tmp) / "roadside.shf"
        db = sqlite3.connect(path)
        try:
            db.executescript("""
                CREATE TABLE Spec(key TEXT PRIMARY KEY, value TEXT);
                INSERT INTO Spec VALUES ('source_type', 'scripted');
                CREATE TABLE Vehicle(id INTEGER PRIMARY KEY, type INTEGER, length REAL,
                    width REAL, height REAL, is_ego_vehicle INTEGER);
                INSERT INTO Vehicle VALUES(1, 4, 0.5, 0.5, 1.7, 0);
                CREATE TABLE Trajectory(vehicle_id INTEGER, sim_time REAL, position_x REAL,
                    position_y REAL, heading_rad REAL, speed REAL, lane_id INTEGER,
                    PRIMARY KEY(vehicle_id, sim_time));
                CREATE TABLE TrafficLightState(sim_time REAL, state INTEGER, stop_point_x REAL,
                    stop_point_y REAL, lane INTEGER);
            """)
            # Road extends to y=160; the pedestrian stops at y=150 after 50 s.
            db.executemany("INSERT INTO Trajectory VALUES(1, ?, 90, ?, 0, ?, 0)",
                           ((i/10, 90 + min(i,500)*0.12, 1.2 if i < 500 else 0.)
                            for i in range(901)))
            db.commit()
        finally:
            db.close()
        path.replace(build_dir / path.name)


def build(task):
    path = ROOT / task
    path.mkdir(parents=True, exist_ok=True)
    compiler = shutil.which("netconvert")
    if not compiler:
        raise RuntimeError("Activate the environment containing eclipse-sumo (netconvert).")
    with TemporaryDirectory() as tmp:
        nodes, edges = ET.Element("nodes"), ET.Element("edges")
        for name, (x, y) in NODES[task].items():
            ET.SubElement(nodes, "node", id=name, x=str(x), y=str(y), type="priority")
        for name, start, end, lanes, priority in EDGES[task]:
            ET.SubElement(edges, "edge", {"id":name, "from":start, "to":end,
                          "numLanes":str(lanes), "priority":str(priority), "speed":"13.9"})
        nfile, efile = Path(tmp)/"nodes.xml", Path(tmp)/"edges.xml"
        ET.ElementTree(nodes).write(nfile)
        ET.ElementTree(edges).write(efile)
        subprocess.run([compiler, "-n", str(nfile), "-e", str(efile), "--no-turnarounds",
                        "--offset.disable-normalization", "-o", str(path/"map.net.xml")], check=True)
    site = "DR_USA_Intersection_MA" if task == "left_turn" else "DR_CHN_Merging_ZS"
    profile = json.loads(Path(__file__).with_name("interaction.json").read_text())[site]["metrics"]
    entry = t.TrapEntryTactic(start_time=0, wait_to_hijack_limit_s=0, default_entry_speed=6)
    lengths = {lane.get("id"): float(lane.get("length"))
               for lane in ET.parse(path/"map.net.xml").iter("lane")}
    for variant, quantile in enumerate(("p25", "p50", "p75", "p90")):
        speeds = profile["speed_mps"]
        speed = min(13.9, max(0.5, speeds[quantile]))
        rate = max(1, round(profile["lane_flow_proxy_vph"][quantile]))
        legs = ROUTES[task].copy()
        if task == "lane_change":
            # Second policy car occupies either the lead or trailing target-lane slot.
            side = "front" if variant % 2 == 0 else "rear"
            gap = profile[f"change_{side}_center_m"]["p50" if variant < 2 else "p75"]
            gap = min(32, max(6, gap))
            offsets = (15, 15 + gap) if side == "front" else (15 + gap, 15)
            for i, offset in enumerate(offsets):
                legs[i] = (("main", i, offset), legs[i][1])
        elif task == "merge":
            # Match the signed difference of remaining distances to the junction.
            lead = profile["merge_main_lead_m"][quantile]
            offset = lengths["main_0"] - (lengths["ramp_0"] - legs[0][0][2]) + lead
            legs[1] = (("main", 0, min(lengths["main_0"]-8, max(8, offset))), legs[1][1])
        routes = [t.Route(begin=start, end=end) for start,end in legs]
        missions = [t.Mission(route=r, entry_tactic=entry) for r in routes[:2]]
        variant_path = path / str(variant)
        variant_path.mkdir(exist_ok=True)
        shutil.copyfile(path/"map.net.xml", variant_path/"map.net.xml")
        actor = t.TrafficActor(name="traffic", depart_speed=speed,
                               speed=t.Distribution(mean=speed/13.9, sigma=0),
                               max_speed=min(13.9, max(speed, speeds["p90"])))
        flow = t.Flow(route=routes[2], rate=rate, begin=1, end=90, actors={actor: 1})
        gen_scenario(t.Scenario(ego_missions=missions,
                     traffic={"traffic": t.Traffic(flows=[flow], engine="SMARTS")}),
                     output_dir=variant_path, seed=31 + variant)
        if task == "left_turn":
            write_pedestrian(variant_path/"build")


if __name__ == "__main__":
    for task_name in TASKS:
        build(task_name)

