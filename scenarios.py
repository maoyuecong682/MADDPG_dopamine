"""Build portable SMARTS scenarios from small SUMO networks."""
from pathlib import Path
import shutil
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
    routes = [t.Route(begin=start, end=end) for start, end in ROUTES[task]]
    entry = t.TrapEntryTactic(start_time=0, wait_to_hijack_limit_s=0, default_entry_speed=6)
    missions = [t.Mission(route=r, entry_tactic=entry) for r in routes[:2]]
    # Four reproducible traffic intensities, varied by episode seed in the runner.
    for variant, rate in enumerate((320, 480, 640, 800)):
        variant_path = path / str(variant)
        variant_path.mkdir(exist_ok=True)
        shutil.copyfile(path/"map.net.xml", variant_path/"map.net.xml")
        flow = t.Flow(route=routes[2], rate=rate, begin=1, end=90,
                      actors={t.TrafficActor(name="traffic"): 1})
        gen_scenario(t.Scenario(ego_missions=missions,
                     traffic={"traffic": t.Traffic(flows=[flow], engine="SMARTS")}),
                     output_dir=variant_path, seed=31 + variant)


if __name__ == "__main__":
    for task_name in TASKS:
        build(task_name)

