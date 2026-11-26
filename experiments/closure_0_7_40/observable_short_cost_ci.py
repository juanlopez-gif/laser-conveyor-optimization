"""Paired 95% intervals of J_when_idle(3.33 s) - J_when_idle(10 s) for the 19 observable-key cells (seeds 9000-9059)."""
import ast, sys
sys.path.insert(0, "experiments"); sys.path.insert(0, "src")
import recheck_0_7_39 as ref
for line in open("docs/papers/paper2/data/recheck_0_7_39_geometry.txt"):
    if line.startswith("observable-key curve"):
        k = ast.literal_eval(line[len("observable-key curve "):line.index(") {")+1])
        short = ref.geometry_config(k, "when_idle", 10/3); full = ref.geometry_config(k, "when_idle", 10)
        d = [ref.run(short, s)["penalized_flow_time"] - ref.run(full, s)["penalized_flow_time"] for s in ref.SEEDS]
        m, h = ref.interval(d)
        print(repr(k), m, h, flush=True)
