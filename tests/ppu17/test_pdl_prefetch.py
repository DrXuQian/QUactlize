"""Host contracts for the bounded first-B-tile experiment, no device needed."""
import importlib.util
import argparse
import copy
import json
from pathlib import Path
import unittest

path = Path(__file__).parent / "pdl_prefetch/run.py"
spec = importlib.util.spec_from_file_location("pdl_prefetch_run", path)
experiment = importlib.util.module_from_spec(spec)
spec.loader.exec_module(experiment)
build_spec=importlib.util.spec_from_file_location("pdl_prefetch_builder",path.with_name("build.py"))
builder=importlib.util.module_from_spec(build_spec)
build_spec.loader.exec_module(builder)


def resources(route=0, controls=None):
    controls = controls or experiment.DEFAULT_CONTROLS
    return [dict(role=role,arm=arm,device=0,threads=256 if role=="sf" else 128,
                 dynamic_bytes=0,static_bytes=0 if role=="sf" else 16000,registers=40,
                 request=controls["sf_carveout" if role=="sf" else "shared_carveout"],
                 selected=37 if controls["sf_carveout" if role=="sf" else "shared_carveout"]==-2
                          else controls["sf_carveout" if role=="sf" else "shared_carveout"],
                 default_blocks=3,selected_blocks=3)
            for role in (("target","sf") if route else ("target",)) for arm in range(3)]


class PrefetchContracts(unittest.TestCase):
    def test_inventory(self):
        points = experiment.correctness_points()
        self.assertEqual(len(points), 96)
        self.assertEqual(len({tuple(p.items()) for p in points}), 96)
        self.assertEqual({p["q"] for p in points}, {8,10,11,12,13,14})
        self.assertEqual({p["compute"] for p in points}, {0,1})
        self.assertEqual({p["route"] for p in points}, {0,1})
        for p in points:
            self.assertEqual(p["k"] % (512 if p["q"] in (11,14) else 256), 0)
        self.assertTrue(any(p["rows"] == "0,0,1,0" for p in points))
        self.assertTrue(any(p["n"] == 48 and p["k"] == 1024 for p in points))
        self.assertEqual(len(experiment.timing_points()), 24)

    def test_chain_receipt(self):
        point = experiment.correctness_points()[0]
        line = ("PPU17_PDL_CHAIN PASS q=8 compute=0 route=0 m=1 n=256 k=512 "
                "input=0 output=2 config=5 grouped=0 comparisons=36 prefetch=1 "
                "raw_equal=1 cpu_oracle=PASS stream=PASS graph=PASS timing=NOT_MEASURED "
                "sf_trigger=early shared_carveout=-1 sf_carveout=-1\n"
                "PPU17_PDL_RESOURCES " + json.dumps(resources()))
        experiment.validate_chain(line, point)
        for old, new in (("m=1 ","m=2 "), ("comparisons=36","comparisons=24"),
                         ("prefetch=1","prefetch=0"), ("n=256","n=48"),
                         ("raw_equal=1","raw_equal=0"), ("cpu_oracle=PASS","cpu_oracle=SKIP")):
            with self.assertRaises(ValueError):
                experiment.validate_chain(line.replace(old,new),point)
        with self.assertRaises(ValueError):
            experiment.validate_chain(line+"\n"+line, point)

    def test_timing_receipt(self):
        point = experiment.timing_points()[0]
        row = dict(q=12, compute=0, route=0, m=1, n=512, k=2048, experts=1,
                   cache="warm", raw_equal=True, producer_config=2, consumer_config=5,
                   metric_scope="GEMM_OPTIONAL_SF_GEMM_CHAIN", warmup_graphs_per_arm=3,
                   **experiment.DEFAULT_CONTROLS, inline_controls=True,resources=resources(),
                   arms={name:dict(median_us=2., samples_us=[2.]*11)
                         for name in ("ordinary","protocol","prefetch")})
        experiment.validate_timing(row, point)
        for key, value in (("raw_equal",False), ("m",2), ("metric_scope","CONSUMER_ONLY")):
            with self.assertRaises(ValueError):
                experiment.validate_timing({**row, key:value},point)
        row["arms"]["prefetch"]["samples_us"][0] = float("nan")
        with self.assertRaises(ValueError):
            experiment.validate_timing(row, point)
        row["arms"]["prefetch"]["samples_us"][0] = 2.
        row["arms"]["prefetch"]["median_us"] = float("nan")
        with self.assertRaises(ValueError):
            experiment.validate_timing(row, point)

    def test_resource_selection_receipts(self):
        point=dict(route=1)
        controls={**experiment.DEFAULT_CONTROLS,"shared_carveout":-2,"sf_carveout":-2}
        rows=resources(1,controls)
        experiment.validate_resources(rows,point,controls)
        for key,value in (("device",1),("threads",256),("request",25),("selected",-1),
                          ("default_blocks",0),("selected_blocks",2),("registers",0),
                          ("static_bytes",-1),("dynamic_bytes",1024)):
            changed=copy.deepcopy(rows);changed[0][key]=value
            with self.assertRaises(ValueError):experiment.validate_resources(changed,point,controls)
        for changed in (rows[:-1],rows+[rows[0]],rows[:5]+[rows[0]]):
            with self.assertRaises(ValueError):experiment.validate_resources(changed,point,controls)
        # Fixed25 is a diagnostic; unlike auto, an occupancy decrease is visible but legal.
        fixed={**controls,"shared_carveout":25};rows=resources(1,fixed);rows[0]["selected_blocks"]=1
        experiment.validate_resources(rows,point,fixed)

    def test_gate_bound_to_full_inventory_and_controls(self):
        points=experiment.correctness_points();identity={"manifest_sha256":"immutable"}
        controls=experiment.DEFAULT_CONTROLS
        gate=dict(identity=identity,controls=controls,status="PASS",phase="correctness",partial=False,
                  points=points,results=[dict(index=i,point=p,status="PASS") for i,p in enumerate(points)])
        experiment.validate_gate(gate,identity,controls)
        for changed in ({**gate,"controls":{**controls,"sf_trigger":"late"}},
                        {**gate,"partial":True},{**gate,"status":"FAIL"},
                        {**gate,"results":gate["results"][:-1]},
                        {**gate,"results":gate["results"][:-1]+[gate["results"][0]]}):
            with self.assertRaises(ValueError):experiment.validate_gate(changed,identity,controls)
        changed=copy.deepcopy(gate);changed["results"][3]["status"]="FAIL"
        with self.assertRaises(ValueError):experiment.validate_gate(changed,identity,controls)

    def test_control_parsing(self):
        for value,want in (("auto",-2),("-2",-2),("default",-1),("-1",-1),("25",25)):
            self.assertEqual(experiment.parse_carveout(value),want)
        for value in ("0","100","-3","nan"):
            with self.assertRaises(argparse.ArgumentTypeError):experiment.parse_carveout(value)
        args=argparse.Namespace(carveout=-2,sf_carveout=-1,sf_trigger="late",serial_sf=False)
        self.assertEqual(experiment.controls_from_args(args)["sf_trigger"],"late")
        args.serial_sf=True
        with self.assertRaises(ValueError):experiment.controls_from_args(args)

    def test_sf_release_codegen_and_negatives(self):
        source=[];native=[]
        for q in (8,10,11,12,13,14):
            for dtype in ("half", "bfloat16"):
                for trigger in ("ordinary","early","late"):
                    symbol=f"expand_scales{'_late' if trigger=='late' else ''}ILi{q}E{dtype}"
                    if trigger=="early":symbol+="Lb1E"
                    ptx=[f".visible .entry {symbol}() {{"]
                    sass=["Function : "+symbol]
                    if trigger!="ordinary":ptx+=["griddepcontrol.wait;"];sass+=["ACQBULK;"]
                    if trigger=="early":ptx+=["griddepcontrol.launch_dependents;"];sass+=["PREEXIT;"]
                    ptx += ["ld.global.u32;", "st.global.u16;"]
                    sass += ["LDG.E;", "STG.E;"]
                    if trigger=="late":ptx+=["griddepcontrol.launch_dependents;"];sass+=["PREEXIT;"]
                    source += ptx+["}"];native+=sass
        ptx="\n".join(source);sass="\n".join(native)
        self.assertEqual(len(builder.inspect_sf(ptx,sass)),36)
        for changed in (ptx.replace("griddepcontrol.wait;", "absent;", 1),
                        ptx.replace("st.global.u16;\ngriddepcontrol.launch_dependents;",
                                    "griddepcontrol.launch_dependents;\nst.global.u16;",1),
                        ptx.replace("griddepcontrol.wait;", "ld.global.u32;\ngriddepcontrol.wait;",1)):
            with self.assertRaises(ValueError):builder.inspect_sf(changed,sass)
        with self.assertRaises(ValueError):
            builder.inspect_sf(ptx,sass.replace("STG.E;\nPREEXIT;","PREEXIT;\nSTG.E;",1))


if __name__ == "__main__":
    unittest.main()
