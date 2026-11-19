# save as steerability/upload_results.py
import argparse, hashlib, json, os, wandb

p = argparse.ArgumentParser()
p.add_argument("--json", required=True, help="results file to upload")
p.add_argument("--project", default="lerobot-dex-eval")
p.add_argument("--entity", default="chacloutier-4B")
p.add_argument("--name", required=True, help="e.g. dualnoise_arm0_lamp_16000_sepbank")
p.add_argument("--bank", default=None, help="placement bank the results were measured on")
p.add_argument("--checkpoint", default=None)
p.add_argument("--note", default="")
a = p.parse_args()

data = json.load(open(a.json))
cfg = {"source_file": os.path.abspath(a.json), "checkpoint": a.checkpoint, "note": a.note, "uploaded_post_hoc": True}
if a.bank:
    cfg.update(placement_bank=a.bank, placement_bank_md5=hashlib.md5(open(a.bank, "rb").read()).hexdigest())

def flat(o, pre=""):                       # every number in the file -> summary key
    if isinstance(o, dict):
        for k, v in o.items(): yield from flat(v, f"{pre}/{k}" if pre else str(k))
    elif isinstance(o, (int, float)) and not isinstance(o, bool):
        yield pre, o

run = wandb.init(entity=a.entity, project=a.project, name=a.name, config=cfg, job_type="post_hoc_upload")
rows = list(flat(data))
for k, v in rows:
    run.summary[k] = v
run.log({"results": wandb.Table(columns=["key", "value"], data=[[k, v] for k, v in rows])})
art = wandb.Artifact(a.name, type="results"); art.add_file(a.json); run.log_artifact(art)
run.finish()
print(f"uploaded {len(rows)} values -> {run.url}")