# ruff: noqa: E501
"""Build a local, self-contained HTML page for labelling clothing and vehicle colours by eye.

Usage: python scripts/colour_label_tool.py [--workspace meva-school] [--persons 150] [--vehicles 60] [--seed 1]
                                           [--out data/share/colour_label/index.html]

Open the page in a browser (double-click the file). Everything stays on this machine: the crops are embedded
in the page, labels are saved in the browser as you go, and "Download labels" writes labels.json.
The page never shows the system's own guess, so the labels are independent. Then score them with
`python scripts/colour_eval.py labels.json`.

Keys: 1 black, 2 white, 3 grey, 4 red, 5 orange, 6 yellow, 7 green, 8 blue, 9 purple, 0 pink, - brown, u unsure,
d / l mark the shade dark / light (toggle, applies to the answer you give next), Backspace goes back, Enter/space skips.
"""
from __future__ import annotations

import argparse
import base64
import json
import random
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
COLOURS = ["black", "white", "grey", "red", "orange", "yellow", "green", "blue", "purple", "pink", "brown"]
KEYS = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "0", "-"]
SWATCH = {"black": "#111", "white": "#f4f4f4", "grey": "#8a8a8a", "red": "#c62828", "orange": "#ef7d00", "yellow": "#f2d21b",
          "green": "#2e8b3e", "blue": "#1f4fbf", "purple": "#7b2fa0", "pink": "#f08cb8", "brown": "#7a4a22"}
VEHICLES = {"car", "truck", "bus", "motorcycle"}
TOOL_VERSION = 1


def pick(rows: list[dict], n: int, key: str, rng: random.Random) -> list[dict]:
    """Round-robin over the predicted colour so rare colours are covered, then fill with random leftovers."""
    buckets: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        buckets[r.get(key) or "none"].append(r)
    for b in buckets.values():
        rng.shuffle(b)
    chosen: list[dict] = []
    order = sorted(buckets, key=lambda k: len(buckets[k]))
    while len(chosen) < n and any(buckets.values()):
        for k in order:
            if buckets[k] and len(chosen) < n:
                chosen.append(buckets[k].pop())
    rng.shuffle(chosen)
    return chosen


def load_items(ws_root: Path, persons: int, vehicles: int, seed: int) -> list[dict]:
    db = sqlite3.connect(ws_root / "evora.sqlite")
    db.row_factory = sqlite3.Row
    media = ws_root / "media"
    rng = random.Random(seed)
    rows = []
    for r in db.execute("SELECT id, camera_id, cls, attrs FROM tracks WHERE cls IN ('person','car','truck','bus','motorcycle')"):
        attrs = json.loads(r["attrs"] or "{}")
        n = int(r["id"].split(":t")[1])
        crop = media / "crops" / r["camera_id"] / f"t{n:06d}_0.jpg"
        if not crop.is_file() or attrs.get("is_ir"):
            continue
        rows.append({"id": r["id"], "camera": r["camera_id"], "cls": r["cls"], "kind": "person" if r["cls"] == "person" else "vehicle",
                     "pred_upper": attrs.get("upper_color"), "pred_lower": attrs.get("lower_color"), "pred_color": attrs.get("color"),
                     "pred_conf": attrs.get("color_conf"), "path": crop})
    ppl = pick([r for r in rows if r["kind"] == "person"], persons, "pred_upper", rng)
    veh = pick([r for r in rows if r["kind"] == "vehicle"], vehicles, "pred_color", rng)
    items = ppl + veh
    for it in items:
        it["jpeg"] = base64.b64encode(it.pop("path").read_bytes()).decode("ascii")
    return items


PAGE = """<!doctype html><html lang="en"><head><meta charset="utf-8"><title>Colour labelling</title>
<meta name="viewport" content="width=device-width,initial-scale=1"><style>
body{font:15px system-ui,sans-serif;margin:0;background:#eceff1;color:#222}
header{padding:10px 16px;background:#fff;border-bottom:1px solid #cfd8dc;display:flex;gap:16px;align-items:center;flex-wrap:wrap}
main{display:flex;gap:24px;padding:16px;flex-wrap:wrap}
#stage{position:relative;background:#fff;padding:8px;border:1px solid #cfd8dc}
#stage img{height:480px;display:block;image-rendering:auto}
.box{position:absolute;border:3px solid;pointer-events:none;box-sizing:border-box}
#up{border-color:#f2d21b}#lo{border-color:#00bcd4}
.panel{max-width:420px}
button{font:inherit;padding:8px 10px;border:1px solid #90a4ae;border-radius:6px;background:#fff;cursor:pointer;margin:3px}
button.col{display:inline-flex;align-items:center;gap:8px;min-width:128px}
.sw{width:18px;height:18px;border:1px solid #666;border-radius:3px;display:inline-block}
button.on{outline:3px solid #1565c0}
#q{font-size:18px;font-weight:600;margin:6px 0 10px}
.small{color:#546e7a;font-size:13px}
</style></head><body>
<header><b>Colour labelling</b><span id="prog"></span>
<button onclick="dl()">Download labels</button><button onclick="back()">Back</button>
<span class="small">Judge the real garment colour, not the lighting. Shadows: pick the colour you believe the cloth is. Press u when you cannot tell.</span></header>
<main><div id="stage"><img id="im"><div class="box" id="up"></div><div class="box" id="lo"></div></div>
<div class="panel"><div id="q"></div><div id="btns"></div>
<div style="margin-top:8px"><button id="bd" onclick="shade('dark')">dark shade (d)</button><button id="bl" onclick="shade('light')">light shade (l)</button>
<button onclick="ans('unsure')">unsure (u)</button><button onclick="ans('skip')">skip (space)</button></div>
<p class="small" id="hint"></p></div></main>
<script>
const ITEMS=__ITEMS__, COLOURS=__COLOURS__, KEYS=__KEYS__, SW=__SW__, STORE="colour-labels-v__VER__-__SIG__";
let state=JSON.parse(localStorage.getItem(STORE)||'{"i":0,"labels":{}}'), step=0, pend=null;
function slots(it){return it.kind==="person"?[["upper","Upper body (yellow box): what colour is the top?"],["lower","Lower body (blue box): what colour are the trousers/skirt?"]]:[["color","Vehicle: what colour is the body?"]];}
function render(){
  const it=ITEMS[state.i]; if(!it){document.getElementById("q").textContent="All done. Click Download labels.";document.getElementById("btns").innerHTML="";return;}
  document.getElementById("im").src="data:image/jpeg;base64,"+it.jpeg;
  const person=it.kind==="person", s=slots(it)[step];
  const up=document.getElementById("up"), lo=document.getElementById("lo");
  up.style.display=person&&s[0]==="upper"?"block":"none"; lo.style.display=person&&s[0]==="lower"?"block":"none";
  const im=document.getElementById("im"); const place=()=>{const w=im.clientWidth,h=im.clientHeight;
    Object.assign(up.style,{left:8+w*.2+"px",top:8+h*.15+"px",width:w*.6+"px",height:h*.35+"px"});
    Object.assign(lo.style,{left:8+w*.2+"px",top:8+h*.5+"px",width:w*.6+"px",height:h*.4+"px"});}; im.onload=place; place();
  document.getElementById("q").textContent=s[1];
  document.getElementById("btns").innerHTML=COLOURS.map((c,k)=>`<button class="col" onclick="ans('${c}')"><span class="sw" style="background:${SW[c]}"></span>${c} (${KEYS[k]})</button>`).join("");
  document.getElementById("prog").textContent=`item ${state.i+1} of ${ITEMS.length}, labelled ${Object.keys(state.labels).length}`;
  document.getElementById("bd").classList.toggle("on",pend==="dark"); document.getElementById("bl").classList.toggle("on",pend==="light");
}
function shade(v){pend=pend===v?null:v;render();}
function ans(c){const it=ITEMS[state.i]; if(!it)return; const slot=slots(it)[step][0];
  const rec=state.labels[it.id]||(state.labels[it.id]={}); rec[slot]=c; if(pend&&c!=="unsure"&&c!=="skip")rec[slot+"_shade"]=pend; pend=null;
  step++; if(step>=slots(it).length){step=0;state.i++;} localStorage.setItem(STORE,JSON.stringify(state)); render();}
function back(){if(step>0){step--;}else if(state.i>0){state.i--;step=slots(ITEMS[state.i]).length-1;} localStorage.setItem(STORE,JSON.stringify(state)); render();}
function dl(){const out=ITEMS.map(it=>{const l=state.labels[it.id]||{};const o={id:it.id,camera:it.camera,cls:it.cls,kind:it.kind,
  pred_upper:it.pred_upper,pred_lower:it.pred_lower,pred_color:it.pred_color,pred_conf:it.pred_conf};
  for(const k of ["upper","lower","color"])if(l[k]!==undefined){o["label_"+k]=l[k];if(l[k+"_shade"])o["label_"+k+"_shade"]=l[k+"_shade"];} return o;});
  const blob=new Blob([JSON.stringify({tool_version:__VER__,workspace:"__WS__",items:out},null,1)],{type:"application/json"});
  const a=document.createElement("a");a.href=URL.createObjectURL(blob);a.download="labels.json";a.click();}
document.addEventListener("keydown",e=>{const k=e.key; const idx=KEYS.indexOf(k);
  if(idx>=0)ans(COLOURS[idx]); else if(k==="u")ans("unsure"); else if(k===" "||k==="Enter"){e.preventDefault();ans("skip");}
  else if(k==="Backspace")back(); else if(k==="d")shade("dark"); else if(k==="l")shade("light");});
render();
</script></body></html>"""


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workspace", default="meva-school")
    ap.add_argument("--persons", type=int, default=150)
    ap.add_argument("--vehicles", type=int, default=60)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", type=Path, default=Path("data/share/colour_label/index.html"))
    args = ap.parse_args()
    root = REPO / "workspaces" / args.workspace
    if not (root / "evora.sqlite").is_file():
        print(f"workspace not found: {root}", file=sys.stderr)
        return 1
    items = load_items(root, args.persons, args.vehicles, args.seed)
    if not items:
        print("no crops to label", file=sys.stderr)
        return 1
    sig = f"{args.workspace}-{args.seed}-{len(items)}"
    page = (PAGE.replace("__ITEMS__", json.dumps(items)).replace("__COLOURS__", json.dumps(COLOURS)).replace("__KEYS__", json.dumps(KEYS))
            .replace("__SW__", json.dumps(SWATCH)).replace("__VER__", str(TOOL_VERSION)).replace("__SIG__", sig)
            .replace("__WS__", args.workspace))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(page, encoding="utf-8")
    n_p = sum(1 for i in items if i["kind"] == "person")
    print(f"wrote {args.out} ({args.out.stat().st_size / 1e6:.1f} MB): {n_p} people (2 questions each) and {len(items) - n_p} vehicles")
    return 0


if __name__ == "__main__":
    sys.exit(main())
