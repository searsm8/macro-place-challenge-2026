"""One-shot script: find best valid proxy per benchmark across all sweeps."""
import torch
from pathlib import Path

sweep_root = Path(__file__).resolve().parent.parent / "sweep"
best = {}

for pt in sorted(sweep_root.rglob("proxy_score.pt")):
    try:
        d = torch.load(pt, weights_only=False)
    except Exception:
        continue
    if d.get("overlap_count", 999) != 0:
        continue
    proxy = d["proxy_cost"]
    parts = pt.parts
    # path: .../sweep/sweep_XXX/<bench>/run_NNN/frames/<bench>/proxy_score.pt
    # parts[-1]=proxy_score.pt, [-2]=bench, [-3]=frames, [-4]=run_NNN, [-5]=bench, [-6]=sweep_XXX
    bench = parts[-5]
    if bench not in best or proxy < best[bench]["proxy"]:
        best[bench] = {
            "proxy": proxy,
            "wl": d["wirelength_cost"],
            "den": d["density_cost"],
            "cong": d["congestion_cost"],
            "sweep": parts[-6],
            "run": parts[-4],
        }

for bench in sorted(best):
    b = best[bench]
    print(
        f"{bench:12s}  proxy={b['proxy']:.4f}  wl={b['wl']:.4f}"
        f"  den={b['den']:.4f}  cong={b['cong']:.4f}"
        f"  [{b['sweep']}/{b['run']}]"
    )

proxies = [b["proxy"] for b in best.values()]
print(
    f"\n{'':12s}  avg  ={sum(proxies)/len(proxies):.4f}"
    f"  min={min(proxies):.4f}  max={max(proxies):.4f}"
    f"  (n={len(proxies)})"
)
