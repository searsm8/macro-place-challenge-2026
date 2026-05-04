import csv, collections

rows = []
with open('sweep/sweep_20260502T002144Z/results.csv') as f:
    reader = csv.DictReader(f)
    for row in reader:
        rows.append(row)

print("Total rows:", len(rows))
benchmarks = sorted(set(r['benchmark'] for r in rows))
print("Benchmarks (%d):" % len(benchmarks), benchmarks)

invalid = [r for r in rows if r['valid'] != 'VALID']
print("Invalid runs:", len(invalid))

# Combo params
combo_params = {}
for r in rows:
    cid = int(r['combo_id'])
    td = float(r['sweep_target_density'])
    gs = int(r['sweep_density_grid_size'])
    if cid not in combo_params:
        combo_params[cid] = (td, gs)
print("\nCombo params (id -> target_density, grid_size):")
for cid in sorted(combo_params):
    td, gs = combo_params[cid]
    print("  %2d: td=%.2f, gs=%d" % (cid, td, gs))

# Per-combo avg proxy across all benchmarks
print("\n--- Avg proxy per combo (across all benchmarks) ---")
combo_proxies = collections.defaultdict(list)
for r in rows:
    combo_proxies[int(r['combo_id'])].append(float(r['proxy']))

results = []
for cid in sorted(combo_proxies):
    vals = combo_proxies[cid]
    avg = sum(vals) / len(vals)
    td, gs = combo_params[cid]
    results.append((avg, cid, td, gs, len(vals)))

results.sort()
print("rank  avg_proxy  n  td    gs   combo_id")
for rank, (avg, cid, td, gs, n) in enumerate(results[:16], 1):
    print("  %2d  %.4f    %2d  %.2f  %3d  %d" % (rank, avg, n, td, gs, cid))

# Best per target_density (averaged over grid sizes and benchmarks)
print("\n--- Avg proxy by target_density (avg over grid sizes) ---")
td_proxies = collections.defaultdict(list)
for r in rows:
    td_proxies[float(r['sweep_target_density'])].append(float(r['proxy']))
for td in sorted(td_proxies):
    vals = td_proxies[td]
    print("  td=%.2f  avg=%.4f  n=%d" % (td, sum(vals)/len(vals), len(vals)))

# Best per grid_size
print("\n--- Avg proxy by grid_size ---")
gs_proxies = collections.defaultdict(list)
for r in rows:
    gs_proxies[int(r['sweep_density_grid_size'])].append(float(r['proxy']))
for gs in sorted(gs_proxies):
    vals = gs_proxies[gs]
    print("  gs=%3d  avg=%.4f  n=%d" % (gs, sum(vals)/len(vals), len(vals)))

# ibm01 breakdown
print("\n--- ibm01 by combo ---")
ibm01 = [(float(r['proxy']), int(r['combo_id']), float(r['sweep_target_density']), int(r['sweep_density_grid_size']))
         for r in rows if r['benchmark'] == 'ibm01']
ibm01.sort()
print("proxy   td    gs   cid")
for proxy, cid, td, gs in ibm01[:10]:
    print("  %.4f  %.2f  %3d  %d" % (proxy, td, gs, cid))
