import uproot
import numpy as np
import pandas as pd
import os

# ====== CONFIG ======
INPUT_FILE = "/data2/segmentlinking/CMSSW_12_2_0_pre2/trackingNtuple_10mu_pt_0p5_50.root"
OUTPUT_CSV = "filtered_particles.csv"
OUTPUT_DIR = "outputCSVs"

ETACUT = 0.9
PTCUT = 1.9
HITTYPE = 4
MUON_ID = 13

N_LAYERS = 6
SENTINEL = -999.0

# ====== LOAD DATA ======
branches = [
    "sim_q","sim_pt","sim_pdgId","sim_eta","sim_simHitIdx",
    "sim_pca_pt","sim_pca_eta","sim_pca_phi","sim_pca_dxy","sim_pca_dz",
    "simhit_x","simhit_y","simhit_z","simhit_hitType","simhit_isLower"
]

with uproot.open(INPUT_FILE) as f:
    tree = f["trackingNtuple/tree"]
    data = {b: tree[b].array(library="np") for b in branches}
    n_events = tree.num_entries

# ====== HELPER ======
def barrel_layer(x, y):
    r = np.sqrt(x**2 + y**2)
    if r < 30: return 0
    elif r < 45: return 1
    elif r < 60: return 2
    elif r < 80: return 3
    elif r < 100: return 4
    elif r < 120: return 5
    else: return -1

# ====== STORAGE ======
rows = []

# ====== MAIN LOOP ======
for evt in range(n_events):
    n_particles = len(data["sim_pdgId"][evt])

    for i in range(n_particles):
        eta = data["sim_eta"][evt][i]
        pt = data["sim_pt"][evt][i]
        pid = data["sim_pdgId"][evt][i]
        q = data["sim_q"][evt][i]

        # --- FILTER!!!! ---
        if abs(eta) > ETACUT or pt < PTCUT or abs(pid) != MUON_ID or q == 0:
            continue

        # --- TARGETS ---
        q = data["sim_q"][evt][i]
        pca_pt = data["sim_pca_pt"][evt][i]

        row = {
            "pca_c": q / pca_pt,
            "pca_eta": data["sim_pca_eta"][evt][i],
            "pca_phi": data["sim_pca_phi"][evt][i],
            "pca_dxy": data["sim_pca_dxy"][evt][i],
            "pca_dz": data["sim_pca_dz"][evt][i],
        }

        # --- INIT HIT STORAGE ---
        hit_x = np.full(N_LAYERS, SENTINEL)
        hit_y = np.full(N_LAYERS, SENTINEL)
        hit_z = np.full(N_LAYERS, SENTINEL)
        mask = np.zeros(N_LAYERS)
        seen = np.zeros(N_LAYERS, dtype=bool)

        # --- GET HITS ---
        hit_indices = data["sim_simHitIdx"][evt][i]

        for h in hit_indices:
            hit_type_arr = data["simhit_hitType"][evt][h]
            hit_type = hit_type_arr[0] if len(hit_type_arr) > 0 else -1

            if hit_type != HITTYPE:
                continue
            if data["simhit_isLower"][evt][h] != 1:
                continue

            x = data["simhit_x"][evt][h]
            y = data["simhit_y"][evt][h]
            z = data["simhit_z"][evt][h]

            layer = barrel_layer(x, y)

            if 0 <= layer < N_LAYERS and not seen[layer]:
                hit_x[layer] = x
                hit_y[layer] = y
                hit_z[layer] = z
                mask[layer] = 1
                seen[layer] = True

            if np.all(seen):
                break

        # --- ADD HITS TO ROW ---
        for j in range(N_LAYERS):
            row[f"hit_{j+1}_x"] = hit_x[j]
            row[f"hit_{j+1}_y"] = hit_y[j]
            row[f"hit_{j+1}_z"] = hit_z[j]
            row[f"hit_{j+1}_r"] = np.sqrt(hit_x[j]**2 + hit_y[j]**2)
            row[f"hit_{j+1}_mask"] = mask[j]

        rows.append(row)

# ====== SAVE CSV ======
df = pd.DataFrame(rows)

os.makedirs(OUTPUT_DIR, exist_ok=True)  # create directory if it doesn't exist

output_path = os.path.join(OUTPUT_DIR, OUTPUT_CSV)

df = pd.DataFrame(rows)
df.to_csv(output_path, index=False)
print(f"Saved {len(df)} rows to {output_path}")
