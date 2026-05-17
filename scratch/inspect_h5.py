import h5py
import os

path = "test/extended/ManufacturedSolutions/results/convergence_data.h5"
if not os.path.exists(path):
    print(f"File {path} not found")
    exit(1)

f = h5py.File(path, "r")
groups = sorted(list(f.keys()), key=lambda x: int(x.split('_')[1]) if x.startswith('config_') else 0)

for gname in groups[-5:]: # show last 5
    g = f[gname]
    print(f"Group: {gname}")
    for attr in g.attrs:
        print(f"  {attr}: {g.attrs[attr]}")
    if 'h' in g:
        print(f"  h: {g['h'][:]}")
f.close()
