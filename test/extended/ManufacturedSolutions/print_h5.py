import h5py

def print_structure(name, obj):
    print(name)
    if isinstance(obj, h5py.Dataset):
        print("  ", obj[:])

f = h5py.File("results/convergence_data.h5", "r")
f.visititems(print_structure)
f.close()
