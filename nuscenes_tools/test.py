import pickle

p = "/mnt/z/dataset/nuscenes/nuscenes_infos_train.pkl"

with open(p, "rb") as f:
    x = pickle.load(f)

print(type(x))
if isinstance(x, dict):
    print(x.keys())