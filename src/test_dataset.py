from utils import load_dataset

paths, labels = load_dataset()

print(len(paths))

print(labels.count(0))

print(labels.count(1))