import pickle

with open("./test_label_constraints.pkl", "rb") as f:
    label_constraints = pickle.load(f)

print(type(label_constraints))
