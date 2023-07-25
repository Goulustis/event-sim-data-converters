import pandas as pd

df = pd.read_csv("poses_all.txt", sep=" ", header=None, skiprows=[0])
df.columns = ["time", "x", "y", "z", "qx", "qy", "qz", "qw"]
df.to_csv("camera_trajectory.csv", index=False)

