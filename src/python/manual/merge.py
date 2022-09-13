import sys
import pandas as pd
import numpy as np

df1 = pd.read_csv(sys.argv[1])
df2 = pd.read_csv(sys.argv[2])

res = pd.concat([df1, df2], axis=0).reset_index(drop=True)
res = res.drop(["index"], axis=1)
res.to_csv(f"{sys.argv[7]}.csv")

npy1 = np.load(sys.argv[3])
npy2 = np.load(sys.argv[4])

npy1_mask = np.load(sys.argv[5])
npy2_mask = np.load(sys.argv[6])

npy = np.concatenate([npy1, npy2], axis=0)
np.save(f"{sys.argv[7]}_tinycells.npy", npy)

npy = np.concatenate([npy1_mask, npy2_mask], axis=0)
np.save(f"{sys.argv[7]}_tinycells_mask.npy", npy)

