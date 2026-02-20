import pandas as pd
import numpy as np

df = pd.read_csv("fraudTest.csv")

columns_to_drop = ["merchant", "gender", "first", "last", "street", "cc_num"]
df_cleaned = df.drop(columns=columns_to_drop)

X = df_cleaned.to_numpy()

# speichern
np.save("fraudTest_cleaned.npy", X)

# später laden
# X_loaded = np.load("fraudTest_cleaned.npy", allow_pickle=True)