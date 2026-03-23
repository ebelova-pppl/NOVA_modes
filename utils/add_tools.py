import numpy as np

def explain_by_zscores(x, X_train, feature_names, topk=8):
    mu = X_train.mean(axis=0)
    sig = X_train.std(axis=0) + 1e-12
    z = (x - mu) / sig
    idx = np.argsort(np.abs(z))[::-1][:topk]
    print("Top feature deviations (z-scores vs training mean):")
    for i in idx:
        print(f"{feature_names[i]:20s}  value={x[i]: .4e}  z={z[i]: .2f}")

def class_medians(X, y, feature_names):
    good = X[y==1]
    bad  = X[y==0]
    print("Feature           median(good)   median(bad)")
    for i, name in enumerate(feature_names):
        print(f"{name:16s} {np.median(good[:,i]): .4e}  {np.median(bad[:,i]): .4e}")
