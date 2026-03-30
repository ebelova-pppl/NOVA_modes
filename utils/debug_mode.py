import joblib
import numpy as np
from nova_mode_loader import load_mode_from_nova
from mode_features import compute_features_for_mode
from add_tools import explain_by_zscores, class_medians

bundle = joblib.load("nova_mode_classifier_bundle.joblib")
clf = bundle["model"]
X_train = bundle["X_train"]
y_train = bundle["y_train"]
feature_names = bundle["feature_names"]

# Load the suspicious mode
path = "/global/cfs/cdirs/m314/nova/nstx_141711/N5/egn05w.1251E+03"
mode, omega, gamma_d, ntor = load_mode_from_nova(path)

x = compute_features_for_mode(
    mode,
    extra_info={"omega": omega, "gamma_d": gamma_d, "path":path}
)

print("Prediction:")
print("P(good) =", clf.predict_proba(x.reshape(1,-1))[0,1])

print("\nZ-score explanation:")
explain_by_zscores(x, X_train, feature_names)
print('')

#print(feature_names)

print("Features for this mode")
for i, name in enumerate(feature_names):
    print(f"{feature_names[i]:20s}  value={x[i]: .4e}")
print('')

class_medians(X_train, y_train, feature_names)
