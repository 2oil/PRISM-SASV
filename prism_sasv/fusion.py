"""3-way (H0/H1/H2) multinomial logistic regression fusion of the three branch scores."""
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

DEFAULT_COLS = ["s_utt_id", "s_pho_tgt", "s_pho_spf"]


def fit(df, cols=DEFAULT_COLS, label_col="hyp", max_iter=1000):
    """df must have one column per entry in `cols` plus `label_col` (values 0/1/2)."""
    scaler = StandardScaler()
    X = scaler.fit_transform(df[cols].values)
    clf = LogisticRegression(max_iter=max_iter).fit(X, df[label_col].values)
    return clf, scaler


def score(clf, scaler, df, cols=DEFAULT_COLS):
    """Returns P(H0 | features) -- the final SASV score."""
    X = scaler.transform(df[cols].values)
    return clf.predict_proba(X)[:, list(clf.classes_).index(0)]


def save(clf, scaler, cols, out_dir, name="fusion"):
    joblib.dump(clf, f"{out_dir}/{name}_clf.joblib")
    joblib.dump(scaler, f"{out_dir}/{name}_scaler.joblib")
    joblib.dump(cols, f"{out_dir}/{name}_cols.joblib")


def load(out_dir, name="fusion"):
    clf = joblib.load(f"{out_dir}/{name}_clf.joblib")
    scaler = joblib.load(f"{out_dir}/{name}_scaler.joblib")
    cols = joblib.load(f"{out_dir}/{name}_cols.joblib")
    return clf, scaler, cols
