import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import joblib

base = Path(__file__).resolve().parent
data_file = base / "data.csv"
proc_file = base / "data" / "processed" / "processed.csv"
model_file = base / "model.pkl"
MAX_ROWS = 20000
features = ["age","sex","smoking_status","pd_l1","tmb","kras_mutated","egfr_mutated"]
target = "benefit"
_model = None

def make_frame(df):
    frame = pd.DataFrame()
    frame["age"] = df["patient_age"]
    frame["sex"] = df["patient_gender"]
    frame["smoking_status"] = df["smoking_history"]
    frame["pd_l1"] = df["PD-L1_expression_level"]
    frame["tmb"] = df["tumor_mutational_burden"]
    frame["kras_mutated"] = df["KRAS_mutation_status"]
    frame["egfr_mutated"] = df["EGFR_mutation_status"]
    frame["benefit"] = ((df["survival_time_months"] >= 18) & (df["immunotherapy_received"] == 1)).astype(int)
    return frame

def encode(df):
    df = df.copy()
    df["sex"] = df["sex"].astype(str).str.lower().map({"male":0,"female":1})
    df["smoking_status"] = df["smoking_status"].astype(str).str.lower().map({"never":0,"former":1,"current":2})
    df["kras_mutated"] = df["kras_mutated"].astype(float)
    df["egfr_mutated"] = df["egfr_mutated"].astype(float)
    return df

def preprocess():
    if not data_file.exists():
        raise FileNotFoundError("missing data.csv")
    df = pd.read_csv(data_file)
    df = make_frame(df)
    df = df.dropna()
    df = encode(df)
    if len(df) > MAX_ROWS:
        df = df.sample(MAX_ROWS, random_state=42)
    proc_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(proc_file, index=False)
    return df

def train():
    df = preprocess()
    X = df[features]
    y = df[target]
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    joblib.dump(model, model_file, compress=3)
    return model

def load_model():
    global _model
    if _model is None:
        if not model_file.exists():
            raise FileNotFoundError("model.pkl missing. run python model.py locally first")
        _model = joblib.load(model_file)
    return _model

def predict_patient(data):
    model = load_model()
    row = {f: data.get(f, 0) for f in features}
    frame = pd.DataFrame([row])
    proba = model.predict_proba(frame)[0][1]
    return float(proba)

if __name__ == "__main__":
    train()
