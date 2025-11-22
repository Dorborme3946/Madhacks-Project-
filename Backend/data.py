
import joblib
import numpy as np

# Load classical model + scaler
classical_model = joblib.load("models/classical_model.pkl")
scaler = joblib.load("models/scaler.pkl")

# Load quantum model
# This may be VQC or saved optimal parameters
quantum_model = joblib.load("models/quantum_model.pkl")


def classical_predict(features):
    """
    features: numpy array of shape (1, n_features)
    """
    scaled = scaler.transform(features)
    prob = classical_model.predict_proba(scaled)[0][1]
    return prob


def quantum_predict(q_features):
    """
    q_features: numpy array of reduced features for quantum model
    """
    # Quantum classifier returns label or probability depending on how you saved it
    prob = quantum_model.predict(q_features)[0]  # returns 0/1 or prob
    return float(prob)
    

def hybrid_risk(classical_prob, quantum_prob):
    # Weighted combination
    a = 0.6
    return a * classical_prob + (1 - a) * quantum_prob
