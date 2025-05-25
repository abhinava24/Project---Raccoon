import secrets
from flask import Flask, request, jsonify
from flask_cors import CORS
import time
from collections import defaultdict
import numpy as np

# --- ML Integration import ---
from ml_algorithm import train_and_evaluate  # <-- Make sure ml_algorithm.py is in the same directory or in PYTHONPATH

app = Flask(__name__)
app.secret_key = secrets.token_hex(32)

CORS(app, supports_credentials=True)

GROUPS = [
    {"id": 0, "name": "Lumpy skin disease"},
    {"id": 1, "name": "Income"},
    {"id": 2, "name": "Score"},
    {"id": 3, "name": "Smoking"},
]
GLOBAL_MODELS = {i: [0.0]*10 for i in range(4)}  # Example "weights" vector
# For robust aggregation, store pending updates
PENDING_UPDATES = {i: [] for i in range(4)}
# For rate limiting
client_update_times = defaultdict(list)

def is_anomalous(update, reference, threshold=5.0):
    diff = np.linalg.norm(np.array(update) - np.array(reference))
    return diff > threshold

def trimmed_mean(updates, trim_ratio=0.1):
    arr = np.array(updates)
    n_clients = arr.shape[0]
    n_trim = int(trim_ratio * n_clients)
    sorted_arr = np.sort(arr, axis=0)
    trimmed = sorted_arr[n_trim:n_clients-n_trim]
    return np.mean(trimmed, axis=0)

@app.route('/join_group/<int:group_id>', methods=['GET'])
def join_group(group_id):
    # No-op: just acknowledge
    return jsonify({"msg": f"Joined group {group_id}"}), 200

@app.route('/get_global/<int:group_id>', methods=['GET'])
def get_global(group_id):
    return jsonify({"weights": GLOBAL_MODELS[group_id]})

@app.route('/submit_update/<int:group_id>', methods=['POST'])
def submit_update(group_id):
    # --- Sybil/Spam defense: Rate limiting by IP ---
    client_ip = request.remote_addr
    now = time.time()
    times = client_update_times[client_ip]
    client_update_times[client_ip] = [t for t in times if now - t < 60]
    if len(client_update_times[client_ip]) >= 1:
        return jsonify({"msg": "Rate limited"}), 429
    client_update_times[client_ip].append(now)

    data = request.get_json()
    update = data.get('update')
    if not isinstance(update, list):
        return jsonify({"msg": "Update must be a list"}), 400
    if len(update) != len(GLOBAL_MODELS[group_id]):
        return jsonify({"msg": "Update shape mismatch"}), 400

    # --- Anomaly detection: reject updates too far from global model ---
    if is_anomalous(update, GLOBAL_MODELS[group_id], threshold=5.0):
        return jsonify({"msg": "Anomalous update rejected"}), 400

    # --- Robust aggregation: Gather updates, apply trimmed mean once enough are collected ---
    PENDING_UPDATES[group_id].append(update)
    N_REQUIRED = 3  # Number of updates to aggregate
    if len(PENDING_UPDATES[group_id]) >= N_REQUIRED:
        agg_update = trimmed_mean(PENDING_UPDATES[group_id], trim_ratio=0.2)
        GLOBAL_MODELS[group_id] = (np.array(GLOBAL_MODELS[group_id]) + agg_update).tolist()
        PENDING_UPDATES[group_id] = []  # Reset buffer
        return jsonify({"msg": f"Aggregated {N_REQUIRED} updates with trimmed mean"}), 200
    else:
        return jsonify({"msg": f"Update accepted (waiting for {N_REQUIRED-len(PENDING_UPDATES[group_id])} more before aggregation)"}), 200

# --- ML Algorithm integration endpoint ---
@app.route("/train/<int:chosen_idx>", methods=["GET"])
def train(chosen_idx):
    # chosen_idx comes from the URL, provided by the frontend/app.js
    results = train_and_evaluate(chosen_idx)
    return jsonify(results)

if __name__ == '__main__':
    app.run(debug=True)
