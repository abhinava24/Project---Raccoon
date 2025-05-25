import React, { useState } from "react";
import axios from "axios";

// Group definitions (should match backend!)
const GROUPS = [
  { id: 0, name: "Lumpy skin disease" },
  { id: 1, name: "Income" },
  { id: 2, name: "Score" },
  { id: 3, name: "Smoking" }
];

// --- Defense: Differential Privacy noise (Laplace mechanism) ---
function addDpNoise(update, epsilon=1.0, sensitivity=1.0) {
  const scale = sensitivity / epsilon;
  // Laplace: sample from Lap(0, scale)
  return update.map(x => x + (Math.random() - 0.5) * 2 * scale);
}

// --- Defense: Masking (random vector) ---
function addMask(update, mask) {
  return update.map((x, i) => x + mask[i]);
}

const App = () => {
  // Group state
  const [userGroup, setUserGroup] = useState(null);

  // Client-side training state
  const [dataset, setDataset] = useState(null);
  const [permissionGranted, setPermissionGranted] = useState(false);
  const [trainingStatus, setTrainingStatus] = useState("Waiting for input...");
  const [log, setLog] = useState("");
  const [mlResults, setMlResults] = useState(null); // To display ML backend output

  // Group join
  const handleJoinGroup = (groupId) => {
    setUserGroup(groupId);
    setLog(`Joined training group ${GROUPS[groupId].name} (index ${groupId})`);
    axios.get(`/join_group/${groupId}`);
    setMlResults(null); // Reset ML results when changing group
  };

  // Dataset upload (kept in memory only)
  const handleDatasetUpload = (event) => {
    const file = event.target.files[0];
    if (file) {
      const reader = new FileReader();
      reader.onload = (e) => {
        setDataset(e.target.result); // CSV or text, for demo
        setLog("Dataset loaded in browser memory.");
      };
      reader.readAsText(file);
    }
  };

  // Grant permission for training
  const handleGrantPermission = () => {
    if (!dataset) {
      setLog("Upload dataset first.");
      return;
    }
    setPermissionGranted(true);
    setLog("Permission granted: ready to train.");
  };

  // Training & federated update
  const handleTrainModel = async () => {
    if (userGroup == null) {
      setLog("Join a group first.");
      return;
    }
    if (!dataset) {
      setLog("Please upload a dataset before training.");
      return;
    }
    if (!permissionGranted) {
      setLog("Please grant permission before training.");
      return;
    }

    setTrainingStatus("Training in progress...");
    setLog("Training locally...");

    try {
      // 1. Get current global weights
      const resp = await axios.get(`/get_global/${userGroup}`);
      const globalWeights = resp.data.weights;

      // 2. Simulate local training: create a dummy delta
      // For real ML, replace this with your own ML code
      const localUpdate = globalWeights.map(v => (Math.random() - 0.5) * 0.05);

      // --- Defense: Add DP noise ---
      const dpUpdate = addDpNoise(localUpdate, 1.0, 1.0);

      // --- Defense: Add random mask ---
      const mask = globalWeights.map(v => (Math.random() - 0.5) * 0.1);
      const update = addMask(dpUpdate, mask);

      // 3. Send masked/noised update to backend
      const postResp = await axios.post(`/submit_update/${userGroup}`, { update });

      setTrainingStatus("Training completed successfully!");
      setLog(postResp.data.msg || "Local training and federated update complete.");

      // 4. --- ML Integration: Call backend ML algorithm with chosen_idx ---
      // This will trigger centralized/federated training on the backend, using the selected group
      const mlResp = await axios.get(`/train/${userGroup}`);
      setMlResults(mlResp.data);

    } catch (error) {
      setTrainingStatus("Training failed. Please try again.");
      setLog("Error during training: " + (error.response?.data?.msg || error.message));
    }
  };

  return (
    <div style={{ padding: 20 }}>
      <h1>Decentralized Training Platform</h1>
      <h3>Select a Training Group</h3>
      <div>
        {GROUPS.map((g) => (
          <button key={g.id} onClick={() => handleJoinGroup(g.id)}>
            {g.name} (index {g.id})
          </button>
        ))}
      </div>
      <div>
        <h3>Upload Your Dataset</h3>
        <input type="file" onChange={handleDatasetUpload} />
      </div>
      <div>
        <h3>Actions</h3>
        <button onClick={handleGrantPermission}>Grant Permission</button>
        <button onClick={handleTrainModel}>Start Training</button>
      </div>
      <p><strong>Status:</strong> {trainingStatus}</p>
      <pre>{log}</pre>
      {/* Show ML results if any */}
      {mlResults && (
        <div style={{marginTop: 20}}>
          <h3>Backend ML Results (for group {userGroup}):</h3>
          <pre>{JSON.stringify(mlResults, null, 2)}</pre>
        </div>
      )}
    </div>
  );
};

export default App;
