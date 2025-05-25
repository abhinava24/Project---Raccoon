import React, { useState } from "react";
import axios from "axios";

const GROUPS = [
  { id: 0, name: "Lumpy skin disease" },
  { id: 1, name: "Income" },
  { id: 2, name: "Score" },
  { id: 3, name: "Smoking" }
];

function addDpNoise(update, epsilon=1.0, sensitivity=1.0) {
  const scale = sensitivity / epsilon;
  return update.map(x => x + (Math.random() - 0.5) * 2 * scale);
}

function addMask(update, mask) {
  return update.map((x, i) => x + mask[i]);
}

const App = () => {
  const [userGroup, setUserGroup] = useState(null);
  const [dataset, setDataset] = useState(null);
  const [permissionGranted, setPermissionGranted] = useState(false);
  const [trainingStatus, setTrainingStatus] = useState("Waiting for input...");
  const [log, setLog] = useState("");

  const handleJoinGroup = (groupId) => {
    setUserGroup(groupId);
    setLog(`Joined training group ${GROUPS[groupId].name} (index ${groupId})`);
    axios.get(`http://localhost:5000/join_group/${groupId}`)
      .catch(e => setLog("Error joining group: " + e.message));
  };

  const handleDatasetUpload = (event) => {
    const file = event.target.files[0];
    if (file) {
      const reader = new FileReader();
      reader.onload = (e) => {
        setDataset(e.target.result);
        setLog("Dataset loaded in browser memory.");
      };
      reader.readAsText(file);
    }
  };

  const handleGrantPermission = () => {
    if (!dataset) {
      setLog("Upload dataset first.");
      return;
    }
    setPermissionGranted(true);
    setLog("Permission granted: ready to train.");
  };

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
      const resp = await axios.get(`http://localhost:5000/get_global/${userGroup}`);
      console.log("Global weights response:", resp);
      console.log("Global weights data:", resp.data);

      const globalWeights = resp.data.weights;
      if (!globalWeights) {
        throw new Error("No weights received from server");
      }
      console.log("Global weights:", globalWeights);

      const localUpdate = globalWeights.map(() => (Math.random() - 0.5) * 0.05);

      const dpUpdate = addDpNoise(localUpdate, 1.0, 1.0);

      const mask = globalWeights.map(() => (Math.random() - 0.5) * 0.1);
      const update = addMask(dpUpdate, mask);

      const postResp = await axios.post(`http://localhost:5000/submit_update/${userGroup}`, { update });
      setTrainingStatus("Training completed successfully!");
      setLog(postResp.data.msg || "Local training and federated update complete.");
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
    </div>
  );
};

export default App;
