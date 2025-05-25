import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score

import os

# File paths (update if your data location is different)
file1 = 'Datasets/Lumpy skin disease data.csv'
file2 = 'Datasets/income.csv'
file3 = 'Datasets/score.csv'
file4 = 'Datasets/smoking.csv'

# Load datasets at module level for re-use
df1 = pd.read_csv(file1)
df2 = pd.read_csv(file2)
df3 = pd.read_csv(file3)
df4 = pd.read_csv(file4)

# Make sure to set correct target column names for your datasets
target1 = 'target1'  # REPLACE with actual column name in Lumpy skin disease data
target2 = 'target2'  # REPLACE with actual column name in income.csv
target3 = 'target3'  # REPLACE with actual column name in score.csv
target4 = 'target4'  # REPLACE with actual column name in smoking.csv

# For easy indexing
dfs = [df1, df2, df3, df4]
target_columns = [target1, target2, target3, target4]

def preprocess_data(df, targetcol):
    x = df.drop(columns=[targetcol])
    y = df[targetcol]

    if y.dtypes == 'object' or y.dtypes.name == 'category':
        new = LabelEncoder()
        y = new.fit_transform(y)

    x_temp, x_test, y_temp, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    x_train, x_val, y_train, y_val = train_test_split(x_temp, y_temp, test_size=0.25, random_state=42)
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_val = scaler.transform(x_val)
    x_test = scaler.transform(x_test)
    subsets = [{"x":x_train, "y":y_train}, {"x":x_val, "y":y_val}, {"x":x_test, "y":y_test}]
    return subsets

class MLmodel(nn.Module):
    def __init__(self, input_dim,hidden_dim, output_dim):
        super(MLmodel, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

def train_localmodel(nodeid,subset,input_dim,output_dim,epochs=50,lr=0.001,hidden_dim=64):
    model = MLmodel(input_dim,hidden_dim, output_dim)
    criteria = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    x_tensor = torch.tensor(subset["x"], dtype=torch.float32)
    y_tensor = torch.tensor(subset["y"], dtype=torch.long)

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        ans = model(x_tensor)
        loss = criteria(ans,y_tensor)
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        model.eval()
    return model

def aggregate_weights(localset):
    globalset= {}
    for key in localset[0].keys():
        globalset[key] = sum(d[key] for d in localset) / len(localset)
    return globalset

def evaluate_model(model, subset):
    model.eval()
    x_tensor = torch.tensor(subset["x"], dtype=torch.float32)
    y_true = subset["y"]
    with torch.no_grad():
        results = model(x_tensor)
    _,y_pred = torch.max(results, 1)
    acc = accuracy_score(y_true, y_pred.numpy())
    return acc

# Main entrypoint for integration: run one round of centralized and federated training for a given chosen_idx
def train_and_evaluate(chosen_idx):
    df = dfs[chosen_idx]
    targetcol = target_columns[chosen_idx]

    # ------- Centralized Training (whole dataset) -------
    central_subsets = preprocess_data(df, targetcol)
    input_dim = central_subsets[0]["x"].shape[1]
    output_dim = len(np.unique(central_subsets[0]["y"]))
    central_model = train_localmodel(0, central_subsets[0], input_dim, output_dim, epochs=50, lr=0.001, hidden_dim=64)
    central_acc = evaluate_model(central_model, central_subsets[2])

    # ------- Federated/Decentralized Training (3 subsets) -------
    idxs = np.random.permutation(len(df))
    split_idxs = np.array_split(idxs, 3)
    local_models = []
    test_x, test_y = [], []

    for i, idx_set in enumerate(split_idxs):
        df_sub = df.iloc[idx_set].reset_index(drop=True)
        subsets = preprocess_data(df_sub, targetcol)
        input_dim = subsets[0]["x"].shape[1]
        output_dim = len(np.unique(subsets[0]["y"]))

        # Start from scratch for local models
        trained_local_model = train_localmodel(
            i, subsets[0], input_dim, output_dim, epochs=50, lr=0.001, hidden_dim=64
        )
        local_models.append(trained_local_model.state_dict())
        test_x.append(subsets[2]["x"])
        test_y.append(subsets[2]["y"])

    # --- Aggregate local models for this dataset ---
    agg_weights = aggregate_weights(local_models)
    fed_model = MLmodel(input_dim, 64, output_dim)
    fed_model.load_state_dict(agg_weights)

    # --- Evaluate the updated global model for this dataset ---
    x_test: np.ndarray  = np.concatenate(test_x)
    y_test: np.ndarray = np.concatenate(test_y)
    fed_test_acc = evaluate_model(fed_model, {"x": x_test, "y": y_test})

    # Return metrics and optionally models if needed
    return {
        "centralized_accuracy": float(central_acc),
        "federated_accuracy": float(fed_test_acc)
        # Optionally, add more outputs (model weights, etc.)
    }