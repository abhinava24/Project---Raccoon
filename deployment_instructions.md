# Federated Learning Project

This repository implements a federated learning system with robust aggregation and ML backend, featuring a React frontend and a Flask backend. Follow the instructions below to set up and run the project end-to-end.

---

## Table of Contents

- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Backend Setup (Flask)](#backend-setup-flask)
- [Frontend Setup (React)](#frontend-setup-react)
- [Running the Full Stack](#running-the-full-stack)
- [Usage](#usage)
- [Troubleshooting](#troubleshooting)

---

## Project Structure

```
my-federated-project/
│
├── backend/
│   ├── app.py
│   ├── ml_algorithm.py
│   ├── defensesystem.py
│   └── datasets/
│       ├── Lumpy skin disease data.csv
│       ├── income.csv
│       ├── score.csv
│       └── smoking.csv
│
└── frontend/
    ├── package.json
    └── src/
        ├── App.js
        └── defensesystem.js
```

---

## Prerequisites

- **Python 3.8+** (for backend)
- **Node.js 14+** (for frontend; includes npm and npx)
- **pip** (Python package manager)

---

## Backend Setup (Flask)

1. **Create and activate a virtual environment (recommended):**
   ```bash
   cd my-federated-project/backend
   python3 -m venv venv
   # On Linux/Mac:
   source venv/bin/activate
   # On Windows:
   venv\Scripts\activate
   ```

2. **Install required Python packages:**
   ```bash
   pip install flask flask-cors numpy pandas scikit-learn torch matplotlib seaborn
   ```

3. **Ensure your datasets are present in `backend/datasets/`**  
   (If you don't have the CSVs, put placeholder files with the same names.)

4. **Run the backend:**
   ```bash
   python app.py
   ```
   - The backend will start on `http://localhost:5000` by default.

---

## Frontend Setup (React)

1. **Open a new terminal.**

2. **Navigate to the frontend directory:**
   ```bash
   cd my-federated-project/frontend
   ```

3. **If you haven't already initialized React, do so:**
   ```bash
   npx create-react-app .
   ```

4. **Install dependencies (e.g. Axios):**
   ```bash
   npm install axios
   ```

5. **Set up proxy for API calls in your `package.json`:**
   ```json
   "proxy": "http://localhost:5000",
   ```
   Add the above line in `frontend/package.json` (usually after the "private" field).

6. **Place your `App.js` and `defensesystem.js` in `frontend/src/`.**

7. **Run the frontend:**
   ```bash
   npm start
   ```
   - The frontend will start on `http://localhost:3000`.

---

## Running the Full Stack

- **Start the backend first** (`python app.py` in `backend/`).
- **Then start the frontend** (`npm start` in `frontend/`).
- **Open** [http://localhost:3000](http://localhost:3000) in your browser to use the platform.

---

## Usage

1. **Select a training group** on the React UI.
2. **Upload your dataset** (for demo, this can be any CSV).
3. **Grant permission** for local training.
4. **Click “Start Training”** to:
    - Simulate local training and submit a federated update.
    - Trigger backend ML code for the selected dataset (chosen_idx).
    - See backend ML results (accuracy, etc.) in the UI.

---
