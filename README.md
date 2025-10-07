🧠 AI-Powered Predictive Maintenance in Manufacturing
Deep Learning–Based Remaining Useful Life (RUL) Prediction using NASA C-MAPSS Dataset
📘 Overview

This project implements an AI-driven Predictive Maintenance system for industrial machines, developed as part of a Final-Year B.Tech project.

The goal is to forecast machinery failures before they occur by analyzing IoT sensor data using deep learning models such as LSTM, GRU, TCN, and Transformer architectures.

The system predicts the Remaining Useful Life (RUL) of components and provides real-time monitoring insights that help reduce downtime, optimize maintenance schedules, and increase safety in manufacturing environments.

🚀 Key Features

✅ Multi-Model Deep Learning Framework

Implements and compares LSTM, GRU, TCN, and Transformer models for RUL prediction.

Automatically saves and loads trained models across datasets (FD001–FD004).

✅ Runtime Accuracy Tracker

Tracks and plots Training vs Validation Loss & MAE for every model and dataset.

Saves visual learning curves in the results/ folder.

✅ Evaluation Metrics & Benchmarking

Calculates MAE, RMSE, and R² for each model.

Automatically logs results in a consolidated CSV report.

✅ Real-World Dataset (NASA C-MAPSS)

Uses turbofan engine sensor data for training and testing.

Supports all four benchmark subsets (FD001–FD004).

✅ Automated Visualization

Generates graphs for:

Actual vs Predicted RUL

Model Training Curves (Loss + MAE)

🧩 Project Structure
predictive_maintenance/
│
├── data/                # NASA C-MAPSS dataset (FD001–FD004)
├── saved_models/        # Trained models (.keras format)
├── results/             # Accuracy plots, RUL graphs, comparison CSV
├── main.py              # Core ML training & evaluation pipeline
└── README.md            # Project documentation

📊 Current Progress
Phase	Description	Status
1	Environment & Dataset Setup	✅ Done
2	Deep Learning Model Development	✅ Done
3	Model Evaluation + Accuracy Tracker	✅ Done
4	Deployment (Flask/Frontend Integration)	🔜 In Progress
5	Documentation & Report	🔜 Pending

📈 Completion Status: ~75%

🧠 Models Implemented
Model	Description	Notes
LSTM	Long Short-Term Memory	Captures long-term dependencies in sensor data
GRU	Gated Recurrent Unit	Simplified version of LSTM with faster training
TCN	Temporal Convolutional Network	Sequence modeling using dilated convolutions
Transformer	Attention-based architecture	Models long-range relationships efficiently
⚙️ Tech Stack

Languages & Frameworks:

Python, TensorFlow/Keras, scikit-learn, NumPy, pandas, matplotlib

Dataset:

NASA C-MAPSS Turbofan Engine Degradation Dataset

Planned Deployment:

Flask / FastAPI (backend)

React.js / Streamlit (frontend dashboard)

💻 How to Run
1️⃣ Clone the Repository
git clone https://github.com/VaibhavSaxena10/predictive_maintenance.git
cd predictive_maintenance

2️⃣ Install Dependencies

Make sure Python (≥3.10) and pip are installed.
Then install required libraries:

pip install -r requirements.txt


💡 If you don’t have a requirements.txt yet, you can create one using:
pip freeze > requirements.txt

3️⃣ Add the Dataset

Download the NASA C-MAPSS dataset and place all files inside the /data folder:

data/
 ├── train_FD001.txt
 ├── train_FD002.txt
 ├── train_FD003.txt
 ├── train_FD004.txt
 ├── test_FD001.txt
 └── ...

4️⃣ Run the Project

Execute the main script:

python main.py


This will:

Preprocess data

Train models (LSTM, GRU, TCN, Transformer)

Evaluate them using MAE, RMSE, and R²

Save comparison results and accuracy graphs in /results/

5️⃣ View Results

After training completes, check:

results/
 ├── FD001/
 │    ├── LSTM_accuracy_curve_FD001.png
 │    ├── GRU_accuracy_curve_FD001.png
 │    ├── ...
 ├── all_datasets_comparison.csv

📁 Outputs

Model Performance CSV: results/all_datasets_comparison.csv

Training Curves: Loss & MAE plots for all models

Actual vs Predicted RUL: Visual comparison for each dataset

🧭 Next Steps

 Add Flask backend for live RUL prediction

 Create simulated IoT data streams

 Build web dashboard for real-time monitoring

 Integrate Explainable AI (SHAP / Grad-CAM)

✍️ Authors & Mentorship

Student: Vaibhav Saxena
Mentor: Ms. Rashika Bangroo

🏁 Project Status

Current Completion: ~75%
Next Phase: Backend deployment using Flask + real-time monitoring interface