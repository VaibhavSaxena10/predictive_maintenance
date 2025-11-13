OptiMendix: AI-Powered Predictive Maintenance Dashboard
Deep Learning–Based Remaining Useful Life (RUL) Prediction using NASA C-MAPSS Data

🚦 Overview
OptiMendix is an end-to-end AI-powered predictive maintenance solution for industrial equipment. Built for clarity, extensibility, and visual polish, this dashboard lets users predict Remaining Useful Life (RUL) and health states using robust deep learning models (LSTM, GRU, Transformer) trained on real-world NASA C‑MAPSS sensor data.

Author: Vaibhav Saxena

Mentor: Ms. Rashika Bangroo

Status: ~100% complete

🚀 Key Features
Multi-Model Deep Learning: Choose and compare LSTM, GRU, TCN, and Transformer architectures for RUL prediction.

Modular Pipeline: Trains and benchmarks on all NASA FD001–FD004 datasets. Models auto-save/load for live prediction.

Interactive Dashboard:

Card-style input panel with animated pill selectors and theme toggle (dark/light).

Modern, mobile-first frontend using Flask/Jinja, HTML/CSS/FontAwesome.

Hover effects and border glow for maximum clarity and user delight.

Evaluation Metrics:

Tracks and compares MAE, RMSE, R² across all models and datasets.

Visual learning curves and training stats saved to /results/.

Automated Visualizations:

Actual vs. predicted RUL graphs.

Model loss/accuracy curves.

🧩 Project Structure
text
optmendix/
├── data/           # NASA CMAPSS datasets (FD001-004)
├── saved_models/   # Trained Keras models per dataset/model
├── results/        # Accuracy and loss plots, comparison CSVs
├── main.py         # Core ML pipeline: training, accuracy, reporting
├── app.py          # Flask backend for API and dashboard
├── templates/      # Jinja2 HTML templates (dashboard/result)
├── static/         # CSS, images, and web assets
└── README.md
🖥️ How to Use
Clone the Repository:

bash
git clone https://github.com/<yourusername>/OptiMendix.git
cd OptiMendix
Install Requirements:

bash
pip install -r requirements.txt
Prepare Data:

Download NASA C-MAPSS data and place the relevant .txt files in /data.

Example:

text
data/
  ├── train_FD001.txt
  ├── test_FD001.txt
  ...
Train and Evaluate Models:

bash
python main.py
Generates benchmarks in /results/ and saves trained models.

Launch the Dashboard:

bash
python app.py
Open http://localhost:5000 to use the full UI.

📊 Screenshots
(Replace with your actual screenshots or result images)

⚙️ Tech Stack
Backend: Python, TensorFlow/Keras, Flask

Frontend: HTML, CSS, JavaScript (+ FontAwesome), Jinja2

Data: NASA C-MAPSS Turbofan Engine Dataset

🤔 Why “OptiMendix”?
“Opti” stands for optimal/optimization, while “Mendix” emphasizes maintenance and repair. The name blends advanced technology and actionable industry insight.

🏆 Roadmap
 Multi-model AI backend (LSTM, GRU, Transformer)

 Desktop/mobile dashboard with card-based UI

 Evaluation metric auto-logging (MAE, RMSE, R²)

 Modern theme and visual polish

 Additional frontend features: tooltips, loaders, explainer modals (planned)

 Expanded real-time streaming/simulated sensor data (planned)

📜 License
MIT License—see LICENSE for details.

📝 Credits
Project lead: Vaibhav Saxena (final-year B.Tech capstone)

Mentorship: Ms. Rashika Bangroo

Data: NASA Armstrong Flight Research Center, C-MAPSS

Frontend: Custom Flask/CSS design with contributions from open source UI/UX patterns

For questions or feedback, please open an issue or contact [your.email@example.com].