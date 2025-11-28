NSCLC immunotherapy helper for metastatic NSCLC.

Problem: use simple clinicogenomic signals to guess durable clinical benefit (benefit=1) versus no benefit (0) so oncologists can triage immunotherapy.
Dataset: copy LungCanC2024_Dataset.csv into the project root as data.csv. The pipeline keeps age, sex, smoking, PD-L1, TMB, KRAS, EGFR and builds the label from survival>=18 months and immunotherapy_received==1.
Training: run `python model.py` once to preprocess data and fit the random forest, which saves to model.pkl.
Serve: run `python app.py` (or set `PORT=10000 python app.py`). Flask loads the saved model, hosts the HTML form at `/`, and exposes `/predict` for JSON calls.
Render: create a free Web Service, upload this repo, set Build Command to `pip install -r requirements.txt && python model.py` so the model file exists, set Start Command to `python app.py`, and add the environment variable `PORT` that Render provides automatically.
