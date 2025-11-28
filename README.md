NSCLC immunotherapy helper for metastatic NSCLC.

Problem: use simple clinicogenomic signals to guess durable clinical benefit (benefit=1) versus no benefit (0) so oncologists can triage immunotherapy.
Dataset: copy LungCanC2024_Dataset.csv into the project root as data.csv. The pipeline keeps age, sex, smoking, PD-L1, TMB, KRAS, EGFR and builds the label from survival>=18 months and immunotherapy_received==1.
Training: run `python model.py` once on your machine. It samples 20k rows, fits a lightweight logistic regression, and overwrites model.pkl. Commit the resulting model.pkl to keep deploys tiny.
Serve: run `python app.py` (or `PORT=10000 python app.py`). Flask loads the saved model, hosts the HTML form at `/`, and exposes `/predict` for JSON calls.
Render: create a Web Service from this repo. Build Command: `pip install -r requirements.txt`. Start Command: `python app.py`. Render provides PORT automatically; since the trained `model.pkl` is tracked in git, no heavy training runs on the server.
