# backend.py (FastAPI)
from fastapi import FastAPI
import pandas as pd
import os
from pathlib import Path

app = FastAPI()

BASE_DIR = Path(__file__).resolve().parent
CSV_FILE = BASE_DIR.parent / "backend_database" / "bmtc_cctv_counts.csv"  # ../backend_database/...

@app.get("/api/passenger_counts")
def get_passenger_counts():
    if not CSV_FILE.exists():
        return {"status": "error", "message": f"CSV file '{CSV_FILE}' not found."}
    try:
        # If first row is header, use header=0; if not, use header=None
        df = pd.read_csv(CSV_FILE, header=0)
        data = df.to_dict(orient="records")
        return {"status": "success", "data": data}
    except Exception as e:
        return {"status": "error", "message": str(e)}
