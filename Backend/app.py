# app.py (Flask)
from flask import Flask, jsonify, send_from_directory
from flask_cors import CORS
import pandas as pd
from pathlib import Path

app = Flask(__name__, static_folder=None)
CORS(app)

BASE_DIR = Path(__file__).resolve().parent
CSV_FILE = BASE_DIR.parent / "backend_database" / "bmtc_cctv_counts.csv"  # ../backend_database/...

# serve frontend index (safe absolute path)
FRONTEND_DIR = BASE_DIR.parent / "frontend"

@app.route('/')
def index():
    return send_from_directory(str(FRONTEND_DIR), 'index.html')

@app.route('/api/busstop-counts', methods=['GET'])
def get_busstop_counts():
    try:
        # Use header=None if the CSV has no header row
        df = pd.read_csv(CSV_FILE, header=None)
        latest_row = df.iloc[-1]
        timestamp = latest_row.iloc[0]

        busstop_data = []
        # adjust these indices if your CSV columns differ
        busstop_columns = [2, 4, 6, 8, 10, 12, 14, 16, 18]

        for i, col_idx in enumerate(busstop_columns, start=1):
            busstop_data.append({
                'name': f'Bus Stop {i}',
                'count': int(latest_row.iloc[col_idx])
            })

        total_count = int(latest_row.iloc[-1])

        return jsonify({
            'timestamp': timestamp,
            'busstops': busstop_data,
            'total': total_count
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # run on port 5000
    app.run(debug=True, port=5000)
