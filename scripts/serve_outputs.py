"""Optional local JSON API for the notebook outputs. The live page (index.html) does not need it:
it reads outputs/*.json directly as static files.

    pip install flask
    python scripts/serve_outputs.py                           # serves ./outputs on http://127.0.0.1:5000
    OUTPUT_DIR=/kaggle/working python scripts/serve_outputs.py
"""
import json
import os

from flask import Flask, jsonify

app = Flask(__name__)
OUTPUT_DIR = os.environ.get('OUTPUT_DIR', 'outputs')


@app.route('/get_kaggle_outputs', methods=['GET'])
def get_kaggle_outputs():
    try:
        with open(os.path.join(OUTPUT_DIR, 'mi_scores.json')) as f:
            mi_data = json.load(f)
        with open(os.path.join(OUTPUT_DIR, 'model_comparison.json')) as f:
            model_data = json.load(f)
        with open(os.path.join(OUTPUT_DIR, 'selected_features.txt')) as f:
            selected_features = f.read().splitlines()
        return jsonify({
            'mi_scores': mi_data,
            'model_comparison': model_data,
            'selected_features': selected_features[:20]
        })
    except FileNotFoundError:
        return jsonify({'error': 'Output files not found'}), 404


if __name__ == '__main__':
    # Local use only: bind to the loopback address, not 0.0.0.0, and do not expose it through a tunnel.
    app.run(host='127.0.0.1', port=5000)
