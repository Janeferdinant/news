!pip install flask scikit-learn numpy pandas plotly xgboost nltk textblob wordcloud
!pip install pyngrok
from pyngrok import ngrok
public_url = ngrok.connect(5000)
print(public_url)
# Paste app.py content


from flask import Flask, jsonify
import json
import os

app = Flask(__name__)

@app.route('/get_kaggle_outputs', methods=['GET'])
def get_kaggle_outputs():
    try:
        with open('/kaggle/working/mi_scores.json') as f:
            mi_data = json.load(f)
        with open('/kaggle/working/model_comparison.json') as f:
            model_data = json.load(f)
        with open('/kaggle/working/selected_features.txt') as f:
            selected_features = f.read().splitlines()
        return jsonify({
            'mi_scores': mi_data,
            'model_comparison': model_data,
            'selected_features': selected_features[:20]
        })
    except FileNotFoundError:
        return jsonify({'error': 'Output files not found'}), 404

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
