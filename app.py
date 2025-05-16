!pip install flask scikit-learn numpy pandas plotly xgboost nltk textblob wordcloud
!pip install pyngrok
from pyngrok import ngrok
public_url = ngrok.connect(5000)
print(public_url)
# Paste app.py content
from flask import Flask
app = Flask(__name__)
# ... (app.py content)
app.run(port=5000)
@app.route('/get_kaggle_outputs', methods=['GET'])
def get_kaggle_outputs():
    with open('/kaggle/working/mi_scores.json') as f:
        mi_data = json.load(f)
    with open('/kaggle/working/model_comparison.json') as f:
        model_data = json.load(f)
    return jsonify({'mi_scores': mi_data, 'model_comparison': model_data})
