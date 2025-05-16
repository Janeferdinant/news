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
