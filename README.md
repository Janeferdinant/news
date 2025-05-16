Fake News Detection with NLP and ACO
Enhanced fake news detection pipeline using a synthetic dataset of 20,000 news articles. Implements EDA, NLP preprocessing, Embedded-Filter ACO feature selection, and Logistic Regression/XGBoost models. Designed for JNTUK AI course (II Year, Term 1).
Features

EDA: Missing values, label/category distribution, word clouds, correlations.
Preprocessing: Text cleaning, TF-IDF, feature engineering (text length, sentiment).
Feature Selection: Embedded-Filter ACO with clustering-based MI.
Models: Logistic Regression and XGBoost.
Outputs: Plotly JSON for interactive reading card integration.

Setup
Kaggle

Upload fake_news_dataset.csv to Kaggle dataset (/kaggle/input/fake-news-dataset/).
Create notebook, copy fake_news_detection_enhanced_kaggle.ipynb.
Run all cells with internet enabled.

Colab

Upload fake_news_dataset.csv and notebook to Colab.
Install dependencies:!pip install flask scikit-learn numpy pandas plotly xgboost nltk textblob wordcloud


Run notebook cells. Use ngrok for Flask API:!pip install pyngrok
from pyngrok import ngrok
public_url = ngrok.connect(5000)
print(public_url)



VS Code

Clone repository:git clone https://github.com/<your-username>/fake-news-detection.git
cd fake-news-detection


Backend:cd backend
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python app.py


Notebook:
Open kaggle_notebook/fake_news_detection_enhanced_kaggle.ipynb in VS Code.
Install Jupyter extension, run with local Python environment.



Usage

Kaggle: View EDA plots, model performance, and save outputs.
Reading Card: Integrate JSON outputs (mi_scores.json, model_comparison.json) into React frontend (see frontend/index.html).
Colab/VS Code: Run notebook or Flask API for local development.

License
MIT
