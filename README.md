Fake News Detection with NLP, ACO, and BERT
A comprehensive fake news detection pipeline using a synthetic dataset of 20,000 news articles. Implements EDA, NLP preprocessing, Embedded-Filter ACO feature selection, and multiple models (Logistic Regression, XGBoost, BERT). Integrates with an interactive React-based reading card. Designed for JNTUK AI course (II Year, Term 1).
Repository Structure
fake-news-detection/
├── backend/
│   ├── app.py                    # Flask API
│   ├── requirements.txt
├── frontend/
│   ├── index.html                # React reading card
├── kaggle_notebook/
│   ├── fake_news_detection_enhanced_kaggle.ipynb
├── data/
│   ├── fake_news_dataset.csv     # Dataset (optional)
├── docs/
│   ├── README.md
│   ├── dataset_description.md
├── LICENSE

Features

EDA: Missing values, label/category distribution, word clouds, sentiment analysis.
Preprocessing: Text cleaning, TF-IDF, feature engineering (text length, sentiment).
Feature Selection: Embedded-Filter ACO with clustering-based MI.
Models: Logistic Regression, XGBoost, DistilBERT.
Frontend: React UI with 8 themes, displaying Kaggle results.
Outputs: Plotly JSON for visualizations.

Setup
Kaggle

Upload fake_news_dataset.csv to Kaggle (/kaggle/input/fake-news-dataset/).
Create notebook, upload fake_news_detection_enhanced_kaggle.ipynb.
Enable GPU and internet in settings.
Run all cells.

Colab

Upload fake_news_dataset.csv and notebook.
Install dependencies:!pip install flask scikit-learn numpy pandas plotly xgboost nltk textblob wordcloud transformers torch


Run notebook. For Flask API:!pip install pyngrok
from pyngrok import ngrok
public_url = ngrok.connect(5000)
print(public_url)
# Paste app.py content


Host index.html via GitHub Pages or local server.

VS Code

Clone repository:git clone https://github.com/<your-username>/fake-news-detection.git
cd fake-news-detection


Backend:cd backend
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python app.py


Notebook:
Open kaggle_notebook/fake_news_detection_enhanced_kaggle.ipynb.
Install Jupyter extension, run with local Python.


Frontend:cd frontend
npx serve



Usage

Kaggle: Generate EDA plots, model results, and JSON outputs.
Reading Card: View selected features, MI scores, and model comparison in React UI.
Colab/VS Code: Run notebook or Flask API for local development.

Deployment

GitHub Pages: Host index.html (frontend).
Heroku/Render: Deploy app.py (backend).
Kaggle: Share notebook publicly.

License
MIT
