# Disaster-Response-Pipeline
## Udacity Data Science Nanodegree Project

### Table of Contents
1. [Description](#description)
2. [Dependencies](#dependencies)
3. [Files Description](#Files-Description)
4. [Acknowledgements](#Acknowledgements)
5. [Screenshots](#screenshots)

### Description
![Deployed app](webapp.png)

This Project is part of Data Science Nanodegree Program by Udacity in collaboration with Figure Eight. The dataset contains pre-labelled social media posts and news from real-life disaster events. I built and deployed a Natural Language Processing (NLP) model to categorize messages on a real time basis.

This project is divided in the following key sections:

1. Processing data, building an ETL pipeline to extract data from source, clean the data and save them in a SQLite DB
2. Build a machine learning pipeline to train the which can classify text message in various categories
3. Run a web app that can classify new messages

### Dependencies
Python 3.11 with the following libraries:
1. Machine Learning: NumPy, SciPy, Pandas, Scikit-Learn
2. Natural Language Process: NLTK
3. SQLlite Database: SQLalchemy
4. Model Loading and Saving: Pickle
5. Web App and Data Visualization: Flask, Plotly

You can find all the requirements and versions in the file "requirements.txt".

### Files Description
1. App folder including the templates folder and "run.py" for the web application
2. Data folder containing "DisasterResponse.db", "disaster_categories.csv", "disaster_messages.csv" and "process_data.py" for data cleaning and transfering.
3. Models folder including "classifier.pkl" and "train_classifier.py" for the Machine Learning model.
4. README file

### Instructions
1. You can run the following commands in the project's directory to set up the database, train model and save the model.
2. To run ETL pipeline to clean data and store the processed data in the database python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/disaster_response_db.db
3. To run the ML pipeline that loads data from DB, trains classifier and saves the classifier as a pickle file python models/train_classifier.py data/disaster_response_db.db models/classifier.pkl
Run the following command in the app's directory to run your web app. python run.py
4. Go to http://0.0.0.0:3001/

### Acknowledgements
* Udacity for providing an excellent Data Scientist training program.
* Figure Eight for providing dataset to train our model.