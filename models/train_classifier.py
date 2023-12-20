import sys
import re
import pandas as pd
from sqlalchemy import create_engine
import nltk
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.metrics import classification_report
from sklearn.multioutput import MultiOutputClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import GridSearchCV
import pickle

# need to download these files from nltk to use the library
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')


def load_data(database_filepath):
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df = pd.read_sql("Select * from Messages", engine)
    X = df['message']
    Y = df.drop(columns=['id', 'message', 'original', 'genre'])
    # Y = Y.drop(columns=['child_alone']) #EDA verified this column only has zeros
    category_names = list(Y.columns)
    return X, Y, category_names

def tokenize(text):
    stop_words = stopwords.words("english")
    lemmatizer = WordNetLemmatizer()
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(w) for w in tokens if w not in stop_words]
    return tokens


def build_model():
    pipeline = Pipeline([
        ('BoW', CountVectorizer(tokenizer=tokenize)),
        ('tfidf-transf', TfidfTransformer()),
        ('classifier', MultiOutputClassifier(LogisticRegression()))
    ])

    parameters = {
        'BoW__max_features': [10, 100, 1000, None],
        'tfidf-transf__smooth_idf': [True, False],
        # 'classifier__estimator__C': [0.001, 0.1, 1, 10],
        # 'classifier__estimator__penalty': ['l1', 'l2'] #takes too long
    }

    cv = GridSearchCV(pipeline, param_grid=parameters)
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    Y_pred = model.predict(X_test)
    Y_pred_df = pd.DataFrame(Y_pred, columns=Y_test.columns)

    for column in category_names:
        print(column)
        print(classification_report(Y_test[column], Y_pred_df[column]))


def save_model(model, model_filepath):
    with open(model_filepath, 'wb') as file:
        pickle.dump(model, file)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()