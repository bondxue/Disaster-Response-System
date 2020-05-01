import sys
import pandas as pd
import numpy as np
import pickle
from sqlalchemy import create_engine

# NLP
import nltk
nltk.download(['punkt', 'wordnet'])
from string import punctuation
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# ML
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier


def load_data(database_filepath):
    """
    function to load data from SQLite database
    inputs:
        database_filepath: string path of database file 
    returns:
        X: df of ml input messages
        y: df of ml labeled output categories 
        category_names: labeled output category names 
    """
    # load data from database
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df = pd.read_sql_table('messages', engine)
    
    X = df['message']
    y = df.drop(['id', 'message', 'original', 'genre'], axis=1)
    category_names = y.columns
    return X, y, category_names

def tokenize(text):
    """
    function to tokenize text data
    input:
        text: text raw data
    return:
        clean_tokens: list of clean tokens
    """
    
    # remove punctations
    text =  ''.join([c for c in text if c not in punctuation])
    
    #tokenize text
    tokens = word_tokenize(text)
    
    # initiate lemmatizer
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for token in tokens:
        # lemmatize, normalize case, and remove leading/trailing white space
        clean_tok = lemmatizer.lemmatize(token).lower().strip()
        clean_tokens.append(clean_tok)
    return clean_tokens


def build_model():
    """
    function to create and train a classifer 
    """
    pipeline =  Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier((AdaBoostClassifier())))
    ])
    # grid search parameters
    parameters = {
    'tfidf__norm':['l2','l1'],
    'vect__stop_words': ['english'],
    'clf__estimator__learning_rate' :[0.1, 0.5],
    'clf__estimator__n_estimators' : [10],
    }
    #create grid search object
    model = GridSearchCV(estimator=pipeline, param_grid = parameters, cv = 3, n_jobs=-1)
    return model


def evaluate_model(model, X_test,y_test, category_names):
    """
    function to generate evalutation report for given model and test data
    inputs:
        model: trained classifer object 
        X_test: test input data
        y_test: test output label data
        category_names: labeled output category names 
    """
    # predict on test data
    y_pred = model.predict(X_test)

    performances = []
    
    for i, col in enumerate(category_names):
        precision, recall, f_score, support = score(y_test.iloc[:,i], y_pred[:,i], average='micro')
        performances.append([precision, recall, f_score])
        
    performances = pd.DataFrame(performances, columns=['precesion', 'recall', 'f1-score'],
                                index = category_names) 
    print(performances)


        
def save_model(model, model_filepath):
    """
    function to store the classifer into a pickle file 
    inputs:
        model: classifer object 
        model_filepath: model file path    
    """
    pickle.dump(model, open(model_filepath, 'wb'))


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