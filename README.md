# Disaster Response System

### Introduction

In this project, I have built a machine learning system to analyze disaster data from [Figure Eight](https://www.figure-eight.com/) to build a model for an API that classifies disaster messages.

### Project Components

File structure of the project:

```python
- app
| - template
| |- master.html  # main page of web app
| |- go.html  # classification result page of web app
|- run.py  # Flask file that runs app

- data
|- disaster_categories.csv  # data to process 
|- disaster_messages.csv  # data to process
|- process_data.py # ETL pipline 
|- DisasterResponse.db   # database to save clean data to

- models
|- train_classifier.py # ML pipline 
|- classifier.pkl  # saved model 

- notebooks
|- ETL Pipeline Preparation.ipynb # test ETL pipline 
|- ML Pipeline Preparation.ipynb # test ML pipline

- README.md
```

There are three main Python scripts this project:

1. `process_data.py`:  contain ETL pipeline
   + Loads the `messages` and `categories` datasets
   + Merges the two datasets
   + Cleans the data
   + Stores it in a SQLite database

2. `train_classifier.py`: contain ML pipeline
   - Loads data from the SQLite database
   - Splits the dataset into training and test sets
   - Builds a text processing and machine learning pipeline
   - Trains and tunes a model using GridSearchCV
   - Outputs results on the test set
   - Exports the final model as a pickle file

3. `run.py `: Flask file that runs app
   + data visualizations using Plotly in the web app

### Running Procedure

1. Run the following commands in the project's root directory to set up database and model.

    - To run ETL pipeline that cleans data and stores in database

        ```
        python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db
        ```

    - To run ML pipeline that trains classifier and saves
        
        ```python
        python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl
        ```

2. Run the following command in the app's directory to run the web app.
    
```python
    python run.py
```
    
3. Go to http://0.0.0.0:3001/


### Demo Display


