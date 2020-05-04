import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar, Pie, Scatter, Layout, Figure
from plotly.subplots import make_subplots
from sklearn.externals import joblib
from sqlalchemy import create_engine


app = Flask(__name__)

def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('messages', engine)

# load model
model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():

    # extract data needed for visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    print(genre_names)

    genre_composition = df.loc[:,'genre':].groupby('genre').sum()
    direct_composition = genre_composition.iloc[0].sort_values()
    news_composition = genre_composition.iloc[1].sort_values()
    social_composition = genre_composition.iloc[2].sort_values()

    # create visuals
    fig0 = make_subplots(rows=1, cols=2, specs=[[{"type": "pie"}, {"type": "bar"}]])

    fig0.add_trace(
        Pie(labels=genre_names, values=genre_counts, textinfo='label+percent'),
        row=1, col=1
    )
    fig0.add_trace(
        Bar(x=genre_names, y=genre_counts,marker=dict(color=[4, 5, 1], coloraxis="coloraxis")),
        row=1, col=2
    )

    fig0.update_layout(title_text='Distribution of Message Genres', showlegend=False)

    trace1 = [Bar(name='direct', x=direct_composition.index.tolist(), y=direct_composition),
              Bar(name='news', x=news_composition.index.tolist(), y=news_composition),
              Bar(name='social', x=social_composition.index.tolist(), y=social_composition)]
    layout1 = Layout(title= 'Labels of Messages under Three Genres',
                     yaxis= {'title': "Count"},
                     barmode='stack'
                    )
    fig1 = Figure(data = trace1, layout = layout1)


    graphs = [fig0, fig1]

    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)

    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '')

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file.
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()
