import json
import plotly
import numpy as np
import pandas as pd
import os
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar,Line,Scatter
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
fn="DisasterResponse.db"
engine = create_engine('sqlite:///'+fn)
df = pd.read_sql_table('Table', engine)

# load model
model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    
    labels=df.copy()
    labels.drop(['id','message','original','genre'],axis=1,inplace=True)
    
    f1=np.load('../models/F1_score.npy',allow_pickle=True)
    accuracy=np.load('../models/Accuracy_score.npy',allow_pickle=True)
    recall=np.load('../models/Recall_score.npy',allow_pickle=True)
    precision=np.load('../models/Precision_score.npy',allow_pickle=True)

    
    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        },
                {
            'data':[
                Scatter(
                    x=labels.columns,
                    y=accuracy,
                    mode='markers'
                )
            ],
            'layout':{
                'title': 'Accuracy_score of each category of the trained model',
                'yaxis': {
                    'title': "Accuracy_score"
                },
                'xaxis':{
                    'title': 'Categories'
                }
            }
        },
        {
            'data':[
                Scatter(
                    x=labels.columns,
                    y=f1,
                    mode='markers'
                    
                )
            ],
            'layout':{
                'title': 'F1_score of each category of the trained model',
                'yaxis': {
                    'title': "F1_score"
                },
                'xaxis':{
                    'title': 'Categories'
                }
            }
        },
                {
            'data':[
                Scatter(
                    x=labels.columns,
                    y=recall,
                    mode='markers'
                )
            ],
            'layout':{
                'title': 'Recall_score of each category of the trained model',
                'yaxis': {
                    'title': "Recall_score"
                },
                'xaxis':{
                    'title': 'Categories'
                }
            }
        },
                {
            'data':[
                Scatter(
                    x=labels.columns,
                    y=precision,
                    mode='markers'
                )
            ],
            'layout':{
                'title': 'Precision_score of each category of the trained model',
                'yaxis': {
                    'title': "Precision_score"
                },
                'xaxis':{
                    'title': 'Categories'
                }
            }
        }
    ]
    
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