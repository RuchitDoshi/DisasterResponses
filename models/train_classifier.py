import sys
import sqlalchemy
from sqlalchemy import create_engine
import pickle
import pandas as pd
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report,f1_score,precision_score, recall_score,accuracy_score
import nltk
nltk.download(['punkt', 'wordnet'])

def load_data(database_filepath):
    '''
    Function to load data and split them into features and labels
    Args: database_filepath: Filepath to load the database in .db format
    
    Output:
    X: features array
    Y: DataFrame containing all the categories
    Columns_names: list of column names for all the categories 
    
    '''
    engine = create_engine('sqlite:///'+ database_filepath)
    df = pd.read_sql_table('Table',engine)
    X = df['message']
    Y=df.copy()
    Y.drop(['id','message','original','genre'],axis=1,inplace=True)
    return X,Y,Y.columns


def tokenize(text):
    '''
    Function to tokenize, lemmatize and case normalize the messages
    Args: text: each message in str format
    
    Output: list of clean words 
    '''
    
    #Tokenize the text
    tokens=word_tokenize(text)
    
    #Initial lemmatizer
    lemmatizer=WordNetLemmatizer()
    clean_tokens=[]
    for tok in tokens:
        #Case normalize and strip words
        clean_tok=lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)
    
    return clean_tokens


def build_model():
    '''
    Function to build model and GridSearch the parameters
    Args: None
    
    Output: Model with the paramaters on which the gridsearch will take place
    '''
    #Build the pipeline
    pipeline = Pipeline([('vect',CountVectorizer(tokenizer=tokenize)),
                    ('tfidf',TfidfTransformer()),
                    ('clf',MultiOutputClassifier(KNeighborsClassifier()))])
    
    #List of Parameters on which GridSearch will take place
    parameters={'clf__estimator__n_neighbors':[3,5],
                'clf__estimator__leaf_size':[30,35]}
    #Build the model with pipeline and paramters
    model =GridSearchCV(pipeline,param_grid=parameters)
    return model


def evaluate_model(model, X_test, Y_test, category_names):
    '''
    Function to evaluate the model and print the classification report on each category
    Args:
    model: Trained model 
    X_test: test set of the features
    Y_test: test set of the groundtruth values
    category_names: list of the category_names
    
    Output: None
    
    '''
    
    #Predict the output
    Y_pred=model.predict(X_test)
    f1=[]
    recall=[]
    precision=[]
    acc=[]
    
    for col in category_names:
        idx=Y_test.columns.get_loc(col)
        
        print('\033[1m')
        print(col + ':')
        print('\033[0m')
        print(classification_report(Y_test.values[:,idx],Y_pred[:,idx]))#prints classification report for each category
        print()
        print('######################################')
        f1.append(f1_score(Y_test.values[:,idx],Y_pred[:,idx]))
        recall.append(recall_score(Y_test.values[:,idx],Y_pred[:,idx]))
        acc.append(accuracy_score(Y_test.values[:,idx],Y_pred[:,idx]))
        precision.append(precision_score(Y_test.values[:,idx],Y_pred[:,idx]))
    
    # Saves these metrics which is used for visualizations in the Web app
    np.save('models/F1_score.npy',np.array(f1))
    np.save('models/Recall_score.npy',np.array(recall))
    np.save('models/Accuracy_score.npy',np.array(acc))
    np.save('models/Precision_score.npy',np.array(precision))
    
    
def save_model(model, model_filepath):
    #Saves the modelfile
    pickle.dump(model, open(model_filepath, 'wb'))
    


def main():
    '''
    Combines all the functions
    '''
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