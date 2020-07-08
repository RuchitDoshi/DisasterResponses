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
    engine = create_engine('sqlite:///'+ database_filepath)
    df = pd.read_sql_table('Table',engine)
    X = df['message']
    Y=df.copy()
    Y.drop(['id','message','original','genre'],axis=1,inplace=True)
    return X,Y,Y.columns


def tokenize(text):
    tokens=word_tokenize(text)
    
    lemmatizer=WordNetLemmatizer()
    clean_tokens=[]
    for tok in tokens:
        clean_tok=lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)
    
    return clean_tokens


def build_model():
    model = Pipeline([('vect',CountVectorizer(tokenizer=tokenize)),
                    ('tfidf',TfidfTransformer()),
                    ('clf',MultiOutputClassifier(KNeighborsClassifier()))])
    return model


def evaluate_model(model, X_test, Y_test, category_names):
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
        print(classification_report(Y_test.values[:,idx],Y_pred[:,idx]))
        print()
        print('######################################')
        f1.append(f1_score(Y_test.values[:,idx],Y_pred[:,idx],average='micro'))
        recall.append(recall_score(Y_test.values[:,idx],Y_pred[:,idx],average='micro'))
        acc.append(accuracy_score(Y_test.values[:,idx],Y_pred[:,idx]))
        precision.append(precision_score(Y_test.values[:,idx],Y_pred[:,idx],average='micro'))
    
    np.save('F1_score.npy',np.array(f1))
    np.save('Recall_score.npy',np.array(recall))
    np.save('Accuracy_score.npy',np.array(acc))
    np.save('Precision_score.npy',np.array(precision))
    
    
def save_model(model, model_filepath):
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