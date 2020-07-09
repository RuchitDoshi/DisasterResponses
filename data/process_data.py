import sys
import numpy as np
import pandas as pd
import sqlalchemy
from sqlalchemy import create_engine
# import sqlite3


def load_data(messages_filepath, categories_filepath):
    '''
    Function to load data and merge them into one file
    Args: 
    messages_filepath: Filepath to load the messages.csv
    categories_filepath: Filepath to load the categories.csv
    
    Output:
    df: combined dataFrame
    
    '''
    messages = pd.read_csv(messages_filepath)
    categories=pd.read_csv(categories_filepath)
    df = messages.merge(categories,how='outer',on=['id'])
    return df


def clean_data(df):
    '''
    Function to clean the combined DataFrame and have columns for each category with binary inputs
    Args: df: Merged dataFrame 
    
    Output:
    df : Clean dataFrame
    
    '''
    categories = pd.DataFrame(df['categories'].str.split(';',expand=True))
    row = categories.iloc[0,:]
    category_colnames =row.apply(lambda x:x[:-2]) 
    
    # rename the columns of `categories`
    categories.columns = category_colnames
    for column in categories:
    # set each value to be the last character of the string
        categories[column] = categories[column].apply(lambda x:str(x)[-1])
    
    # convert column from string to numeric
        categories[column] = categories[column].apply(lambda x:int(x))
    
    
    #converting categories into binary format of 1's and 0's
    categories=categories.apply(lambda x:(x!=0).astype(int))
    
    # drop the original categories column from `df`
    df=df.drop(['categories'],axis=1,inplace=False)
    
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df,categories],axis=1)
    
    # check number of duplicates
    duplicateDFRow=df[df.duplicated()]
    
    # drop duplicates
    df=df.drop_duplicates()
    
    # check number of duplicates
    duplicateDFRow=df[df.duplicated()]
    
    assert(len(duplicateDFRow)==0)
    return df


def save_data(df, database_filename):
    '''
    Saves the clean DataFrame to the database
    
    Args: database_filename: Filepath to store the filename
    
    Output:None
    '''
    engine = create_engine('sqlite:///'+ database_filename)
    df.to_sql('Table', engine, index=False)
     


def main():
    '''
    Function to combine all above functions
    '''
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()