#import libraries
import sys
import numpy as np
import pandas as pd
import sqlalchemy
from sqlalchemy import create_engine


#function to load and merge .csv files in a DataFrame
def load_data(messages_filepath, categories_filepath):
    messages = pd.read_csv(messages_filepath)
    categories=pd.read_csv(categories_filepath)

    #merging the two files in one dataFrame
    df = messages.merge(categories,how='outer',on=['id'])
    return df



def clean_data(df):
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
    
    #Asserting zero duplicates in the dataframe
    assert(len(duplicateDFRow)==0)
    return df


def save_data(df, database_filename):
    
    #saving the file to a Database and naming the table as 'Table' 
    engine = create_engine('sqlite:///' + database_filename)
    df.to_sql('Table', engine, index=False)
     


def main():
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