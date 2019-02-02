import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    '''
    Reading the raw data

    Parameter:
    messages_filepath(str): the file path of message csv file
    categories_filepath(str): the file path of catergories csv file

    Return:
    df(Dataframe): the merge file of message and catergories

    '''
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = pd.merge(messages,categories,how="outer",on="id")
    return df


def clean_data(df):

    '''
    clean data, extract categories into a list 
    and change the labels into numerical types.

    Parameter:
    df(Dataframe): the data set 

    Return:
    df(Dataframe): the data set with categories 

    '''
    categories = df.categories.str.split(";",n=-1, expand=True)
    # select the first row of the categories dataframe
    row = categories.iloc[0].values

    # use this row to extract a list of new column names for categories.
    # one way is to apply a lambda function that takes everything 
    # up to the second to last character of each string with slicing
    category_colnames = [ x[:-2] for x in row ]
    categories.columns = category_colnames
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].astype(str).str[-1]
        
        # convert column from string to numeric
        categories[column] = categories[column].astype(int)

    #CHANGE THE VALUES LARGER THAN 1 INTO 1
    categories[categories>1]=1
    # drop the original categories column from `df`
    df.drop("categories",axis=1,inplace=True)

    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df,categories],axis=1)

    # drop duplicates
    df.drop_duplicates(inplace=True)

    return df



def save_data(df, database_filename):
    '''
    Save Dataframe into database file

    Parameter:
    df(Dataframe): the data set 
    database_filename(str): the name of the databas file

    Return:
    None

    '''
    engine = create_engine('sqlite:///'+database_filename)
    df.to_sql("DisasterResponse", engine, index=False)
  


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