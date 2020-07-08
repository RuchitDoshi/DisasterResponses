# DisasterResponses

1. Installation
2. Project Motivation
3. File Description
4. How to run the file and Web app?
5. Results
6. Licensing, Authors, and Acknowledgements

# Installation:
The code should run with no issues using Python versions 3.* Libraries such as Sklearn, nltk and flask would be required. Pickle and sqlalchemy libraries will be required to store the model and connect to a SQL database.

# Project Motivation:
Following a disaster, typically there are millions and millions of communication either direct or via social media right at the time when the disaster response orgranizations have the least capacity to filter and pull the messages which are most important. Typically 1 in 1000 messages are important to the people at such organizations. FigureEight have build a dataset containing real messages that were sent during disaster events and categorize them into different categories.

Inorder to tackle this problem, a supervised learning model can be build to categorize these events so that the user can send the messages to an appropriate disaster relief agency. The problem has been decided into three parts:

1. ETL pipeline:
In a Python script, process_data.py, a data cleaning pipeline is written that:

* Loads the messages and categories datasets
* Merges the two datasets
* Cleans the data
* Stores it in a SQLite database

2. Machine Leanring Pipeline:
In a Python script, train_classifier.py, a machine learning pipeline is written that:

* Loads data from the SQLite database
* Splits the dataset into training and test sets
* Builds a text processing and machine learning pipeline
* Trains and tunes a model using GridSearchCV
* Outputs results on the test set
* Exports the final model as a pickle file

3. Flask Web App
An app will be created where an emergency worker can input a new message and get classification results in several categories. The web app will also display visualizations of the data. 

