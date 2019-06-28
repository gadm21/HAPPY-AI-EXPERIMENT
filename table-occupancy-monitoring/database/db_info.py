import mysql.connector
import json

# Read JSON Configuration File #
with open('config.json') as f:
    data = json.load(f)

database = data['database']

# MySQL Database Connector #
def db_connection():

	mydb = mysql.connector.connect(
		host = database['host'],
		user = database['user'],
		passwd = database['passwd'],
		database = database['database']
	)

	db_cursor = mydb.cursor(buffered=True, dictionary=True)
	return mydb, db_cursor