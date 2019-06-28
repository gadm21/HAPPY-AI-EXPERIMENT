# Midnight MySQL cleaning bad record

# Import Libraries
import database.db_info as db

# MYSQL Database Connection #
mydb, db_cursor = db.db_connection()

# Function to clean
def remove_bad_records():
    query = "DELETE FROM people WHERE dwell_time < 3;"
    db_cursor.execute(query)
    mydb.commit()

def update_index(): 
    query1 = "SET @count = 0;"
    query2 = "UPDATE people SET people.id = @count:= @count + 1;"    
    query3 = "ALTER TABLE people AUTO_INCREMENT = 1;"
    db_cursor.execute(query1)
    db_cursor.execute(query2)
    db_cursor.execute(query3)
    mydb.commit()

remove_bad_records()
update_index()
