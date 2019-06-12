# Midnight MySQL cleaning bad record

# Import Libraries
import database.db_info as db

# MYSQL Database Connection #
mydb, db_cursor = db.db_connection()

# Function to clean
def remove_bad_records():
    query = "DELETE FROM people WHERE dwell_time < 5;"
    db_cursor.execute(query)
    mydb.commit()

remove_bad_records()