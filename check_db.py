import sqlite3

# Connect to the database
conn = sqlite3.connect('chatbot.db')
c = conn.cursor()

# Get list of tables
c.execute("SELECT name FROM sqlite_master WHERE type='table';")
tables = c.fetchall()
print("Tables in the database:")
for table in tables:
    print(f"  - {table[0]}")

print("\n" + "="*50)

# Check master table
print("Users table (master):")
try:
    c.execute("SELECT * FROM master")
    users = c.fetchall()
    for user in users:
        print(f"  ID: {user[0]}, Name: {user[1]}, Region: {user[2]}, Type: {user[3]}, Created: {user[4]}")
except Exception as e:
    print(f"  Error reading master table: {e}")

print("\n" + "="*50)

# Check chats table
print("Chat history table (chats):")
try:
    c.execute("SELECT * FROM chats")
    chats = c.fetchall()
    for chat in chats:
        print(f"  ID: {chat[0]}, User ID: {chat[1]}, Message: {chat[2][:50]}{'...' if len(chat[2]) > 50 else ''}, Response: {chat[3][:50]}{'...' if len(chat[3]) > 50 else ''}, Time: {chat[4]}")
except Exception as e:
    print(f"  Error reading chats table: {e}")

# Close connection
conn.close()