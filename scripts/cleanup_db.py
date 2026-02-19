import sqlite3
import os

DB_NAME = "metrics.db"
IDS_TO_REMOVE = [4, 5]

if not os.path.exists(DB_NAME):
    print("Database not found.")
    exit(1)

conn = sqlite3.connect(DB_NAME)
cursor = conn.cursor()

try:
    # Remove inferences first (Constraint usually cascades but let's be safe or explicit)
    # Actually, if FK has ON DELETE CASCADE it handles it. 
    # Let's check schema/assumption. Default sqlite usually requires pragma foreign_keys=ON.
    # We will delete explicitly.
    
    print(f"Removing Experiments: {IDS_TO_REMOVE}")
    
    placeholders = ', '.join('?' * len(IDS_TO_REMOVE))
    
    cursor.execute(f"DELETE FROM inferences WHERE experiment_id IN ({placeholders})", IDS_TO_REMOVE)
    print(f"Deleted {cursor.rowcount} inferences.")
    
    cursor.execute(f"DELETE FROM experiments WHERE id IN ({placeholders})", IDS_TO_REMOVE)
    print(f"Deleted {cursor.rowcount} experiments.")
    
    conn.commit()
    print("Database cleanup successful.")

except Exception as e:
    print(f"Error: {e}")
    conn.rollback()
finally:
    conn.close()
