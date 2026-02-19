
import sqlite3
import datetime
import os

DB_NAME = "metrics.db"

def init_db():
    """Initialize the database schema."""
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    
    # Experiments table: High-level run info
    # Added UNIQUE constraint for Upsert logic
    c.execute('''CREATE TABLE IF NOT EXISTS experiments
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  model_name TEXT,
                  dataset_name TEXT,
                  timestamp DATETIME,
                  global_wer REAL,
                  global_cer REAL,
                  avg_latency REAL,
                  UNIQUE(model_name, dataset_name))''')
    
    # Inferences table: Granular file-level info
    c.execute('''CREATE TABLE IF NOT EXISTS inferences
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  experiment_id INTEGER,
                  audio_filepath TEXT,
                  ground_truth TEXT,
                  prediction TEXT,
                  wer REAL,
                  cer REAL,
                  duration REAL,
                  inference_time REAL,
                  rtf REAL,
                  FOREIGN KEY(experiment_id) REFERENCES experiments(id) ON DELETE CASCADE,
                  UNIQUE(experiment_id, audio_filepath))''')
    
    conn.commit()
    conn.close()
    print(f"Database {DB_NAME} initialized.")

def clear_db():
    """Clear all data from the database."""
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("DROP TABLE IF EXISTS inferences")
    c.execute("DROP TABLE IF EXISTS experiments")
    conn.commit()
    conn.close()
    print(f"Database {DB_NAME} cleared.")
    init_db()

def log_experiment(model_name, dataset_name, global_wer, global_cer, avg_latency):
    """Log a new experiment and return its ID. Upserts if (model_name, dataset_name) exists."""
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    timestamp = datetime.datetime.now()
    
    # Use INSERT OR REPLACE to handle upserts
    # Note: REPLACE deletes the old row and inserts a new one, so the ID might change.
    # This is fine as long as we use the returned ID for inferences.
    # However, we must ensure old inferences for this experiment are removed if the ID *doesn't* change (UPDATE)
    # or if it does (REPLACE triggers deletion if we used foreign keys properly, but SQLite FKs are off by default)
    # Safer approach: Check existing, delete it manually if exists (cascading to inferences if manual), then insert.
    
    # Enable foreign keys
    c.execute("PRAGMA foreign_keys = ON")
    
    # Delete existing experiment with same unique key (this cascades delete to inferences if tables created with ON DELETE CASCADE)
    # But since we just altered the schema above, we might need to recreate tables for ON DELETE CASCADE to work or handle it manually.
    # Given the user wants to "clear db" first, we will start fresh with correct schema.
    
    c.execute("DELETE FROM experiments WHERE model_name=? AND dataset_name=?", (model_name, dataset_name))
    
    c.execute("INSERT INTO experiments (model_name, dataset_name, timestamp, global_wer, global_cer, avg_latency) VALUES (?, ?, ?, ?, ?, ?)",
              (model_name, dataset_name, timestamp, global_wer, global_cer, avg_latency))
    exp_id = c.lastrowid
    conn.commit()
    conn.close()
    return exp_id

def log_inferences(experiment_id, items):
    """
    Log a batch of inference results.
    items: List of dictionaries containing:
           audio_filepath, text, pred_text, wer, cer, duration, inference_time, rtf
    """
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    data = []
    for item in items:
        # data tuple matching the columns in the INSERT statement
        data.append((
            experiment_id,
            item.get('audio_filepath', ''),
            item.get('text', ''),
            item.get('pred_text', ''),
            item.get('wer', 0.0),
            item.get('cer', 0.0),
            item.get('duration', 0.0),
            item.get('inference_time', 0.0),
            item.get('rtf', 0.0)
        ))
        
    c.executemany("""INSERT INTO inferences 
                     (experiment_id, audio_filepath, ground_truth, prediction, wer, cer, duration, inference_time, rtf) 
                     VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""", data)
    conn.commit()
    conn.close()
    print(f"Logged {len(items)} inferences for Experiment ID {experiment_id}.")

def get_all_experiments():
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("SELECT * FROM experiments")
    rows = c.fetchall()
    conn.close()
    return rows

def get_inferences_for_experiment(experiment_id):
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("SELECT * FROM inferences WHERE experiment_id=?", (experiment_id,))
    rows = c.fetchall()
    conn.close()
    return rows
