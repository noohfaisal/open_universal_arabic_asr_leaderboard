
import sqlite3
import pandas as pd
import os

DB_NAME = "metrics.db"
OUTPUT_HTML = "db_report.html"

def main():
    if not os.path.exists(DB_NAME):
        print(f"Error: {DB_NAME} not found.")
        return

    conn = sqlite3.connect(DB_NAME)

    # Read experiments
    df_exp = pd.read_sql_query("SELECT * FROM experiments", conn)
    
    # Read inferences (joined with experiment name for clarity)
    df_inf = pd.read_sql_query("""
        SELECT 
            i.id, 
            e.model_name, 
            e.dataset_name, 
            i.audio_filepath, 
            i.wer, 
            i.cer, 
            i.inference_time, 
            i.rtf, 
            i.ground_truth, 
            i.prediction 
        FROM inferences i
        JOIN experiments e ON i.experiment_id = e.id
        ORDER BY i.id DESC
    """, conn)
    
    conn.close()

    # Create HTML
    html_string = f"""
    <html>
    <head>
        <title>Leaderboard Database Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            h2 {{ color: #333; }}
            table {{ border-collapse: collapse; width: 100%; margin-bottom: 30px; }}
            th, td {{ text-align: left; padding: 8px; border-bottom: 1px solid #ddd; }}
            th {{ background-color: #f2f2f2; }}
            tr:hover {{ background-color: #f5f5f5; }}
            .high-wer {{ color: red; font-weight: bold; }}
        </style>
    </head>
    <body>
        <h1>📊 Metrics Database Report</h1>
        
        <h2>🧪 Experiments (Runs)</h2>
        {df_exp.to_html(index=False, classes='table table-striped')}
        
        <h2>🔍 Detailed Inferences (Last 100)</h2>
        <p>Showing most recent results first.</p>
        {df_inf.head(100).to_html(index=False, classes='table table-striped')}
        
    </body>
    </html>
    """

    with open(OUTPUT_HTML, "w", encoding="utf-8") as f:
        f.write(html_string)
    
    print(f"Successfully generated visual report: {os.path.abspath(OUTPUT_HTML)}")

if __name__ == "__main__":
    main()
