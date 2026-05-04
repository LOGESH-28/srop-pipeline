import sqlite3, json
conn = sqlite3.connect('srop.db')
row = conn.execute("SELECT state_json FROM sessions WHERE id='e870c099-cdbc-425e-817b-fa0e5c88c8d0'").fetchone()
print(json.dumps(json.loads(row[0]), indent=2))
conn.close()