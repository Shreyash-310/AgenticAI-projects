# setup_db.py  (run once to create example DB)
import sqlite3

conn = sqlite3.connect("D:/GenAI-Practice/AgenticAI-Projects/SmartAssistant/app/data/uploads/system_details.db")
cur = conn.cursor()

cur.execute("""
CREATE TABLE IF NOT EXISTS system_info (
    id INTEGER PRIMARY KEY,
    Provision TEXT,
    System_Data TEXT,
)
""")
print('Table created successfully')
# cur.executemany("""
# INSERT INTO customers (Provision, System_Data)
# VALUES (?, ?)
# """, [
#     ("Survivor annuity percentage", "0.5"),
#     ("Plan eligibility for emlpoyees", "All Employees"),
#     ("Eligibility to participate after termination of employement", "Date reemployement"),
#     ("Notice of withdrawal", "30 Days"),
#     ("Forfeiture allocation percentage", "1"),
#     ("Distribution eligibility for terminated employees", "Yes"),
#     ("Distribution eligibility for active employees", "59.5 years"),
#     ("Eligibility of withdrawal of entire individual account during service", "5 years"),
#     ("Maximum withdrawal amount for an active particiant without minimum service", "India"),
#     ("Alice", "India"),
#     ("Bob", "USA"),
#     ("Charlie", "India"),
#     ("Alice", "India"),
#     ("Bob", "USA"),
#     ("Charlie", "India"),
#     ("Alice", "India"),
#     ("Bob", "USA"),
#     ("Charlie", "India"),
#     ("Bob", "USA"),
#     ("Charlie", "India"),
# ])

conn.commit()
conn.close()


"""
    ("Survivor annuity percentage", "0.5"),
    ("Plan eligibility for emlpoyees", "All Employees"),
    ("Eligibility to participate after termination of employement", "Date reemployement"),
    ("Notice of withdrawal", "30 Days"),
    ("Forfeiture allocation percentage", "1"),
    ("Distribution eligibility for terminated employees", "Yes"),
    ("Distribution eligibility for active employees", "59.5 years"),
    ("Eligibility of withdrawal of entire individual account during service", "5 years"),
    ("Maximum withdrawal amount for an active particiant without minimum service", "India"),
    ("Alice", "India"),
    ("Bob", "USA"),
    ("Charlie", "India"),
    ("Alice", "India"),
    ("Bob", "USA"),
    ("Charlie", "India"),
    ("Alice", "India"),
    ("Bob", "USA"),
    ("Charlie", "India"),
    ("Bob", "USA"),
    ("Charlie", "India"),
"""