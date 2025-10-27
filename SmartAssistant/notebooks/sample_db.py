# setup_db.py  (run once to create example DB)
import sqlite3

conn = sqlite3.connect(r"D:/GenAI-Practice/AgenticAI-Projects/SmartAssistant/app/data/uploads/system_details.db")
cur = conn.cursor()

table_query = "CREATE TABLE IF NOT EXISTS system_info (id INTEGER PRIMARY KEY,Provision TEXT,System_Data TEXT)"

cur.execute(table_query)

print('Table created successfully')

cur.executemany("""
INSERT INTO system_info (Provision, System_Data)
VALUES (?, ?)
""", [
    ("Survivor annuity percentage", "0.5"),
    ("Plan eligibility for emlpoyees", "All Employees"),
    ("Eligibility to participate after termination of employement", "Date reemployement"),
    ("Notice of withdrawal", "30 Days"),
    ("Forfeiture allocation percentage", "1"),
    ("Distribution eligibility for terminated employees", "Yes"),
    ("Distribution eligibility for active employees", "59.5 years"),
    ("Eligibility of withdrawal of entire individual account during service", "5 years"),
    ("Maximum withdrawal amount for an active particiant without minimum service", "Employer contributions for 2 full plan years"),
    ("Hardship Withdrawal Eligibility", "Yes"),
    ("Spousal consent for beneficary change in case of separated participants", "Not Required"),
    ("Spousal consent for beneficary change in case of married participants", "Required"),
    ("Annuity contracts are transferable", "No"),
    ("Notice for denial of a claim", "90 days"),
    ("Notice for denial of a claim for person with disability", "45 days"),
    ("Review request for denial of a claim", "60 days"),
    ("Review request for denial of a claim for person with disability", "180 days"),
    ("Interest Free Loan", "No"),
    ("Bob", "Present value of the vested portion of participant, individual account"),
    ("Spousal consent for loan available", "Yes"),
])

print('Table rows added successfully')

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