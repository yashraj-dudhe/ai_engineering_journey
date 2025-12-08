import sqlite3

conn = sqlite3.connect(":memory:")
cursor = conn.cursor()

cursor.execute("create table employees (name TEXT, department TEXT, salary INT)")
data = [
    ("Alice","Sales",5000),
    ("Bob", "Sales", 4000),
    ("Charlie", "Sales", 3000), # Should be cut
    ("David", "Eng", 9000),
    ("Eve", "Eng", 8000),
    ("Frank", "Eng", 7000),     # Should be cut
    ("Grace", "HR", 4500)
]

cursor.executemany("INSERT INTO employees values(?,?,?)",data)
conn.commit()

query = """
select name,department,salary,rank
from(
    select 
        name, 
        department,
        salary,
        RANK() OVER (PARTITION BY department order by salary desc) as rank
    from employees
)
where rank<=2
"""
print(f"{"name":<10} {"Dept":<10} {"Salary":<10} {"rank":<10}")
print("-"*40)


try:
    cursor.execute(query)
    for row in cursor.fetchall():
        print(f"{row[0]:<10} {row[1]:<10} {row[2]:<10} {row[3]:<5}")

except sqlite3.OperationalError as e:
    print(f"sql error {e}")
conn.close()

