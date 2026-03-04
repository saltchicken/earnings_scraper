import sqlite3


def print_table_schema(
    db_name="sp500_financials.db", table_name="quarterly_financials"
):
    try:
        with sqlite3.connect(db_name) as conn:
            cursor = conn.cursor()

            # Method 1: Print the original CREATE TABLE statement
            cursor.execute(
                "SELECT sql FROM sqlite_master WHERE type='table' AND name=?",
                (table_name,),
            )
            result = cursor.fetchone()

            if result and result[0]:
                print(f"--- Schema for table '{table_name}' ---")
                print(result[0])
            else:
                print(f"Table '{table_name}' not found in database '{db_name}'.")
                return

            print("\n--- Detailed Column Info ---")
            print(
                f"{'CID':<5} | {'Name':<20} | {'Type':<10} | {'Not Null':<10} | {'Primary Key'}"
            )
            print("-" * 65)

            # Method 2: Use PRAGMA to get structured info about each column
            cursor.execute(f"PRAGMA table_info({table_name})")
            columns = cursor.fetchall()

            for col in columns:
                cid, name, ctype, notnull, default, pk = col
                print(
                    f"{cid:<5} | {name:<20} | {ctype:<10} | {bool(notnull):<10} | {bool(pk)}"
                )

    except sqlite3.OperationalError as e:
        print(f"Error accessing database: {e}")


if __name__ == "__main__":
    print_table_schema()
