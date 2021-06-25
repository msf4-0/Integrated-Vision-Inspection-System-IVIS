import psycopg2
from passlib.hash import bcrypt, argon2, bcrypt_sha256
from time import perf_counter

conn = psycopg2.connect(
    "host=localhost port=5432 dbname=eye user=shrdc password=shrdc")


def create_user():
    new_user = {}
    new_user["username"] = input("Username: ")
    new_user["password"] = argon2.hash(input("Password: "))
    new_user["email"] = input("Email: ")

    with conn:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO Login (username, psd,email) VALUES (%s, %s,%s);", (new_user["username"], new_user["password"], new_user["email"]))
            cur.execute("select * from Login;")
            conn.commit()
            rows = cur.fetchall()
            print(rows)

    return new_user, rows


new_user, rows = create_user()  # Create New User

# with conn:
#     with conn.cursor() as cur:
#         cur.execute("select * from Login;")

#         cur.fetchone()
#         conn.commit()


cur.close()
conn.close()
