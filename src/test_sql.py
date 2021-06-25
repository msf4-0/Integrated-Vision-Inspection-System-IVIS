# streamlit_app.py

# import streamlit as st
# import psycopg2

# # Initialize connection.
# # Uses st.cache to only run once.


# @st.cache(allow_output_mutation=True, hash_funcs={"_thread.RLock": lambda _: None})
# def init_connection():
#     return psycopg2.connect(**st.secrets["postgres"])


# conn = init_connection()

# # Perform query.
# # Uses st.cache to only rerun when the query changes or after 10 min.


# @st.cache(ttl=600)
# def run_query(query):
#     with conn.cursor() as cur:
#         cur.execute(query)
#         return cur.fetchall()


# rows = run_query("SELECT * from playground;")

# # Print results.
# for row in rows:
#     st.write(row)

# %% ADD TABLE

import psycopg2
from passlib.hash import bcrypt, argon2, bcrypt_sha256
from time import perf_counter


conn = psycopg2.connect(
    "host=localhost port=5432 dbname=eye user=shrdc password=shrdc")

# cur = conn.cursor()

# Create new table to store user credentials
create_username_table = "CREATE TABLE Login (id serial PRIMARY KEY, username text, psd text, email text);"
with conn:
    with conn.cursor() as cur:
        cur.execute(create_username_table)
        conn.commit()

# %% CREATE
import psycopg2
from passlib.hash import bcrypt, argon2, bcrypt_sha256
from time import perf_counter
import pandas as pd

conn = psycopg2.connect(
    "host=localhost port=5432 dbname=eye user=shrdc password=shrdc")
column_names = ["id", "username", "password", "email"]


def create_user():
    new_user = {}
    new_user["username"] = input("Username: ")
    new_user["password"] = argon2.hash(input("Password: "))
    print(f'password: {new_user["password"]}')
    new_user["email"] = input("Email: ")

    with conn:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO Login (username, psd,email) VALUES (%s, %s,%s);", (new_user["username"], new_user["password"], new_user["email"]))
            cur.execute("select * from Login;")
            conn.commit()
            rows = cur.fetchall()
            df = pd.DataFrame(rows, columns=column_names)
            df.set_index('id')
            # more options can be specified also
            print(df)
            print(rows)

    return new_user, rows


new_user, rows = create_user()  # Create New User

# with conn:
#     with conn.cursor() as cur:
#         cur.execute("select * from Login;")

#         cur.fetchone()
#         conn.commit()


conn.close()

# %% LOGIN
import psycopg2
from passlib.hash import bcrypt, argon2, bcrypt_sha256
from time import perf_counter

conn = psycopg2.connect(
    "host=localhost port=5432 dbname=eye user=shrdc password=shrdc")


def login():
    old_user = {}
    old_user["username"] = input("Username: ")
    old_user["password"] = input("Password: ")
    print(f'Login password: {old_user["password"]}')

    with conn:
        with conn.cursor() as cur:
            cur.execute("SELECT psd FROM Login where username=%s;",
                        [old_user["username"]])

            conn.commit()
            rows = cur.fetchall()
            print(rows)

    return old_user, rows


old_user, rows = login()  # Create New User

if rows is not None:
    password = rows[0][0]
    print(f"Retrieved password: {password}")
    print(argon2.verify(old_user["password"], password))



conn.close()


# %%
from passlib.hash import bcrypt, argon2, bcrypt_sha256
from time import perf_counter

password = b"How are you" * 10
time1 = perf_counter()
h = bcrypt.hash(password)
time2 = perf_counter()
print(f"Bcrypt:{h},Hashing Time taken: {time2-time1}\n")

time1 = perf_counter()
print(bcrypt.verify(password, h))
time2 = perf_counter()
print(f"Bcrypt:{h},Verify Time taken: {time2-time1}\n")

time1 = perf_counter()
h = argon2.hash(password)
time2 = perf_counter()
print(f"Argon2ID:{h},Hashing Time taken: {time2-time1}\n")

print(f"Argon2:{h}")
time1 = perf_counter()
print(argon2.verify(password, h))
time2 = perf_counter()
print(f"Argon2ID:{h},Verify Time taken: {time2-time1}\n")

time1 = perf_counter()
h = bcrypt_sha256.hash(password)
time2 = perf_counter()
print(f"Bcrypt-SHA256:{h},Hashing Time taken: {time2-time1}\n")

time1 = perf_counter()
print(bcrypt_sha256.verify(password, h))
time2 = perf_counter()
print(f"Bcrypt-SHA256:{h},Verify Time taken: {time2-time1}\n")

# %%
