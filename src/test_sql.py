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

# %%

import psycopg2
from passlib.hash import bcrypt, argon2, bcrypt_sha256
from time import perf_counter

conn = psycopg2.connect(
    "host=localhost port=5432 dbname=eye user=shrdc password=shrdc")

cur = conn.cursor()

cur.execute(
    "CREATE TABLE test (id serial PRIMARY KEY, num integer, data varchar);")

cur.execute("INSERT INTO test (num, data) VALUES (%s, %s)",
            ...      (100, "abc'def"))

cur.execute("select * from test;")
cur.fetchone()

conn.commit()

cur.close()
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
