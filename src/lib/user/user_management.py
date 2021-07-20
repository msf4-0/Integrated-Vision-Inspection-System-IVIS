"""
Title: User Management
Date: 25/6/2021
Author: Chu Zhen Hao
Organisation: Malaysian Smart Factory 4.0 Team at Selangor Human Resource Development Centre (SHRDC)
"""

import sys
from pathlib import Path
from enum import IntEnum
SRC = Path(__file__).resolve().parents[2]  # ROOT folder -> ./src
LIB_PATH = SRC / "lib"
TEST_MODULE_PATH_PARENT = SRC / "test" / "test_page"

for path in sys.path:
    if str(LIB_PATH) not in sys.path:
        sys.path.insert(0, str(LIB_PATH))  # ./lib
    else:
        pass

    if str(TEST_MODULE_PATH_PARENT) not in sys.path:
        sys.path.insert(0, str(TEST_MODULE_PATH_PARENT))
    else:
        pass


import psycopg2
from passlib.hash import argon2
from core.utils.log import std_log  # logger

# ------------------TEMP
import streamlit as st
from streamlit import cli as stcli  # Add CLI so can run Python script directly
from streamlit import session_state as SessionState

# conn = psycopg2.connect(
#     "host=localhost port=5432 dbname=eye user=shrdc password=shrdc")


@st.cache(allow_output_mutation=True, hash_funcs={"_thread.RLock": lambda _: None})
def init_connection():
    return psycopg2.connect(**st.secrets["postgres"])


conn = init_connection()


class ACCOUNT_STATUS(IntEnum):  # User Status
    NEW = 0  # Pending account activation
    ACTIVE = 1  # Account activated
    LOCKED = 2  # Account locked
    LOGGED_IN = 3  # Account logged-in
    LOGGED_OUT = 4  # Account logged-out


# TODO: move to form_manager
def check_if_field_empty(new_user, field_placeholder, field_name):
    empty_fields = []
    # all_field_filled = all(new_user)
    # if not all_field_filled:  # IF there are blank fields, iterate and produce error message
    for key, value in new_user.items():
        if value == "":
            field_placeholder[key].error(
                f"Please do not leave {field_name[key]} field blank")
            empty_fields.append(key)

        else:
            pass

    return not empty_fields


def create_usertable(conn=conn):  # Create Table
    # create relation : user
    create_username_table = """CREATE TABLE IF NOT EXISTS user (
                                user_id INT GENERATED BY DEFAULT AS IDENTITY,
                                emp_id TEXT,
                                username TEXT NOT NULL,
                                first_name TEXT,
                                last_name TEXT,
                                email TEXT,
                                department TEXT,
                                position TEXT,
                                psd TEXT NOT NULL,
                                role VARCHAR (30),
                                created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                                updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                                last_activity TIMESTAMP WITH TIME ZONE,
                                status varchar(15) DEFAULT 'NEW' ,
                                PRIMARY KEY (use r_id,role)
                                            );"""
    with conn:
        with conn.cursor() as cur:
            cur.execute(create_username_table)
            conn.commit()

# Create User


def create_user(user, conn=conn):
    # def create_user(new_user):
    # new_user = {}

    create_usertable(conn)  # create user table if does not exist
    # Employee Corporate Details
    std_log("User Entry")
    # new_user["emp_id"] = input("Employee ID: ")
    # new_user["first_name"] = input("First Name: ")
    # new_user["last_name"] = input("Last Name: ")
    # new_user["email"] = input("Employee Email: ")
    # new_user["department"] = input("Department: ")
    # new_user["position"] = input("Position: ")

    # # Account Corporate Details
    # new_user["username"] = input("Username: ")
    # new_user["role"] = input("Role: ")
    psd = argon2.hash(user["psd"])
    std_log(f'password: {user["psd"]}')

    with conn:
        with conn.cursor() as cur:
            cur.execute(""" INSERT INTO user (emp_id,first_name,last_name,email,department,position,username,role, psd)
                            VALUES (%s,%s, %s,%s,%s,%s,%s,%s,%s)
                            RETURNING user_id,username,created_at as create_user;""",
                        [user["emp_id"], user["first_name"], user["last_name"], user["email"], user["department"], user["position"],
                         user["username"], user["role"], psd])
            # cur.execute("select * from Login;")

            conn.commit()
            user_create = cur.fetchone()
            std_log(user_create)
            user = {}


# >>>> User Login
class UserLogin:
    def __init__(self) -> None:

        # TODO:Temporary
        self.id = None
        self.username = None
        # self.first_name
        # self.last_name
        # self.email
        # self.department
        # self.position
        self.psd = None
        # self.role
        self.status = None
        self.session_id = None
        self.attempts = 0

    def user_verification(self, user, conn=conn):
        """Verify user credentials

        Args:
            user (Dict): Contains username and password input
            conn (connect object, optional): psycopg2.connect object. Defaults to conn.

        Returns:
            Boolean: Login Fail/Pass
        """

        # --Testing
        # user = {}
        # user["username"] = input("Username: ")
        # user["psd"] = input("Password: ")
        std_log(f"Login password: {user['psd']}")
        # -----Testing

        with conn:  # open connections to Database
            with conn.cursor() as cur:
                # QUERY user id, hashed password and account status
                cur.execute("SELECT id,psd,status FROM user WHERE username=%s;",
                            [user["username"]])

                conn.commit()  # commit SELECT query password
                user_exist = cur.fetchone()
                std_log(user_exist[0])

        if user_exist is not None:  # if user exists
            self.id = user_exist[0]
            self.psd = user_exist[1]
            self.status = user_exist[2]
            self.username = user['username']
            # std_log(f"Retrieved password: {psd}")

            # compare password with hash
            verification = argon2.verify(user.pop('psd'), self.psd)
            delattr(self, 'psd')  # REMOVE password
            self.attempts += 1  # INCREMENT login attempts counter

            # LOCK account if more than 3 password attempts
            if self.attempts > 3:
                self.update_status("LOCKED")
            return verification
            # returns True is MATCH
            # returns False if NOT MATCH

        elif user_exist is None:
            # User does not exist in database
            # return False if user does not exist in database
            return False

    def update_status(self, status):
        # >>>> Update account status
        self.status = status
        with conn:  # open connections to Database
            with conn.cursor() as cur:
                cur.execute("""UPDATE user 
                                SET status = %s
                                WHERE id = %s;""", [self.status, self.id])

                conn.commit()

    def update_psd(self):
        self.psd = argon2.hash(self.psd)
        with conn:
            with conn.cursor() as cur:
                cur.execute("""UPDATE user 
                            SET psd = %s,
                                status = %s
                            WHERE username = %s;""",
                            [self.psd, self.status, self.username])

                conn.commit()

                delattr(self, 'psd')  # REMOVE password


# user_create = create_user()  # Create New User

# user_exist, user_entry_flag = user_login()  # Create New User


# conn.close()