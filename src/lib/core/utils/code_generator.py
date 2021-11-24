"""
Title: GUID Generator
Date: 23/6/2021
Author: Chu Zhen Hao
Organisation: Malaysian Smart Factory 4.0 Team at Selangor Human Resource Development Centre (SHRDC)
"""
import random
import string
import uuid
import secrets
from passlib.hash import argon2


def RandomAlphaNum():
    # get random password pf length 8 with letters, digits, and symbols
    characters = string.ascii_letters + string.digits + string.punctuation
    _code = ''.join(random.choice(characters) for i in range(10))
    return _code


def GuidGen():

    _code = uuid.uuid4()
    return _code


def get_random_string(length=12, allowed_chars=(
    'abcdefghijklmnopqrstuvwxyz'
    'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
)):
    """
    Return a securely generated random string.

    The bit length of the returned value can be calculated with the formula:
        log_2(len(allowed_chars)^length)

    For example, with default `allowed_chars` (26+26+10), this gives:
      * length: 12, bit length =~ 71 bits
      * length: 22, bit length =~ 131 bits
    """

    return ''.join(secrets.choice(allowed_chars) for i in range(length))


def make_random_password(length=12,
                         allowed_chars='abcdefghjkmnpqrstuvwxyz'
                         'ABCDEFGHJKLMNPQRSTUVWXYZ'
                         '23456789'):
    """
    Generate a random password with the given length and given
    allowed_chars. The default value of allowed_chars does not have "I" or
    "O" or letters and digits that look similar -- just to avoid confusion.
    """
    return get_random_string(length, allowed_chars)
