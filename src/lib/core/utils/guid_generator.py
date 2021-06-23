"""
Title: GUID Generator
Date: 23/6/2021
Author: Chu Zhen Hao
Organisation: Malaysian Smart Factory 4.0 Team at Selangor Human Resource Development Centre (SHRDC)
"""
import random
import string
import uuid


def RandomAlphaNum():
    # get random password pf length 8 with letters, digits, and symbols
    characters = string.ascii_letters + string.digits + string.punctuation
    _code = ''.join(random.choice(characters) for i in range(10))
    return _code


def GuidGen():

    _code = uuid.uuid4()
    return _code
