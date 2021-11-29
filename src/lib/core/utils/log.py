"""
Title: Logger
Date: 5/7/2021
Author: Chu Zhen Hao
Organisation: Malaysian Smart Factory 4.0 Team at Selangor Human Resource Development Centre (SHRDC)
"""
import logging

DEBUG = True
if DEBUG:
    # added module name, function name, and also line number
    FORMAT = '[%(levelname)s] %(asctime)s - [%(module)s.%(funcName)s: %(lineno)d] %(message)s'
    LEVEL = logging.DEBUG
else:
    FORMAT = '[%(levelname)s] %(asctime)s - %(message)s'
    LEVEL = logging.INFO
DATEFMT = '%d-%b-%y %H:%M:%S'
LOG_FILE = 'test.log'

# create logger
logger = logging.getLogger(__name__)
# set log level for all handlers to debug
logger.setLevel(LEVEL)

# create formatter
formatter = logging.Formatter(FORMAT, datefmt=DATEFMT)

# create console handler and setup level & formatter
consoleHandler = logging.StreamHandler()
consoleHandler.setLevel(LEVEL)
consoleHandler.setFormatter(formatter)
logger.addHandler(consoleHandler)

# create file handler and setup for logger
# fileHandler = logging.FileHandler(LOG_FILE)
# fileHandler.setLevel(LEVEL)
# fileHandler.setFormatter(formatter)
# logger.addHandler(fileHandler)

# must set this to False to make sure that child logger does not propagate
#  its message to the root logger
logger.propagate = False

# It's not recommend to use the functions below as the logger will not be able to tell
# where the logger command was called, e.g. the `filename` will always be this script: log.py
# rather than the filename where the logger was called outside this script.


def std_log(msg):
    logger.info(msg)


def log_debug(msg):
    logger.debug(msg)


def log_info(msg):
    logger.info(msg)


def log_error(msg):
    logger.error(msg)


def log_warning(msg):
    logger.warning(msg)
