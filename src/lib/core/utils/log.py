"""
Title: Logger
Date: 5/7/2021
Author: Chu Zhen Hao
Organisation: Malaysian Smart Factory 4.0 Team at Selangor Human Resource Development Centre (SHRDC)
"""
import logging
import sys

FORMAT = '[%(levelname)s] %(asctime)s - %(message)s'
DATEFMT = '%d-%b-%y %H:%M:%S'

# logging.basicConfig(filename='test.log',filemode='w',format=FORMAT, level=logging.INFO)
logging.basicConfig(format=FORMAT, level=logging.INFO,
                    stream=sys.stdout, datefmt=DATEFMT)

log = logging.getLogger()


def std_log(msg):
    log.info(msg)


def log_debug(msg):
    log.debug(msg)


def log_info(msg):
    log.info(msg)


def log_error(msg):
    log.error(msg)


def log_warning(msg):
    log.warning(msg)
