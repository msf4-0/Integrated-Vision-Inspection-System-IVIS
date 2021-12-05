# You may also refer to the notebook in the same directory as this script for reference

import json
import sys
from pathlib import Path

from bs4 import BeautifulSoup
import requests
import pandas as pd
import toml
import psycopg2
from psycopg2.extras import NamedTupleCursor

SRC = Path(__file__).resolve().parents[3]  # ROOT folder -> ./src
LIB_PATH = SRC / "lib"
if str(LIB_PATH) not in sys.path:
    sys.path.insert(0, str(LIB_PATH))  # ./lib

from path_desc import TFOD_MODELS_TABLE_PATH, CLASSIF_MODELS_NAME_PATH, SEGMENT_MODELS_TABLE_PATH
from core.utils.log import logger


def connect_db(**context):
    # load from toml file if context is not provided
    if not context:
        # assuming you run this script from the root directory of this project
        toml_filepath = ".streamlit/secrets.toml"
        parsed = toml.load(open(toml_filepath))

        # # Connect to PostgreSQL db
        conn = psycopg2.connect(**parsed['postgres'])
    else:
        conn = psycopg2.connect(**context)

    # Check and verify connection
    with conn.cursor(cursor_factory=NamedTupleCursor) as cur:
        cur.execute('SELECT version();')
        conn.commit()

        # display the PostgreSQL database server version
        db_version = cur.fetchone().version

    print(f"PostgreSQL database version: {db_version}")
    print(f"PostgreSQL connection status: {conn.info.status}")
    print(
        f"You are connected to database '{conn.info.dbname}' as user '{conn.info.user}' on host '{conn.info.host}' at port '{conn.info.port}'.")
    return conn

    # ************************** DB Functions **************************


def db_fetchone(sql_message, query_vars=None, conn=None, return_output=False, fetch_col_name=False, return_dict=False):
    assert conn is not None, "Please connect to db and pass in for faster computation."

    with conn.cursor(cursor_factory=NamedTupleCursor) as cur:
        try:
            if query_vars:
                cur.execute(sql_message, query_vars)
            else:
                cur.execute(sql_message)

            conn.commit()
            if not return_output:
                return
            return_one = cur.fetchone()  # return tuple

            if return_dict:
                # Convert results to pure Python dictionary
                return_one = dict(return_one)

            if fetch_col_name:
                # Obtain Column names from query
                column_names = [desc[0] for desc in cur.description]
                return return_one, column_names
            else:
                return return_one
        except psycopg2.Error as e:
            conn.rollback()
            print(e)


def check_if_pretrained_models_exist(conn):
    """Check whether the CSV files exists, and also whether data is stored in database"""
    sql_query = """
        SELECT id FROM models WHERE model_type_id = 1;
    """
    record = db_fetchone(sql_query, conn=conn, return_output=True)
    if TFOD_MODELS_TABLE_PATH.exists() and \
            CLASSIF_MODELS_NAME_PATH.exists() and \
            SEGMENT_MODELS_TABLE_PATH.exists() and \
            record:
        return True
    return False


def insert_to_db(new_df, conn):
    sql_query = """
                INSERT INTO public.models (
                    name,
                    metrics,
                    model_type_id,
                    framework_id,
                    deployment_id
                    )
                OVERRIDING SYSTEM VALUE
                VALUES (
                    %s,
                    %s,
                    %s,
                    %s,
                    %s
                    )
    """
    for row in new_df.values:
        query_vars = row.tolist()
        db_fetchone(sql_query, query_vars, conn=conn)


def scrape_setup_model_details(conn):
    # ## Delete existing pretrained_models record

    # Delete existing pretrained models before inserting
    # `model_type_id` = 1 for pretrained_models type
    # NOTE: This is actually only needed for debugging, real installation only needs
    # to run this once, hence does not require deleting previously stored pretrained_models
    sql_query = """
    DELETE FROM public.models WHERE model_type_id = 1
    """
    db_fetchone(sql_query, conn=conn)

    # reset the ID to start from latest ID + 1
    table_name = 'models'
    # use this or use ALTER SEQUENCE. False to ensure to take next value
    # https://coderedirect.com/questions/335605/postgresql-using-subqueries-with-alter-sequence-expressions
    sql_msg = f"""
        SELECT SETVAL('{table_name}_id_seq', (SELECT MAX(id) + 1 FROM {table_name}), false)
    """
    db_fetchone(sql_msg, conn=conn)

    # ************************ Scraping TFOD Model Zoo ************************
    logger.info("Scraping TensorFlow Models information")
    link = "https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md"
    df = pd.read_html(link)[0]

    data = requests.get(link)
    soup = BeautifulSoup(data.content, "html.parser")
    model_links = soup.select("td a[href]")
    model_links = [link['href'] for link in model_links]
    df['model_links'] = model_links
    df.columns = ["Model Name", "Speed (ms)", "COCO mAP",
                  "Outputs", "model_links"]

    # save to CSV before formatting for database
    df.to_csv(TFOD_MODELS_TABLE_PATH, index=False)

    # Modifying to insert into database
    # Need columns: id, name, description, metrics, model_path, model_type_id,
    #  framework_id, deployment_id
    new_df = df[['Model Name']].copy()
    new_df['metrics'] = json.dumps({'metrics': 'mAP'})
    new_df['model_type_id'] = 1  # for pretrained_model type
    new_df['framework_id'] = 1  # for Tensorflow type
    new_df['deployment_id'] = 2  # for Object Detection

    # ## Insert to DB
    insert_to_db(new_df, conn)

    # *************** Scraping Keras pretrained model functions ***************

    logger.info("Scraping Keras Pretrained Classification Models information")
    URL = "https://www.tensorflow.org/api_docs/python/tf/keras/applications"
    data = requests.get(URL)
    soup = BeautifulSoup(data.text, 'html.parser')

    names = soup.select("code")
    model_names = []
    # NOTE: This might not work in the future depending on how TensorFlow
    # structure their documentation
    for i in names:
        if '(...)' in i.text:
            link = i.text.replace('(...)', '')
            model_names.append(link)

    df = pd.DataFrame(model_names, columns=['Model Name'])
    # save to CSV before formatting for database
    df.to_csv(CLASSIF_MODELS_NAME_PATH, index=False)

    df['metrics'] = json.dumps({"metrics": ["Accuracy", "F1 Score"]})
    df['model_type_id'] = 1  # for pretrained_model type
    df['framework_id'] = 1  # for Tensorflow type
    df['deployment_id'] = 1  # for image classification

    # ## Insert to DB
    insert_to_db(df, conn)

    # *************** Scraping keras-unet-collections models ***************

    logger.info("Scraping Keras U-net Models information")
    URL = "https://github.com/yingkaisha/keras-unet-collection"
    data = requests.get(URL)
    soup = BeautifulSoup(data.text, 'html.parser')

    tables = soup.select("table")
    for table in tables:
        if 'models' in table.th.text:
            models_table = table
        if 'losses' in table.th.text:
            losses_table = table

    model_links = models_table.select("td a")
    model_links = [i['href'] for i in model_links]

    model_df = pd.read_html(models_table.prettify())[0]
    model_df['links'] = model_links
    model_df.columns = ["model_func", "Model Name", "Reference", "links"]
    # dropping the last two models built with Transformers as they don't work for new NumPy version,
    # refer to the repo for more details https://github.com/yingkaisha/keras-unet-collection
    model_df.drop(model_df.index[-2:], inplace=True)
    model_df.to_csv(SEGMENT_MODELS_TABLE_PATH, index=False)

    # # for the available loss functions
    loss_df = pd.read_html(losses_table.prettify())[0]

    # dropping experimental loss functions
    experimental_row_idxs = loss_df.loc[loss_df['Name'].str.contains(
        'experimental')].index
    loss_df = loss_df.drop(experimental_row_idxs).reset_index(drop=True)

    loss_links = losses_table.select("td a")
    loss_links = [i['href'] for i in loss_links]

    loss_df['links'] = loss_links
    loss_df.columns = ['loss_func', 'Name', 'Reference', 'links']

    df = model_df[['Model Name']].copy()
    df['metrics'] = json.dumps(
        {"metrics": "mAP", "losses": loss_df.loss_func.tolist()})
    df['model_type_id'] = 1  # for pretrained_model type
    df['framework_id'] = 1  # for Tensorflow type
    df['deployment_id'] = 3  # for image segmentation

    # ## Insert to DB

    insert_to_db(df, conn)


def main():
    # connect to DB using .streamlit/secrets.toml
    conn = connect_db()
    # then scrape all the pretrained model details and update our DB
    scrape_setup_model_details(conn)


if __name__ == '__main__':
    main()
