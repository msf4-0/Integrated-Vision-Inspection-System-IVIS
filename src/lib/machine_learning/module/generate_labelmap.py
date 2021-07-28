"""
Title: Training Management
Date: 23/7/2021
Author: Chu Zhen Hao
Organisation: Malaysian Smart Factory 4.0 Team at Selangor Human Resource Development Centre (SHRDC)
"""

import sys
from pathlib import Path
from typing import List

# >>>>>>>>>>>>>>>>>>>>>>TEMP>>>>>>>>>>>>>>>>>>>>>>>>

SRC = Path(__file__).resolve().parents[3]  # ROOT folder -> ./src
LIB_PATH = SRC / "lib"
DATA_DIR = Path.home() / '.local/share/integrated-vision-inspection-system/app_media'

if str(LIB_PATH) not in sys.path:
    sys.path.insert(0, str(LIB_PATH))  # ./lib
else:
    pass

# >>>> User-defined Modules >>>>
# from path_desc import chdir_root
# from core.utils.log import log_info, log_error  # logger
# from data_manager.database_manager import init_connection, db_fetchone, db_no_fetch, db_fetchall
# from core.utils.file_handler import bytes_divisor, create_folder_if_not_exist
# from core.utils.helper import split_string, join_string
# <<<<<<<<<<<<<<<<<<<<<<TEMP<<<<<<<<<<<<<<<<<<<<<<<

from object_detection.protos.string_int_label_map_pb2 import StringIntLabelMap, StringIntLabelMapItem
from google.protobuf import text_format


def label_map_to_text(classes: List, start=1) -> bytes:
    # 'id' must start from 1
    msg = StringIntLabelMap()
    for id, name in enumerate(classes, start=start):
        msg.item.append(StringIntLabelMapItem(
            id=id, name=name))

    text = str(text_format.MessageToBytes(msg, as_utf8=True), 'utf-8')
    return text


def label_map_to_pbtxt(labelmap_text: bytes, filepath: Path) -> None:
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with filepath.open("w", encoding="utf-8") as f:
        f.write(labelmap_text)


if __name__ == '__main__':
    txt = label_map_to_text(['Hello', 'World', 'Car', 'Plane', 'shrdc'])
    print(txt)
    label_map_to_pbtxt(txt, 'label_map.pbtxt')
