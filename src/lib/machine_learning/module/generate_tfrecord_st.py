""" Sample TensorFlow XML-to-TFRecord converter

usage: generate_tfrecord.py [-h] [-x XML_DIR] [-l LABELS_PATH] [-o OUTPUT_PATH] [-i IMAGE_DIR] [-c CSV_PATH]

optional arguments:
  -h, --help            show this help message and exit
  -x XML_DIR, --xml_dir XML_DIR
                        Path to the folder where the input .xml files are stored.
  -l LABELS_PATH, --labels_path LABELS_PATH
                        Path to the labels (.pbtxt) file.
  -o OUTPUT_PATH, --output_path OUTPUT_PATH
                        Path of output TFRecord (.record) file.
  -i IMAGE_DIR, --image_dir IMAGE_DIR
                        Path to the folder where the input image files are stored. Defaults to the same directory as XML_DIR.
  -c CSV_PATH, --csv_path CSV_PATH
                        Path of output .csv file. If none provided, then no file will be written.
"""

import os
import glob
import pandas as pd
import io
import xml.etree.ElementTree as ET
import argparse
import sys
from pathlib import Path

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # Suppress TensorFlow logging (1)
import tensorflow.compat.v1 as tf  # must use compat.v1
from PIL import Image
from object_detection.utils import dataset_util, label_map_util
from collections import namedtuple

SRC = Path(__file__).resolve().parents[3]  # filepath -> ./src
LIB_PATH = SRC / "lib"

# added this to use `xml_to_csv`
if str(LIB_PATH) not in sys.path:
    sys.path.insert(0, str(LIB_PATH))  # ./lib

from machine_learning.module.xml_parser import xml_to_csv

# Initiate argument parser
parser = argparse.ArgumentParser(
    description="Sample TensorFlow XML-to-TFRecord converter"
)
parser.add_argument(
    "-x",
    "--xml_dir",
    help="Path to the folder where the input .xml files are stored.",
    type=str,
)
parser.add_argument(
    "-l", "--labels_path", help="Path to the labels (.pbtxt) file.", type=str
)
parser.add_argument(
    "-o", "--output_path", help="Path of output TFRecord (.record) file.", type=str
)
parser.add_argument(
    "-i",
    "--image_dir",
    help="Path to the folder where the input image files are stored. "
    "Defaults to the same directory as XML_DIR.",
    type=str,
    default=None,
)
parser.add_argument(
    "-e",
    "--image_ext",
    # LABEL STUDIO outputs XML filenames without extensions included,
    # therefore the script needs to append the file extension at the end
    help="Extension of the images used, excluding the 'dot'. REQUIRED FOR LABEL STUDIO",
    type=str,
    nargs="*",
    default=None,
)
parser.add_argument(
    "-c",
    "--csv_path",
    help="Path of output .csv file. If none provided, then no file will be " "written.",
    type=str,
    default=None,
)

args = parser.parse_args()

if args.image_dir is None:
    args.image_dir = args.xml_dir

label_map = label_map_util.load_labelmap(args.labels_path)
label_map_dict = label_map_util.get_label_map_dict(label_map)

error_images = []


def class_text_to_int(row_label):
    return label_map_dict[row_label]


def split(df, group):
    data = namedtuple("data", ["filename", "object"])
    gb = df.groupby(group)
    return [
        data(filename, gb.get_group(x))
        for filename, x in zip(gb.groups.keys(), gb.groups)
    ]


def create_tf_example(group, path):
    image_path = os.path.join(path, group.filename)

    if not os.path.exists(image_path):
        if args.image_ext:
            for image_ext in args.image_ext:
                new_image_path = f"{image_path}.{image_ext}"
                if os.path.exists(new_image_path):
                    image_path = new_image_path
                    break
    if not os.path.exists(image_path):
        raise FileNotFoundError(
            f"Image {image_path} is not found. If you are using Label Studio, "
            'try pass in -e <IMAGE_EXTENSIONS> as an argument (e.g. -e "jpg png") '
            "to append the image extension at the end as the XML file exported "
            "from Label Studio did not include file extension in the filename."
        )

    with tf.io.gfile.GFile(image_path, "rb") as fid:
        encoded_jpg = fid.read()

    # WARNING, THESE FOLLOWING 3 LINES MIGHT READ THE IMAGE IN THE ROTATED DIMENSIONS
    # which will cause the width and height to be inverted!
    # encoded_jpg_io = io.BytesIO(encoded_jpg)
    # image = Image.open(encoded_jpg_io)
    # width, height = image.size

    filename = group.filename.encode("utf8")
    image_format = b"jpg"
    xmins = []
    xmaxs = []
    ymins = []
    ymaxs = []
    classes_text = []
    classes = []
    error = False

    for row in group.object.itertuples():
        width, height = row.width, row.height
        xmin = row.xmin / width
        xmax = row.xmax / width
        ymin = row.ymin / height
        ymax = row.ymax / height
        # sanity checks to make sure the annotations are correct
        if xmin < 0:
            error = True
            print(f"[WARNING] Error with {filename.decode()}, xmin {xmin} < 0")
            print(f"\t row.xmin = {row.xmin} ; width = {width}")
        if xmax > 1:
            error = True
            print(f"[WARNING] Error with {filename.decode()}, xmax {xmax} > 1")
            print(f"\t row.xmax = {row.xmax} ; width = {width}")
        if ymin < 0:
            error = True
            print(f"[WARNING] Error with {filename.decode()}, ymin {ymin} < 0")
            print(f"\t row.ymin = {row.ymin} ; height = {height}")
        if ymax > 1:
            error = True
            print(f"[WARNING] Error with {filename.decode()}, ymax {ymax} > 1")
            print(f"\t row.ymax = {row.ymax} ; height = {height}")

        if error:
            error_images.append(image_path)
            return "error"

        xmins.append(xmin)
        xmaxs.append(xmax)
        ymins.append(ymin)
        ymaxs.append(ymax)
        classes_text.append(row.classname.encode("utf8"))
        classes.append(class_text_to_int(row.classname))

    tf_example = tf.train.Example(
        features=tf.train.Features(
            feature={
                "image/height": dataset_util.int64_feature(height),
                "image/width": dataset_util.int64_feature(width),
                "image/filename": dataset_util.bytes_feature(filename),
                "image/source_id": dataset_util.bytes_feature(filename),
                "image/encoded": dataset_util.bytes_feature(encoded_jpg),
                "image/format": dataset_util.bytes_feature(image_format),
                "image/object/bbox/xmin": dataset_util.float_list_feature(xmins),
                "image/object/bbox/xmax": dataset_util.float_list_feature(xmaxs),
                "image/object/bbox/ymin": dataset_util.float_list_feature(ymins),
                "image/object/bbox/ymax": dataset_util.float_list_feature(ymaxs),
                "image/object/class/text": dataset_util.bytes_list_feature(
                    classes_text
                ),
                "image/object/class/label": dataset_util.int64_list_feature(classes),
            }
        )
    )

    return tf_example


def main(_):

    writer = tf.python_io.TFRecordWriter(args.output_path)
    path = os.path.join(args.image_dir)
    examples = xml_to_csv(args.xml_dir)
    grouped = split(examples, "filename")

    for group in grouped:
        tf_example = create_tf_example(group, path)
        if tf_example == "error":
            continue
        writer.write(tf_example.SerializeToString())
    writer.close()
    print("Successfully created the TFRecord file: {}".format(args.output_path))
    if args.csv_path is not None:
        examples.to_csv(args.csv_path, index=None)
        print("Successfully created the CSV file: {}".format(args.csv_path))

    if error_images:
        # NOTE: originally asks for user input, but now automatically remove bad images
        print(f"\n[WARNING] {len(error_images)} images found with error")
        # delete = input(
        #     "Recommended to remove them and re-generate the TFrecords again.\n"
        #     "The generated TFRecord files already skipped the images with error.\n"
        #     "So you may choose to not remove them and proceed without error.\n"
        #     "Do you want to remove them? [y | n]\n"
        # )
        # if delete:
        print(f"[INFO] Removing {len(error_images)} images ...")
        for p in error_images:
            os.remove(p)
            os.remove(os.path.splitext(p)[0] + ".xml")
        print("Done.")


if __name__ == "__main__":
    tf.app.run()
