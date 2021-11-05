"""
General Parser:
1. JSON
2. YAML
Date: 21/6/2021
Author: Chu Zhen Hao
Organisation: Malaysian Smart Factory 4.0 Team at Selangor Human Resource Development Centre (SHRDC)

"""
import base64
import io
import json
import os
import shutil
import sys
import tarfile
import urllib
from glob import glob, iglob
from pathlib import Path
from typing import IO, Dict, List, Tuple, Union
from zipfile import ZipFile
from copy import deepcopy
import streamlit as st
from streamlit.uploaded_file_manager import UploadedFile
import yaml

# >>>>>>>>>>>>>>>>>>>>>>TEMP>>>>>>>>>>>>>>>>>>>>>>>>

SRC = Path(__file__).resolve().parents[3]  # ROOT folder -> ./src
LIB_PATH = SRC / "lib"

if str(LIB_PATH) not in sys.path:
    sys.path.insert(0, str(LIB_PATH))  # ./lib
else:
    pass

# >>>> User-defined Modules >>>>

from core.utils.helper import remove_suffix
from core.utils.log import logger  # logger
from path_desc import chdir_root, get_temp_dir

# <<<<<<<<<<<<<<<<<<<<<<TEMP<<<<<<<<<<<<<<<<<<<<<<<

SUPPORTED_ARCHIVE_EXT = ['.zip', '.gz', '.bz2', '.xz']
IMAGE_EXTENSIONS = ['.jpg', '.png', '.jpeg']

# ************************** DEPRECATED **************************


def bytes_divisor(value: Union[int, float], power: int = 1) -> Union[int, float]:
    """Convert bytes size

    Args:
        value (Union[int,float]): bytes size value
        power (int, optional): Conversion power (2^power bytes). Defaults to 1.

    Returns:
        Union[int,float]: Converted bytes size value
    """
    byte_unit = 1024
    converted_byte = value * (byte_unit**(power))
    return converted_byte


#-----------JSON PARSER-------------------#


#-----------YAML PARSER-------------------#

#-----------FILE HANDLER------------------#
def file_open(path):
    """general file opener"""
    with open(path, mode='r') as file:
        file = file.read()
    return file


def multi_file_open(path):
    """general multi-file opener"""

    file_list = []
    with open(path, mode='r') as files:
        file_list = files
    return file_list


def delete_file_directory(path: Union[Path, str]):

    if path:
        # convert path string into path-like object
        path = Path(path) if isinstance(path, str) else path
        logger.info(path)

        if path.is_dir():

            # remove all contents if path is a folder directory
            try:
                shutil.rmtree(path)
                logger.info(f"Successfully deleted {path.name} folder")
                return True
            except Exception as e:
                error_msg = f"{e}: Failed to remove directory"
                return False
        elif path.is_file():

            # remove file if path is filepath

            try:
                path.unlink()
                logger.info(f"Successfully deleted {path.name}")
                return True

            except FileNotFoundError as e:
                error_msg = f"{e}: {path.name} does not exists"
                logger.error(error_msg)
                return False
        else:
            not_path_error_msg = f"Invalid path given......"
            logger.error(not_path_error_msg)
            return False


def file_search(path=str(Path.home())):
    """
    File Search
    - Recursive true to search sub-directories with " ** "
    - Use wild-card for non-specific files ('*.extension')
    - CWD/*/*
    """
    file_list = []
    file_list = glob(pathname=path)
    # for file in glob(pathname=path):
    #     file_list.append(file)

    return file_list


def i_file_search(path=str(Path.home()), recursive=False):
    """Iterative File Search

    Args:
        path (str, optional): Path to file or folder. Defaults to str(Path.home()).

    Returns:
        List: List of files in that directory
    """

    file_list = []
    for file in iglob(pathname=path, recursive=recursive):
        file_list.append(file)
    return file_list

#-----------FILE STORAGE-------------------#


def create_folder_if_not_exist(path: Path) -> None:
    path = Path(path)
    if not path.is_dir():
        try:
            path.mkdir(parents=True)
            logger.info(f"Created Directory at {str(path)}")
        except FileExistsError as e:
            logger.info(f"Directory already exists: {e}")
    else:
        logger.info(f"Directory already exist")
        pass


def move_file(src: Union[str, Path], dst: Union[str, Path]) -> bool:
    """Move files from src to dst

    Args:
        src (Union[str, Path]): Source path
        dst (Union[str, Path]): Destination path

    Returns:
        bool: True if successfully moved all files

    Raises:
        AssertationError: "Failed to moved all files"
    """

    # For assertation of number of contents moved
    initial_foldersize = len([x for x in src.rglob('*')])

    # Iterate to all contents in `dst`
    for file in src.iterdir():
        dest_path = dst / file.relative_to(src)  # get destination file path

        # returns file path to destination for each file moved
        dst_return = shutil.move(src=str(file),
                                 dst=str(dest_path))
        logger.info(f"Moved {file.name} to {dst_return}")

    assert len([x for x in Path(dst).rglob('*')]
               ) == initial_foldersize, "Failed to moved all files"

    return True


def write_file(filename: str, dst: Union[str, Path], file: str = None, byte_stream=None):
    """Write files to disk

    Args:
        filename (str): Name of file with extension
        dst (Union[str,Path]): Destination on the disk
        file (str, optional): String contents. Defaults to None.
        byte_stream (file-like object, optional): File-like object or BytesIO object. Defaults to None.
    """

    dst_path = Path(dst) / str(filename)
    mode = 'wb' if byte_stream else 'w'

    logger.info(f"Writing file to {str(dst_path)}")
    with open(file=dst_path, mode=mode) as f:
        f.seek(0)
        f.write(byte_stream.read())


def save_uploaded_extract_files(dst: Union[str, Path], filename: Union[str, Path] = None, fileObj: UploadedFile = None) -> bool:
    """Save file object locally to disk. Supports extraction of local archived files to dst.

    Args:
        filename (Union[str, Path]): Name of file
        fileObj (UploadedFile): Bytes-like object
        dst (Union[str, Path]): Destination to store extracted files

    Returns:
        bool: True if successful. False otherwise.
    """
    filename = Path(filename).name
    dst = Path(dst)
    create_folder_if_not_exist(dst)
    with get_temp_dir() as tmp_dir:

        tmp_dir = Path(tmp_dir)
        folder_name = remove_suffix(filename)
        src = tmp_dir / folder_name
        extract_archive(tmp_dir, filename, file_object=fileObj)
        move_file(src=src, dst=dst)


def json_load(file, int_keys=False):
    with io.open(file, encoding='utf8') as f:
        data = json.load(f)
        if int_keys:
            return {int(k): v for k, v in data.items()}
        else:
            return data


def toJSON(obj):
    """Serialise Python Class objects to JSON string

    Args:
        obj (class): Python Class object

    Returns:
        str: Serialised JSON 
    """
    return json.dumps(obj, default=lambda o: o.__dict__,
                      sort_keys=True, indent=4)


def read_yaml(filepath):
    # if not os.path.exists(filepath):
    #     filepath = find_file(filepath)
    with io.open(filepath, encoding='utf-8') as f:
        data = yaml.load(f, Loader=yaml.FullLoader)
    return data


def read_bytes_stream(filepath):
    with open(filepath, mode='rb') as f:
        return io.BytesIO(f.read())


# This code is based on https://github.com/streamlit/demo-self-driving/blob/230245391f2dda0cb464008195a470751c01770b/streamlit_app.py#L48  # noqa: E501
def download_file(url, download_to: Path, expected_size=None):
    # Don't download the file twice.
    # (If possible, verify the download using the file length.)
    if download_to.exists():
        if expected_size:
            if download_to.stat().st_size == expected_size:
                return
        else:
            st.info(f"{url} is already downloaded.")
            if not st.button("Download again?"):
                return

    download_to.parent.mkdir(parents=True, exist_ok=True)

    # These are handles to two visual elements to animate.
    weights_warning, progress_bar = None, None
    try:
        weights_warning = st.warning("Downloading %s..." % url)
        progress_bar = st.progress(0)
        with open(download_to, "wb") as output_file:
            with urllib.request.urlopen(url) as response:
                length = int(response.info()["Content-Length"])
                counter = 0.0
                MEGABYTES = 2.0 ** 20.0
                while True:
                    data = response.read(8192)
                    if not data:
                        break
                    counter += len(data)
                    output_file.write(data)

                    # We perform animation by overwriting the elements.
                    weights_warning.warning(
                        "Downloading %s... (%6.2f/%6.2f MB)"
                        % (url, counter / MEGABYTES, length / MEGABYTES)
                    )
                    progress_bar.progress(min(counter / length, 1.0))
    # Finally, we remove these visual elements by calling .empty().
    finally:
        if weights_warning is not None:
            weights_warning.empty()
        if progress_bar is not None:
            progress_bar.empty()

#------------------------File Archiver----------------------#


def check_archiver_format(filename: str = None):
    """Extract archiver format

    Args:
        filename (str or path-like object): Path to archive file or extension of archive format

    Returns:
        str: Archiver format
    """
    filename = Path(filename)
    # if EXTENSION parsed, archive format is "filename"
    if (len(filename.parts)) == 1:
        archive_format = str(filename)

    # if file path parsed, archive format is suffix of "filename"
    elif (len(filename.parts)) > 1:
        archive_format = filename.suffix
    else:
        logger.info("Parsing error")

    try:
        # get file extension if shutil cannot parse/find format
        archive_format_list = {
            ".zip": "zip",
            ".gz": "gztar",
            ".bz2": "bztar",
            ".xz": "xztar"
        }

        if archive_format in archive_format_list.keys():

            return archive_format_list[archive_format]

        else:
            logger.error("Archive format invalid!")
            st.error("Archive format invalid!")

    except KeyError as e:
        error_msg = f"{e}: Archive format invalid!"
        logger.error(error_msg)
        st.error(error_msg)


def file_unarchiver(filename: Union[str, Path], extract_dir: Union[str, Path]):
    """Unpack archive file

    Args:
        filename (str or path-like object): Path to archive file
        extract_dir (str or path-like object): Output directory

    Returns:
        str: Success string
    """
    extract_dir = Path(extract_dir)
    filename = Path(filename)

    if not extract_dir.exists():
        # create directory if extract directory does not exist
        extract_dir.mkdir(parents=True)
    else:
        pass

    if filename.exists():

        try:  # Shutil will try to extract file extension from archive file
            shutil.unpack_archive(filename=filename, extract_dir=extract_dir)
        except ValueError as e:
            error_msg = f"{e}: Shutil unable to extract archive format from filename"
            logger.error(error_msg)
            st.error(error_msg)

            # get archiver format from Path suffix
            archive_format = check_archiver_format(filename)

            shutil.unpack_archive(
                filename, extract_dir, format=archive_format)

    else:
        # pass IOError
        error_msg = f"{IOError}: File does not exist"
        logger.error(error_msg)
        st.error(error_msg)

    logger.info("Successfully Unarchive")


def extract_one_to_bytes(uploaded_archive: UploadedFile, filename: str) -> bytes:
    """Extract or read a single file from the archive, either from zipfile or tarfile.

    This function should only be used to extract one file, and not multiple files. If
    extracting multiple files, you should open the zipfile/tarfile and write the for loop
    within it.
    """
    # uploaded_archive = deepcopy(uploaded_archive)
    # either use deepcopy or seek to allow reading again
    uploaded_archive.seek(0)
    if uploaded_archive.name.endswith('.zip'):
        with ZipFile(file=uploaded_archive, mode='r') as zipObj:
            file_content = zipObj.read(filename)
    else:
        with tarfile.open(fileobj=uploaded_archive) as tar:
            f = tar.extractfile(filename)
            file_content = f.read()
    uploaded_archive.seek(0)
    return file_content


def extract_archive(dst: Union[str, Path], archived_filepath: Path = None, file_object=None):
    """Extract archive to folder

    Args:
        dst (Union[str, Path]): Destination to store unpacked files
        archived_filepath (Path, optional): Path to archive file. Neglected if file_object passed. Defaults to None.
        file_object (byte-like/file, optional): Byte-stream object / file-like object. Defaults to None.
    """
    dst = Path(dst)
    create_folder_if_not_exist(dst)

    if isinstance(file_object, UploadedFile):
        archived_filepath = file_object.name

    # Make sure it is Path() object
    archived_filepath = Path(archived_filepath)
    archive_extension = archived_filepath.suffix

    if (archive_extension not in SUPPORTED_ARCHIVE_EXT):

        # RETURN TO CALLER IF ARCHIVE FORMAT NOT SUPPORTED !!!!
        error_msg = f"Archiving format {archive_extension} is not supported"
        logger.error(error_msg)
        st.error(error_msg)
        return

    if archive_extension == '.zip':
        file = file_object if file_object else archived_filepath
        with ZipFile(file=file, mode='r') as zipObj:
            logger.info(f"Extracting {archived_filepath.name} to {dst}")
            zipObj.extractall(path=dst)

    else:
        tar_read_format_list = {
            ".gz": "r:gz",
            ".bz2": "r:bz2",
            ".xz": "r:xz"
        }

        # get correct compression  read mode
        tar_mode = tar_read_format_list.get(archive_extension)

        if file_object:
            fileobj = deepcopy(file_object)
            name = None
        else:
            fileobj = None
            name = archived_filepath

        with tarfile.open(name=name,
                          fileobj=fileobj,
                          mode=tar_mode) as tarObj:
            logger.info(f"Extracting {archived_filepath.name} to {dst}")
            tarObj.extractall(path=dst)


def list_files_in_archived(archived_filepath: Path = None, file_object=None,
                           return_content_size: bool = False,
                           skip_dir: bool = False) -> Union[List[str], Tuple[List[str], int]]:
    """List files in the zip (.zip) and tar archives.

    Supported tar compressions:
    1. gzip (.tar.gz)
    2. bz2 (.tar.bz2)
    3. xz (.tar.xz)


    Args:
        archived_filepath (Path): Filepath to archives or filename(to get extensions). Defaults to None.
        file_object (Any): file-like object of archives. Defaults to None.
        skip_dir (bool): If True, skip directories. Defaults to False.

    Returns:
        List[str]: A list of member files in the archives.
    """
    # NOTE: deepcopy to avoid clearing the file contents after reading
    # file_object = deepcopy(file_object)
    # or use seek
    file_object.seek(0)

    if isinstance(file_object, UploadedFile):
        archived_filepath = file_object.name

    # Make sure it is Path() object
    archived_filepath = Path(archived_filepath)

    # get extension of the archive file
    archive_extension = archived_filepath.suffix

    if (archive_extension not in SUPPORTED_ARCHIVE_EXT):

        # RETURN TO CALLER IF ARCHIVE FORMAT NOT SUPPORTED !!!!
        error_msg = f"Archiving format {archive_extension} is not supported"
        logger.error(error_msg)
        st.error(error_msg)
        return

    content_size = 0
    filepaths = []
    if archive_extension == '.zip':
        file = file_object if file_object else archived_filepath

        with ZipFile(file=file, mode='r') as zipObj:
            if not skip_dir:
                filepaths = sorted(zipObj.namelist())
                if return_content_size:
                    members = zipObj.infolist()
                    for m in members:
                        content_size += m.file_size
            else:
                members = zipObj.infolist()
                for m in members:
                    if return_content_size:
                        content_size += m.file_size
                    if not m.is_dir():
                        filepaths.append(m.filename)
    else:
        tar_read_format_list = {
            ".gz": "r:gz",
            ".bz2": "r:bz2",
            ".xz": "r:xz"
        }

        tar_mode = tar_read_format_list.get(archive_extension)

        if file_object:
            fileobj = file_object
            name = None
        else:
            fileobj = None
            name = archived_filepath

        with tarfile.open(name=name,
                          fileobj=fileobj,
                          mode=tar_mode) as tarObj:
            if not skip_dir:
                filepaths = sorted(tarObj.getnames())
                if return_content_size:
                    members = tarObj.getmembers()
                    for m in members:
                        content_size += m.size
                # file_object.seek(0, 0)
            else:
                members = tarObj.getmembers()
                for m in members:
                    if return_content_size:
                        content_size += m.size
                    if m.isfile():
                        filepaths.append(m.name)
    file_object.seek(0)
    if return_content_size:
        return filepaths, content_size
    return filepaths


def check_image_files(file_list: List[str]) -> List[str]:
    """Check the contents in the tarfile to verify that all of them are images."""
    for fname in file_list:
        ext = os.path.splitext(fname)[-1]
        if ext not in IMAGE_EXTENSIONS:
            st.error(
                f"'{ext}' file extension is not a supported image format.")
            st.stop()
        if fname.startswith((r'.', r'/')):
            st.error(f'{fname} is not a valid filepath. '
                     'Absolute / relative filepath is not accepted.')
            st.stop()
    return file_list


def get_member(name: str, archived_filepath: Path = None, file_object: IO = None, decode: str = 'utf-8') -> Union[IO, str]:
    """Obtain bytes-stream of file member in the archived file without extraction

    Args:
        name (str): Name of file (obtained from `list_files_in_archived()`)
        archived_filepath (Path, optional): Path to archive file. Neglected if file_object passed. Defaults to None.
        file_object (IO, optional): Byte-stream object / file-like object. Defaults to None.
        decode (str, optional): Decode format. Defaults to 'utf-8'.

    Returns:
        Union[IO,str]: Byte-stream of file member
    """

    # Make sure it is Path() object
    archived_filepath = Path(archived_filepath)

    # get extension of the archive file
    archive_extension = archived_filepath.suffix

    if (archive_extension not in SUPPORTED_ARCHIVE_EXT):

        # RETURN TO CALLER IF ARCHIVE FORMAT NOT SUPPORTED !!!!
        error_msg = f"Archiving format {archive_extension} is not supported"
        logger.error(error_msg)
        st.error(error_msg)
        return

    if archive_extension == '.zip':

        file = file_object if file_object else archived_filepath

        with ZipFile(file=file, mode='r') as zipObj:

            file_bytes = zipObj.read(str(name))

            if decode:
                try:
                    file_bytes = file_bytes.decode(decode)
                except Exception as e:
                    logger.error(f"{e}")

    else:
        tar_read_format_list = {
            ".gz": "r:gz",
            ".bz2": "r:bz2",
            ".xz": "r:xz"
        }

        tar_mode = tar_read_format_list.get(archive_extension)

        if file_object:
            fileobj = deepcopy(file_object)
            filename = None
        logger.info(file_object)
        with tarfile.open(name=filename,
                          fileobj=fileobj,
                          mode=tar_mode) as tarObj:

            file_bytes = tarObj.extractfile(str(name)).read()

            if decode:
                try:
                    file_bytes = file_bytes.decode(decode)
                except Exception as e:
                    logger.error(f"{e}")
            # file_object.seek(0)

    return file_bytes


# NOTE DEPRECATED -> USE file_archive_handler
def single_file_archiver(archive_filename: Path, target_filename: Path, target_root_dir, target_base_dir, archive_extension=".zip"):
    """Archiver for single files

    Args:
        archive_filename (Path): Archive output file path with full extensions (.tar.gz,.tar.xz,.tar.bz2,.zip)
        target_filename (Path): Filepath of file to be archived
        target_root_dir (Path): Root directory of file
        target_base_dir (Path): Base directory of file (Folder containing the file)
        archive_extension (str, optional): Archive compression format type (.gz,.xz,.bz2,.zip). Defaults to ".zip".


    """

    os.chdir(target_root_dir)
    archive_filename = Path(archive_filename).with_suffix(
        archive_extension)  # return filename with extension
    if archive_extension == ".zip":  # zip file
        with ZipFile(file=archive_filename, mode='w') as zip:
            zip.write(target_filename,
                      arcname=target_base_dir)
            logger.info(f"Successfully archived folder: {archive_filename}")
            st.success(f"Successfully archived folder: {archive_filename}")

    else:  # remaining is tarball
        tar_write_format_list = {
            ".gz": "w:gz",
            ".bz2": "w:bz2",
            ".xz": "w:xz"
        }
        if archive_extension in tar_write_format_list.keys():

            tar_mode = tar_write_format_list[archive_extension]

        else:
            logger.error("Archive format invalid!")
            st.error("Archive format invalid!")

        with tarfile.open(archive_filename, tar_mode) as tar:
            tar.add(target_filename, arcname=target_base_dir)
            logger.info(f"Successfully archived folder: {archive_filename}")
            st.success(f"Successfully archived folder: {archive_filename}")
    chdir_root()


def file_archiver(archive_filename: Path, target_root_dir: Path, target_base_dir: Path, archive_format: str = "zip"):
    """Packed batched files into archived-format

    ### Example:

    >>> target_root_dir=Path('/home/macintosh/Desktop')
    >>> target_base_dir=Path('testing/test.html')
    >>> archive_filename=Path('./testing')
    >>> file_archive(archive_filename,target_root_dir,target_base_dir,'.gz')

    Folder path tree:

    home \n
    |- [.....some_intermediate_paths......]
      |- usr (Root)
        |- Downloads (Base)
            |- file1  \n  
            |- file2 \n
            |- file3 \n

    Args:
        archive_filename (Path): Output Folder path to files without archive extension. Can be relative or absolute. eg: Folder to be archived: home/[.....some_intermediate_paths......]/usr/Downloads
        target_root_dir (Path): Root directory to the 'archive_filename'. eg:home/[.....some_intermediate_paths......]/usr
        target_base_dir (Path): Relative path of Folder-to-be-archived to Root dir. eg: ./Downloads
        archive_extension (str, optional): Archive format extension (.gz, .xz, .bz2, .zip). Defaults to ".zip".
    """

    current_working_dir = Path.cwd()  # save current working directory
    os.chdir(str(target_root_dir))  # change to target root directory
    try:
        archived_name = shutil.make_archive(base_name=str(
            archive_filename), format=archive_format, root_dir=target_root_dir, base_dir=target_base_dir)
        logger.info(f"Successfully archived {archived_name}")
        # return back to initial working directory
        os.chdir(current_working_dir)
        return(archived_name)
    except:
        error_msg = "Failed to archive file"
        logger.info(error_msg)
        st.error(error_msg)
        # return back to initial working directory
        # os.chdir(current_working_dir)
    chdir_root()


def file_archive_handler(archive_filename: Path, target_filename: Path, archive_extension: str = ".zip"):
    """File archive handler for single / batch files

    - Will change directory to 'target_root_dir'
    - if archive_filename = './some_archive_name'. The output file path will be '<target_root_dir> / some_archive_name.<archive_extension>
    - else if archive_filename is absolute = 'some_root/...../some_archive_name'. The output file path will be 'some_root/...../some_archive_name.<archive_extension>'
    - .gz, .xz, .bz2 formats are file compression of TAR files. Resultant file extensions will be '.tar.<archive_extension>'. eg. '.tar.gx'

    ### Example:

    >>> target_root_dir=Path('/home/macintosh/Desktop')
    >>> target_base_dir=Path('testing/test.html')
    >>> target_filename=target_root_dir / target_base_dir eg: /home/macintosh/Desktop/testing/test.html for single file ; /home/macintosh/Desktop/testing for batch files
    >>> archive_filename=Path('./testing') => Relative to target_root_dir
    >>> file_archive_handler(archive_filename,target_filename,'.gz')

    Folder path tree:

    home \n
    |- [.....some_intermediate_paths......]
      |- usr (Root)
        |- Downloads (Base)
            |- file1  \n  
            |- file2 \n
            |- file3 \n


    Args:
        archive_filename (Path): Output Folder path to files without archive extension. Can be relative or absolute. eg: Folder to be archived: home/[.....some_intermediate_paths......]/usr/Downloads
        target_root_dir (Path): Root directory to the 'archive_filename'. eg:home/[.....some_intermediate_paths......]/usr
        target_base_dir (Path): Relative path of Folder-to-be-archived to Root dir. eg: ./Downloads
        archive_extension (str, optional): Archive compression format extension (.gz, .xz, .bz2, .zip). Defaults to ".zip".
    """

    archive_filename = Path(archive_filename).resolve()
    archive_format = check_archiver_format(
        archive_extension)  # get archive file extension

    # >>>> SINGLE FILE
    if target_filename.is_file():
        logger.info(f"Path is file {target_filename}")
        target_root_dir = target_filename.parents[1]
        target_base_dir = Path(target_filename).resolve(
        ).parent.relative_to(target_root_dir)
        file_archiver(
            archive_filename, target_filename, target_root_dir,
            target_base_dir, archive_extension)

    # >>>> BATCH FILE
    elif target_filename.is_dir():
        target_root_dir = target_filename.parent
        target_base_dir = target_filename.relative_to(target_root_dir)
        logger.info(f"Path is directory {target_filename}")
        file_archiver(archive_filename, target_root_dir,
                      target_base_dir, archive_format)

    else:
        error_msg = f"{FileNotFoundError} File does not exist"
        logger.error(error_msg)
        st.error(error_msg)

    chdir_root()


def st_download_button(zip_filepath: Path, download_filename: str, btn_name: str):
    with open(zip_filepath, 'rb') as f:
        bytes = f.read()
        b64 = base64.b64encode(bytes).decode()
        href = f'<a href="data:file/zip;base64,{b64}" download=\'{download_filename}\'>\
            download file \
        </a>'
        st.markdown(href, unsafe_allow_html=True)
