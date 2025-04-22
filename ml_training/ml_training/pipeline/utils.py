import os
import shutil
import zipfile


def create_dir(path: str) -> None:
    os.makedirs(path, mode=0o777, exist_ok=True)


def delete_dir(path: str) -> None:
    if os.path.exists(path):
        shutil.rmtree(path)


def delete_file(path: str) -> None:
    if os.path.exists(path):
        os.remove(path)


def zip_directory(folder_path: str, zip_file: str):
    zip_file = zipfile.ZipFile(f"{zip_file}.zip", "w")
    for folder_name, subfolders, filenames in os.walk(folder_path):
        for filename in filenames:
            file_path = os.path.join(folder_name, filename)
            zip_file.write(file_path)
    zip_file.close()
