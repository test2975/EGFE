from os import makedirs, path


def create_folder(folder_path: str):
    path.isdir(folder_path) or makedirs(folder_path, exist_ok=True)
