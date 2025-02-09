import os

def compressed_filepath(filePath):
    dir_path = os.path.dirname(filePath)
    file_name, _ = os.path.splitext(os.path.basename(filePath))
    return os.path.join(dir_path, f"{file_name}_compressed.jpeg")