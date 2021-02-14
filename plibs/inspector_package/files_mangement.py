import os
import codecs

def read_file(path):
    with open(path, 'r') as f:
        data = f.read()
        f.close()
    return data

def file_exists(path):
    return os.path.isfile(path)

def write_file(path, content):
    f = codecs.open(path, "w", encoding='utf8')
    f.write(str(content))
    f.close()
