import os

def recursively_list_files(path):
    walker = os.walk(path)
    while True:
        try:
            root, dirs, files = walker.next()
            for filename in files:
                yield os.path.join(root, filename)
        except StopIteration:
            break
        
