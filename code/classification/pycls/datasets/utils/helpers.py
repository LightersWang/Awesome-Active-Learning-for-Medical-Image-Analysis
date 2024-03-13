import csv

def load_imageset(path, set_name):
    """
    Returns the image set `set_name` present at `path` as a list.
    Keyword arguments:
        path -- path to data folder
        set_name -- image set name - labeled or unlabeled.
    """
    reader = csv.reader(open(os.path.join(path, set_name+'.csv'), 'rt'))
    reader = [r[0] for r in reader]
    return reader