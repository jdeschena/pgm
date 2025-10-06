import fsspec
from pathlib import Path


def params2key(**kwargs):
    bool_entries = []
    non_bool_entries = []

    for k, v in kwargs.items():
        # Replace / in path so that it doesn't create sub-directories
        if type(v) is str:
            v = v.replace("/", "#")
        if type(v) is bool:
            bool_entries.append((k, v))
        else:
            non_bool_entries.append((k, v))

    bool_entries.sort(key=lambda x: x[0])
    non_bool_entries.sort(key=lambda x: x[0])

    name = ""
    for k, v in non_bool_entries:
        name += str(k) + "=" + str(v)
        name += "_"

    for i, (k, v) in enumerate(bool_entries):
        if v:
            name += k
        else:
            name += "no_" + k

        if i != len(bool_entries) - 1:
            name += "_"

    return name


def fsspec_exists(filename):
    """Check if a file exists using fsspec."""
    if isinstance(filename, Path):
        filename = str(filename)
    fs, _ = fsspec.core.url_to_fs(filename)
    return fs.exists(filename)


def fsspec_listdir(dirname):
    """Listdir in manner compatible with fsspec."""
    if isinstance(dirname, Path):
        dirname = str(dirname)
    fs, _ = fsspec.core.url_to_fs(dirname)
    return fs.ls(dirname)


def fsspec_mkdirs(dirname, exist_ok=True):
    """Mkdirs in manner compatible with fsspec."""
    if isinstance(dirname, Path):
        dirname = str(dirname)
    fs, _ = fsspec.core.url_to_fs(dirname)
    fs.makedirs(dirname, exist_ok=exist_ok)



def vars_to_cache_name(dataset_name, **kwargs):
    bool_entries = []
    non_bool_entries = []

    for k, v in kwargs.items():
        if type(v) is bool:
            bool_entries.append((k, v))
        else:
            non_bool_entries.append((k, v))

    bool_entries.sort(key=lambda x: x[0])
    non_bool_entries.sort(key=lambda x: x[0])

    name = f"{dataset_name}_"
    for k, v in non_bool_entries:
        name += str(k) + "=" + str(v)
        name += "_"

    for i, (k, v) in enumerate(bool_entries):
        if v:
            name += k
        else:
            name += "no_" + k

        if i != len(bool_entries) - 1:
            name += "_"

    return name