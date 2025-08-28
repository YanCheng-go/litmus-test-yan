from deforestation.io import FILENAME_TIME_RE


def test_filename_regex():
    for name in ["tile_2016_1.tif", "tile_2020-2.tiff", "X_1999_2.TIF"]:
        m = FILENAME_TIME_RE.search(name)
        assert m and m.group(1).isdigit() and m.group(2) in {"1","2"}
