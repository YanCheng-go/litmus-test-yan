"""
Read the first tif in ./data/ and prints example information.
"""

import numpy as np
import rasterio

#Dictionary for the bands as they are indexed by rasterio
BANDS = {
    1: 'coastal',
    2: 'blue',
    3: 'green',
    4: 'red',
    5: 'nir',
    6: 'swir1',
    7: 'swir2',
    8: 'thermal1',
    9: 'thermal2'
}

def read_tif(image_file):
    """
    Opens an image file and returns a multidimensional numpy array

    Arguments:
        image_file {str}    The location of the image file

    Returns:
        numpy array
    """
    with rasterio.open(image_file, 'r') as src:
        # src is a rasterio dataset representation. It contains all sorts
        # of useful info, like the extent of the image, the
        # location of where it is, the number of bands etc.
        print("\nThe Number of bands in this image is {} \n".format(src.count))
        print("Band number 5 is the {} band \n".format(BANDS[5]))

        # The read method returns a numpy array. If you do not add any arguments,
        # the entire image with all bands is returned. You can select specific
        # bands by using an int or list for the band indices.
        #
        # NOTE: Rasterio bands are indexed from 1, the returned numpy arrays are
        # indexed from 0.
        return src.read()


if __name__ == '__main__':
    FILENAME = 'data/Peninsular_Malaysia_2017_1_Landsat8.tif'
    LANDSAT_IMAGE = read_tif(FILENAME)

    print(FILENAME, ': ', LANDSAT_IMAGE[4][300:320])

    # You will find some 'holes' in the data, this is where Landsat did not
    # measure a valid surface reflectance in the composites period, due to
    # atmospheric disturbances (clouds). These values are NaN.
    print("\nThe mean {} value of all non clouded areas in Peninsular Malaysia are {} \n".format(
        BANDS[5],
        np.nanmean(LANDSAT_IMAGE[4])
    ))

    print("The numpy array of a band has the same shape as the image in pixels {} \n".format(
        LANDSAT_IMAGE[0].shape
    ))
