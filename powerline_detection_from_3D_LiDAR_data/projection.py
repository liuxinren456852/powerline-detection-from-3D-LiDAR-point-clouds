import laspy
import numpy as np
import pdal


def getMinMax(file_name):
    """
    Get the minimum and maximum value of Z
    :param file_name: lidar filename
    :return: min, max
    """
    inFile = laspy.read('2d_projection/' + file_name + '.las')
    pcd = np.vstack([inFile.z]).transpose((1, 0))  # X, Y, Z data
    dataset = pcd[:, 0]

    return min(dataset), max(dataset)


def projection_3d_to_2d_slicing(file_name):
    """
    Split a given lidar into several slices and then project them to create 2d images
    :param file_name: lidar filename
    :return:
    """
    minZ, maxZ = getMinMax(file_name)
    print(minZ)
    print(maxZ)
    diff_z = maxZ - minZ
    nbr_slice = 4
    step = diff_z / nbr_slice
    for i in np.arange(0, nbr_slice):
        json = """
                   [
                        "2d_projection/""" + file_name + """.las", 
                        {
                            "type":"filters.range",
                            "limits":"Z[""" + str(minZ + (i * step)) + """:""" + str(minZ + ((i + 1) * step)) + """]" 
                        }, 
                        { 
                            "type":"writers.las", 
                            "filename":"2d_projection/""" + file_name + """/las/""" + file_name + str(i + 1) + """.las", 
                            "extra_dims":"all"
                        },
                        {
                            "type":"writers.gdal",
                            "filename":"2d_projection/""" + file_name + """/tif/""" + file_name + str(i + 1) + """.tif",
                            "data_type":"float",
                            "output_type":"max",
                            "resolution": 0.2,
                            "pdal_metadata": "true",
                            "metadata": "true"
                        }
                    ] 
                    """
        pipeline = pdal.Pipeline(json)
        count = pipeline.execute()
        arrays = pipeline.arrays
        metadata = pipeline.metadata
        log = pipeline.log


if __name__ == '__main__':
    filename = 'L004'
    projection_3d_to_2d_slicing(filename)
