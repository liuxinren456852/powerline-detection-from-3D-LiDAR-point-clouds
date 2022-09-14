import laspy
import numpy as np
import statistics as sts
import pdal
import sectioning as sec
import os


def groundFilter(filename):
    """
    Filter Ground points
    :param filename:
    :return:
    """
    json1 = """
            [
                {
                     "type":"readers.ply",
                     "filename":"data/""" + filename + """.ply"
                },
                {
                    "type":"filters.csf"
                },
                {
                    "type":"filters.range",
                    "limits":"Classification[2:2]"
                },
                {
                     "type":"writers.las",
                     "filename":"intermediate/GroundFilter.las",
                     "extra_dims":"all"
               }         
            ]
          """
    pl = pdal.Pipeline(json1)
    ct = pl.execute()
    logs = pl.log


def getGroundMeanZ():
    """
    Get mean Z at Ground level
    :return: mean Z
    """
    inFile = laspy.read('intermediate/GroundFilter.las')
    dataset = np.vstack([inFile.z]).transpose()  # X, Y, Z data
    avg_z = sts.mean(dataset[:, 0])
    # print(avg_z)
    return avg_z


def preprocessing_pipeline(filename):
    """
    PDAL preprocessing pipeline and features computation
    :param filename:
    :return:
    """
    groundFilter(filename)
    minZ = getGroundMeanZ()
    minZ = minZ + 4
    maxZ = minZ + 12
    json = """
         [
            {
                "type":"readers.ply",
                "filename":"data/""" + filename + """.ply"
            },                 
             {
                  "type":"filters.smrf"
             },
             {
                 "type":"filters.range",
                 "limits":"Classification![2:2]"
             } , 
             {
                 "type":"writers.las",
                 "filename":"intermediate/intermediateGround.las",
                 "extra_dims":"all"
             },
              {
                "type":"filters.range",
                "limits":"Z[""" + str(minZ) + """:""" + str(maxZ) + """]"  
             },
            {
                "type":"writers.las",
                "filename":"intermediate/intermediateZ.las",
                "extra_dims":"all"
            },  
            {
                "type":"filters.range",
                 "limits":"scalar_Intensity[:8)" 
            },   
            {
                "type":"writers.las",
                "filename":"intermediate/intermediateIntensity.las",
                "extra_dims":"all"
            } , 
            {
                "type":"filters.radialdensity",
                "radius": 1.0
            }, 
            {
                 "type":"filters.range",
                 "limits":"RadialDensity[:500)" 
            }, 
            {
                "type":"writers.las",
                "filename":"intermediate/intermediateRadialDensity.las",
                "extra_dims":"all"
            },  
            {
                "type":"filters.optimalneighborhood",
                "max_k": 50
            },
            {
                "type":"filters.normal",
                "knn":25
            },
            {
                "type":"filters.covariancefeatures",
                "knn":25,
                "threads": 2,
                "feature_set": "all"
            },
            {
                "type":"writers.las",
                "filename":"preprocessed_data/""" + filename + """.las",
                "extra_dims":"all"
            }
         ]
         """

    pipeline = pdal.Pipeline(json)
    count = pipeline.execute()
    arrays = pipeline.arrays
    metadata = pipeline.metadata
    log = pipeline.log

    print(len(arrays))
    print(arrays)


def convert_bytes(num):
    num /= 1000000
    return "%3.1f " % num


def file_size(file_path):
    """
    this function will return the file size
    """
    if os.path.isfile(file_path):
        file_info = os.stat(file_path)
        return convert_bytes(file_info.st_size)


if __name__ == '__main__':
    file_name = "L003"
    filepath = 'data/' + file_name + '.ply'
    f_size = file_size(filepath)
    if float(f_size) <= 500:
        preprocessing_pipeline(file_name)
    else:
        sec.sectioning_lidar(file_name)
