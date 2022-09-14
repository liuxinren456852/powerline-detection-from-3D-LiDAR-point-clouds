import pdal
import preprocessing as prep
from plyfile import PlyData


def getMinMaxGpsTime(filename):
    """
    Get the minimum and maximum values of GpsTime
    :param filename: filename
    :return: min, max GpsTime
    """
    plydata = PlyData.read('data/' + filename + '.ply')
    gps_data = plydata['vertex']['scalar_GPSTime']

    return min(gps_data), max(gps_data)


def sectioning_lidar(filename):
    """
    Split larger lidar into several sections and then preprocess them
    :param filename:filename without extension
    :return:
    """
    min_gps, max_gps = getMinMaxGpsTime(filename)
    diff_gps = max_gps - min_gps
    nbr_section = 2
    step = diff_gps / nbr_section

    prep.groundFilter(filename)
    minZ = prep.getGroundMeanZ()
    minZ = minZ + 4
    maxZ = minZ + 12

    print(minZ, maxZ, step)

    for i in range(0, nbr_section):
        print(i)
        json = """
        [
            {
                "type":"readers.ply",
                "filename":"data/""" + filename + """.ply"
            },  
            {
                "type":"filters.range",
                 "limits":"scalar_GPSTime[""" + str(min_gps + (i * step)) + """:""" + str(min_gps + ((i + 1) * step)) \
               + """]" 
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
                "filename":"preprocessed_data/""" + filename + """_Sec""" + str(i + 1) + """.las",
                "extra_dims":"all"
            }
        ]
        """
        # print(i * step)
        pipeline = pdal.Pipeline(json)
        count = pipeline.execute()
