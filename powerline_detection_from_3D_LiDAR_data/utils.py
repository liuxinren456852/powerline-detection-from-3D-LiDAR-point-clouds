from osgeo import gdal
import pandas as pd
import numpy as np
import tifffile
import cv2 as cv
import laspy
from sklearn.metrics import confusion_matrix, accuracy_score


def get_original_coordinates(image_path):
    """
    Retrieve original LiDAR (Xg, Yg) points from the image, then create a 3D array (image) where the couple of coordinates
    (Xg, Yg)  are set as pixel values to map the original image.
    :param image_path: full path of the tiff image to be read
    :return: corresponding (Xg, Yg) coordinates as 3D image
    """
    src_ds = gdal.Open(image_path)
    rast_array = np.array(src_ds.GetRasterBand(1).ReadAsArray())
    col_size = rast_array.shape[1]
    xyz = gdal.Translate("destination.xyz", src_ds)
    df = pd.read_csv('destination.xyz', sep=" ", header=None)
    df.columns = ["x", "y", "value"]
    df['xy'] = list(zip(df.x, df.y))

    xy_array = df['xy'].values
    xy_array_2d = np.array(xy_array).reshape(-1, col_size)  # reshape 2d array
    xy_array_3d = np.stack((xy_array_2d,) * 3, -1)  # reshape 3d array
    return xy_array_3d


def apply_hough_transform(image_path, threshold, minLineLength, maxLineGap):
    """
    Apply Probabilistic Hough Line Transform to detect lines in image :
    ---first the 2D array image is converted  to 3D array, then the type from 32bits to 8bits
    :param minLineLength: the minimum length of the line
    :param maxLineGap: maximum gap between two segments of line
    :param threshold: number of votes
    :param image_path: full path of the tiff image
    :return: detected lines image
    """
    with tifffile.TiffFile(image_path) as tif:
        image = tif.pages[0].asarray()

        img = np.stack((image,) * 3, -1)
        img = img.astype(np.uint8)
        grayed = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        src = cv.threshold(grayed, 0, 255, cv.THRESH_BINARY_INV | cv.THRESH_OTSU)[1]

        dst = cv.Canny(src, 50, 150, None, 3)

        # Copy edges to the images that will display the results in BGR
        cdstP = cv.cvtColor(dst, cv.COLOR_GRAY2BGR)

        # Probabilistic Hough Line Transform
        linesP = cv.HoughLinesP(dst, 1, np.pi / 180, threshold, None, minLineLength, maxLineGap)
        if linesP is not None:
            for i in range(0, len(linesP)):
                l = linesP[i][0]
                cv.line(cdstP, (l[0], l[1]), (l[2], l[3]), (0, 0, 255), 3, cv.LINE_AA)

    return cdstP


def show_image(img):
    """
    Display an image
    :param img: image to be displayed
    :return:
    """
    cv.imshow("Detected Lines (in red) - Standard Hough Line Transform", img)
    cv.waitKey()


def get_detected_line_points(img, xy_array_3d):
    """
    Retrieve (Xg, Yg) coordinates of each pixel of the detected lines.
    :param img: detected lines image
    :param xy_array_3d: 3D array of (Xg, Yg) coordinates
    :return: list of points (Xg, Yg) of the detected lines
    """
    detected_lines_points = []
    for idx in np.ndindex(img.shape):
        if img[idx] != 0:
            detected_lines_points.append(xy_array_3d[idx])
        else:
            continue

    return detected_lines_points


def detected_lines_points_labelling(lidar_path, detected_lines_coord):
    """
    Labelling of points cloud in the original LiDAR
    1 is set for powerline, and 0 for others
    :param lidar_path: full path of the original LiDAR
    :param detected_lines_coord: list of points (Xg, Yg) of the detected lines
    :return: pandas dataframe partially labeled with detected  lines
    """
    inFile = laspy.read(lidar_path)
    pcd = np.vstack([inFile.x, inFile.y, inFile.z, inFile.classification]).transpose((1, 0))
    df = pd.DataFrame(pcd, columns=['x', 'y', 'z', 'classification'])
    df = df.astype(float)
    df_cp = df.copy()
    df_cp[['x', 'y']] = df_cp[['x', 'y']].round(decimals=2)
    df_cp['prediction'] = np.zeros(df_cp.shape[0])
    print(df_cp.prediction.value_counts())

    # loop through detected lines coordinate
    for i in range(0, len(detected_lines_coord)):
        point = detected_lines_coord[i]
        x = round(point[0], 2)
        y = round(point[1], 2)
        df_cp.loc[(df['x'] == x) & (df_cp['y'] == y), 'prediction'] = 1

    print(df_cp.prediction.value_counts())

    df['prediction'] = df_cp.prediction.values

    return df


def find_neighborhood_points(df, r):
    """
    Find neighboring points in a sphere of a given radius
    :param r: radius
    :param df: pandas dataframe partially labeled with detected  lines
    :return: final dataframe with all the labels set
    """
    df_zeros = df[df['prediction'] == 0]
    df_ones = df[df['prediction'] == 1]

    array_ones = df_ones[['x', 'y']].values
    for i in range(0, len(array_ones)):
        x = array_ones[i][0]
        y = array_ones[i][1]
        x_min = x - r
        x_max = x + r
        y_min = y - r
        y_max = y + r
        df_zeros.loc[(((df_zeros['x'] >= x_min) & (df_zeros['x'] <= x_max)) &
                      ((df_zeros['y'] >= y_min) & (df_zeros['y'] <= y_max))), 'prediction'] = 1

    df_final = pd.concat([df_zeros, df_ones], axis=0)
    df_final.sort_index(axis=0)

    return df_final


def save_to_las(df, filepath):
    """
    Convert pandas dataframe to las file
    :param df: pandas dataframe
    :param filepath: full path of the las file to save
    :return:
    """
    data = df.values

    # 1. Create a new header
    header = laspy.LasHeader(point_format=3, version="1.2")
    header.add_extra_dim(laspy.ExtraBytesParams(name="classification", type=np.int32))
    header.add_extra_dim(laspy.ExtraBytesParams(name="prediction", type=np.int32))
    header.offsets = np.min(data[:, 0:3], axis=0)
    header.scales = np.array([0.1, 0.1, 0.1])

    # 2. Create a Las
    las = laspy.LasData(header)
    las.x = data[:, 0]
    las.y = data[:, 1]
    las.z = data[:, 2]
    las.classification = data[:, 3]
    las.prediction = data[:, 4]

    las.write(filepath)


def get_IoU_Accuracy(df):
    """
    For a given dataframe, compute and display IoU accuracy
    :param df: pandas dataframe
    :return:
    """
    cm = confusion_matrix(df['classification'], df['prediction'])
    acc = round((cm[1][1] / (cm[1][1] + cm[0][1] + cm[1][0])) * 100, 2)
    a = accuracy_score(df['classification'], df['prediction'])

    print("++++++++++++++++++++++++++++++++++++++++++++++++++")
    print("Accuracy is {}%".format(a * 100))
    print("IoU Accuracy : %.2f%%" % acc)
    print("++++++++++++++++++++++++++++++++++++++++++++++++++")


def lidar_to_csv(filepath):
    """
    Convert las file to csv
    :param filepath:
    :return: 
    """
    inFile = laspy.read(filepath)
    pcd = np.vstack([inFile.x, inFile.y, inFile.z, inFile.classification, inFile.prediction]).transpose((1, 0))
    pcd_df = pd.DataFrame(pcd, columns=['x', 'y', 'z', 'classification', 'prediction'])

    return pcd_df
