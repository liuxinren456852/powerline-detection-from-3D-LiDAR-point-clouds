import os
import time
from os import listdir
from os.path import isfile, join
from utils import *
import projection as proj


def main(file_to_predict):
    proj.projection_3d_to_2d_slicing(file_to_predict)
    tif_path = '2d_projection/' + file_to_predict + '/tif/'
    las_path = '2d_projection/' + file_to_predict + '/las/'
    detected_lines_path = '2_projection/' + file_to_predict + '/detected_lines/'
    tif_files = [file for file in listdir(tif_path) if isfile(join(tif_path, file))]
    las_files = [file for file in listdir(las_path) if isfile(join(las_path, file))]
    df_list = []
    for i in range(0, len(tif_files)):
        print('current file : ', tif_files[i])
        image_path = tif_path + tif_files[i]
        xy_array_3d = get_original_coordinates(image_path)
        cdstP = apply_hough_transform(image_path, 50, 65, 5)
        path = r''.join(os.path.join(os.getcwd()) + '/' + detected_lines_path + tif_files[i]).replace('\\', '/')
        cv.imwrite(path, cdstP)
        print(cdstP.shape)
        # show_image(cdstP)
        detected_lines_coord = get_detected_line_points(cdstP, xy_array_3d)
        print(len(detected_lines_coord))
        lidar_path = las_path + las_files[i]
        df = detected_lines_points_labelling(lidar_path, detected_lines_coord)
        df_list.append(df)

    print(len(df_list))
    df = pd.concat(df_list, axis=0)
    df.reset_index(drop=True, inplace=True)
    print('value counts hough :')
    print(df.prediction.value_counts())

    df_n = find_neighborhood_points(df, 0.75)
    print('value counts neighborhood :')
    print(df_n.prediction.value_counts())

    lidar_df = lidar_to_csv('results/xgboost/' + file_to_predict + '.las')
    df_not_cable = lidar_df[lidar_df['prediction'] == 0]
    df_final = pd.concat([df_not_cable, df_n], axis=0)

    print('resultat final :')
    print(df_final.prediction.value_counts())

    get_IoU_Accuracy(df_final)

    save_path = 'results/hough/' + file_to_predict + '.las'
    save_to_las(df_final, save_path)


if __name__ == "__main__":
    start_time = time.time()
    file_to_predict = 'L003'
    main(file_to_predict)
    total_time = round((time.time() - start_time), 2)
    print("--- %s seconds ---" % total_time)

