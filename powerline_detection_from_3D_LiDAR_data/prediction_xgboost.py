import time
import laspy
import xgboost
import numpy as np
import pandas as pd
from os import listdir
from os.path import isfile, join
from sklearn.model_selection import train_test_split
from utils import get_IoU_Accuracy, save_to_las


def load_lidar_data(lidar_data):
    """
    Load lidar data and create pandas dataframe
    :param lidar_data:
    :return: dataframe
    """
    lidar_points = np.array((lidar_data.x, lidar_data.y, lidar_data.z, lidar_data.Curvature, lidar_data.NormalZ,
                             lidar_data.Verticality, lidar_data.Planarity, lidar_data.Linearity, lidar_data.Scattering,
                             lidar_data.scalar_Intensity, lidar_data.Omnivariance, lidar_data.Eigenentropy,
                             lidar_data.Anisotropy, lidar_data.OptimalKNN, lidar_data.RadialDensity,
                             lidar_data.scalar_Label)).transpose()

    lidar_df = pd.DataFrame(lidar_points,
                            columns=['x', 'y', 'z', 'Curvature', 'NormalZ', 'Verticality', 'Planarity', 'Linearity',
                                     'Scattering', 'scalar_Intensity', 'Omnivariance', 'Eigenentropy', 'Anisotropy',
                                     'OptimalKNN', 'RadialDensity', 'classification'])

    target_mapper = {1: 0, 2: 0, 3: 0, 4: 0, 5: 1, 6: 0, 7: 0, 8: 0}
    lidar_df.replace({'classification': target_mapper}, inplace=True)

    return lidar_df


def prepare_data(df_list):
    """
    Concatenate training data and drop unused features
    :param df_list: list of dataframe
    :return: X and y train
    """
    df = pd.concat(df_list, axis=0)
    df.reset_index(drop=True, inplace=True)
    X = df.drop(['x', 'y', 'z', 'Scattering', 'Anisotropy', 'classification'], axis=1)
    y = df['classification']

    return X, y


def train_model(X, y):
    """
    XGBoost training
    :param X: features
    :param y: label
    :return: fitted model
    """
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.7, random_state=42)
    pos_rate = y.value_counts()[0] / y.value_counts()[1]
    model_xgboost = xgboost.XGBClassifier(learning_rate=0.01,
                                          n_estimators=5000,
                                          max_depth=4,
                                          min_child_weight=6,
                                          gamma=1.0,
                                          subsample=0.5,
                                          colsample_bytree=0.4,
                                          colsample_bylevel=0.5,
                                          colsample_bynode=0.5,
                                          reg_alpha=0.01,
                                          objective='binary:logistic',
                                          scale_pos_weight=pos_rate,
                                          nthread=4,
                                          seed=27,
                                          eval_metric='logloss',
                                          verbosity=1)
    eval_set = [(X_train, y_train), (X_valid, y_valid)]

    model_xgboost.fit(X_train,
                      y_train,
                      early_stopping_rounds=10,
                      eval_set=eval_set,
                      verbose=True)

    return model_xgboost


def filter_candidate_clusters(df):
    """
    Postprocessing based on cluster filtering
    :param df: dataframe
    :return: filtered df
    """
    list_cluster = []
    max_ClusterID = np.max(df['ClusterID'])
    for i in range(1, max_ClusterID + 1):
        df_cluster = df[df['ClusterID'] == i]
        try:
            percent = ((df_cluster[df_cluster['prediction'] == 0].shape[0] / df_cluster.shape[0]) * 100)
            # print(percent)
            if percent <= 90:
                list_cluster.append(i)
        except:
            continue

    df_cluster = df[df['ClusterID'].isin(list_cluster)]
    df_cluster.reset_index(drop=True, inplace=True)

    return df_cluster


def predict(df_list, model, filename):
    """
    XGBoost prediction
    :param df_list: list of test dataframe
    :param model: fitted model
    :param filename: filename
    :return:
    """
    test_df = pd.concat(df_list, axis=0)
    test_df.reset_index(drop=True, inplace=True)
    X = test_df.drop(['x', 'y', 'z', 'Scattering', 'Anisotropy', 'classification'], axis=1)
    y_pred = model.predict(X)
    df = test_df[['x', 'y', 'z', 'classification']]
    df['prediction'] = y_pred
    # df = filter_candidate_clusters(df)
    get_IoU_Accuracy(df)
    save_path = 'results/xgboost/' + filename + '.las'
    save_to_las(df, save_path)
    save_to_las(df[df['prediction'] == 1], '2d_projection/' + filename + '.las')


def main(file_to_predict):
    data_dir = 'preprocessed_data/'
    lidar_files = [file for file in listdir(data_dir) if isfile(join(data_dir, file))]
    train_df_list = []
    test_df_list = []
    for i in range(0, len(lidar_files)):
        if lidar_files[i][0:4] == file_to_predict:
            lidar_test = laspy.read(data_dir + lidar_files[i])
            lidar_df = load_lidar_data(lidar_test)
            test_df_list.append(lidar_df)
        else:
            lidar_train = laspy.read(data_dir + lidar_files[i])
            lidar_df = load_lidar_data(lidar_train)
            train_df_list.append(lidar_df)

    trainX, trainY = prepare_data(train_df_list)
    print(trainX.shape)
    print(trainY.shape)
    model = train_model(trainX, trainY)
    predict(test_df_list, model, file_to_predict)


if __name__ == "__main__":
    start_time = time.time()
    file_to_predict = 'L001'
    main(file_to_predict)
    total_time = round((time.time() - start_time), 2)
    print("--- %s seconds ---" % total_time)
