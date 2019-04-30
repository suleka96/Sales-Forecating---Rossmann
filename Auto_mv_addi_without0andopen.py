import tensorflow as tf
import matplotlib as mplt
mplt.use('agg')  # Must be before importing matplotlib.pyplot or pylab!
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from math import sqrt
from sklearn.linear_model import LinearRegression
import csv
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV


class MLConfig():
    input_size = 1

    num_steps = range(2,14)

    fileNames = ['store85_1.csv', 'store519_1.csv', 'store725_1.csv', 'store749_1.csv', 'store165_1.csv',
                 'store925_1.csv', 'store1089_1.csv', 'store335_1.csv']
    column_min_max_all = [[[0, 17000], [1, 7]], [[0, 14000], [1, 7]], [[0, 14000], [1, 7]], [[0, 15000], [1, 7]],
                          [[0, 9000], [1, 7]], [[0, 15000], [1, 7]], [[0, 21000], [1, 7]], [[0, 33000], [1, 7]]]

    columns = ['Sales', 'DayOfWeek', 'SchoolHoliday', 'Promo']
    features = len(columns)



config = MLConfig()

def execution():

    for i in range(len(config.fileNames)):

        fileName = '{}{}'.format('/home/suleka/Documents/sales_prediction/LSTM_Sales_finalRun/', config.fileNames[i])
        name_string_test = '{}{}'.format("/home/suleka/Documents/sales_prediction/LSTM_Sales_finalRun/results_resultsWithoutZero_test_", config.fileNames[i])
        name_string_val = '{}{}'.format("/home/suleka/Documents/sales_prediction/LSTM_Sales_finalRun/results_resultsWithoutZero_validation__result_", config.fileNames[i])


        with open(name_string_test, "a") as f:
            writer = csv.writer(f)
            writer.writerows(zip(["Store"],["Algorithm"], ["Time steps"], ["RMSE"], ["MAE"], ["MAPE"], ["RMSPE"]))

        with open(name_string_val, "a") as f:
            writer = csv.writer(f)
            writer.writerows(zip(["Store"],["Algorithm"], ["Time steps"], ["RMSE"], ["MAE"], ["MAPE"], ["RMSPE"]))

        for n in config.num_steps:
            # Linear_Regression(n, config.fileName[i],config.column_min_max[i],name_string_test,name_string_val)
            Random_Forest_Regressor(n, fileName,config.column_min_max_all[i],name_string_test,name_string_val)
            XGB(n, fileName,config.column_min_max_all[i],name_string_test,name_string_val)




def segmentation(data,num_steps):

    num_steps_with_lable_features = num_steps +1

    seq = [price for tup in data[config.columns].values for price in tup]

    seq = np.array(seq)

    # split into items of features
    seq = [np.array(seq[i * config.features: (i + 1) * config.features])
           for i in range(len(seq) // config.features)]

    # split into groups of num_steps
    temp_X = np.array([seq[i: i + num_steps_with_lable_features] for i in range(len(seq) - num_steps)])

    # replacenig sales of each last time step with 0 as that is what we are going to predict
    for i in range(len(temp_X)):
        temp_X[i][(len(temp_X[i]) - 1)][0] = 0

    X = []

    for dataslice in temp_X:
        temp = dataslice.flatten()
        X.append(temp)

    X = np.asarray(X)

    col_index_to_remove = (num_steps_with_lable_features*config.features)- config.features
    X = np.delete(X, [col_index_to_remove, col_index_to_remove+1,col_index_to_remove+2,col_index_to_remove+3], 1)


    y = np.array([seq[i +  num_steps] for i in range(len(seq) -  num_steps)])

    # get only sales value
    y = [y[i][0] for i in range(len(y))]

    y = np.asarray(y)

    return X, y


def scale(data,column_min_max):

    for i in range (len(column_min_max)):
        data[config.columns[i]] = (data[config.columns[i]] - column_min_max[i][0]) / ((column_min_max[i][1]) - (column_min_max[i][0]))

    return data

def rescle(test_pred, column_min_max):

    prediction = [(pred * (column_min_max[0][1] - column_min_max[0][0])) + column_min_max[0][0] for pred in test_pred]

    return prediction

def convert_log(data):

    for i in range (len(config.column_min_max)):
        data[config.columns[i]] = np.log(data[config.columns[i]])
    return data

def convert_from_log(pred_vals):

    converted_prediction = [ np.exp(pred) for pred in pred_vals]

    return converted_prediction


def pre_process(num_steps, fileName, column_min_max):
    store_data = pd.read_csv(fileName)

    store_data = store_data.drop(store_data[(store_data.Open == 0) & (store_data.Sales == 0)].index)


    validation_len = len(store_data[(store_data.Month == 6) & (store_data.Year == 2015)].index)
    test_len = len(store_data[(store_data.Month == 7) & (store_data.Year == 2015)].index)
    train_size = int(len(store_data) - (validation_len + test_len))

    train_data = store_data[:train_size]
    validation_data = store_data[(train_size - num_steps): validation_len + train_size]
    test_data = store_data[((validation_len + train_size) - num_steps):]
    original_val_data = validation_data.copy()
    original_test_data = test_data.copy()

    # -------------- processing train data---------------------------------------
    scaled_train_data =  scale(train_data,column_min_max)
    train_X, train_y = segmentation(scaled_train_data,num_steps)

    # -------------- processing validation data---------------------------------------
    scaled_validation_data =  scale(validation_data,column_min_max)
    val_X, val_y = segmentation(scaled_validation_data,num_steps)

    # -------------- processing test data---------------------------------------
    scaled_test_data =  scale(test_data,column_min_max)
    test_X, test_y = segmentation(scaled_test_data,num_steps)

    # ----segmenting original validation data-----------------------------------------------
    nonescaled_val_X, nonescaled_val_y = segmentation(original_val_data,num_steps)

    # ----segmenting original test data---------------------------------------------
    nonescaled_test_X, nonescaled_test_y = segmentation(original_test_data,num_steps)

    return train_X, train_y, test_X, test_y, val_X, val_y, nonescaled_test_y, nonescaled_val_y


def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    itemindex = np.where(y_true == 0)
    y_true = np.delete(y_true, itemindex)
    y_pred = np.delete(y_pred, itemindex)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100



def get_scores(name,pred_vals,nonescaled_y,model, num_steps,fileName,name_string):

    # print(name)
    meanSquaredError = mean_squared_error(nonescaled_y, pred_vals)
    rootMeanSquaredError = sqrt(meanSquaredError)
    # print("RMSE:", rootMeanSquaredError)
    mae = mean_absolute_error(nonescaled_y, pred_vals)
    # print("MAE:", mae)
    mape = mean_absolute_percentage_error(nonescaled_y, pred_vals)
    # print("MAPE:", mape)
    rmse_val = RMSPE(nonescaled_y, pred_vals)
    # print("RMSPE:", rmse_val)

    # plot(nonescaled_y,pred_vals,"Random Forest Regressor addi featres mv Prediction Vs Truth.png")
    write_results(rootMeanSquaredError,mae,mape,rmse_val, model,num_steps,fileName,name_string)


def RMSPE(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    itemindex = np.where(y_true == 0)
    y_true = np.delete(y_true, itemindex)
    y_pred = np.delete(y_pred, itemindex)
    return np.sqrt(np.mean(np.square(((y_true - y_pred) / y_pred)), axis=0))

def write_results(rootMeanSquaredError,mae,mape,rmse_val, model,num_steps,fileName,name_string):

    if model == "l":
        algorithm = "linear Regression"

    elif model == "r":
        algorithm = "random forest regressor"
    else:
        algorithm = "xgboost"


    with open(name_string, "a") as f:
        writer = csv.writer(f)
        writer.writerows(zip([fileName], [algorithm], [num_steps], [rootMeanSquaredError], [mae], [mape], [rmse_val]))






def XGB(n, fileName, column_min_max,name_string_test, name_string_val):
    train_X, train_y, test_X, test_y, val_X, val_y, nonescaled_test_y, nonescaled_val_y = pre_process(n, fileName, column_min_max)


    gsc = GridSearchCV(
        estimator= XGBRegressor(),
        param_grid={
            'learning_rate': (0.1, 0.01,0.75),
            'max_depth': range(2, 5),
            'subsample': (0.5, 1,0.1, 0.75),
            'colsample_bytree': (1, 0.1, 0.75),
            'n_estimators': (50, 100, 1000)
        },
        cv=5, scoring='neg_mean_squared_error', verbose=0, n_jobs=-1)

    grid_result = gsc.fit(train_X, train_y)
    best_params = grid_result.best_params_

    print("\nBest: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    xgb = XGBRegressor(learning_rate=best_params["learning_rate"],
                          max_depth=best_params["max_depth"], subsample=best_params["subsample"],
                          colsample_bytree=best_params["colsample_bytree"], n_estimators=best_params["n_estimators"],
                          coef0=0.1, shrinking=True, tol=0.001, cache_size=200, verbose=False, max_iter=-1)

    bst = xgb.fit(train_X,train_y)

    preds = bst.predict(test_X)

    pred_vals = rescle(preds,column_min_max)

    # pred_vals = convert_from_log(preds)

    pred_vals = np.asarray(pred_vals)

    nonescaled_y = nonescaled_test_y.flatten()

    print("--------------- test set ------------------")

    get_scores("---------XGBoost Regressor----------", pred_vals, nonescaled_y,'x',n,fileName,name_string_test)

    # plot(nonescaled_y, pred_vals, "XGBoost Regressor addi featres mv Prediction Vs Truth.png")

    preds = bst.predict(val_X)

    pred_vals = rescle(preds, column_min_max)

    # pred_vals = convert_from_log(preds)

    pred_vals = np.asarray(pred_vals)

    nonescaled_y = nonescaled_val_y.flatten()

    print("--------------- validation set ------------------")

    get_scores("---------XGBoost Regressor----------", pred_vals, nonescaled_y, 'x', n, fileName, name_string_val)


def Random_Forest_Regressor(n, fileName, column_min_max,name_string_test,name_string_val):
    train_X, train_y, test_X, test_y, val_X, val_y, nonescaled_test_y, nonescaled_val_y = pre_process(n, fileName, column_min_max)

    gsc = GridSearchCV(
        estimator=RandomForestRegressor(),
        param_grid={
            'max_depth': range(2, 5),
            'n_estimators': (50, 100, 1000)
        },
        cv=5, scoring='neg_mean_squared_error', verbose=0, n_jobs=-1)

    grid_result = gsc.fit(train_X, train_y)
    best_params = grid_result.best_params_

    print("\nBest: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    rfr = RandomForestRegressor(
                       max_depth=best_params["max_depth"], n_estimators=best_params["n_estimators"],verbose=False)

    rfr.fit(train_X, train_y)

    # rfr.aic(train_X)

    rfr_prediction = rfr.predict(test_X)

    # pred_vals = convert_from_log(rfr_prediction)

    pred_vals = rescle(rfr_prediction, column_min_max)

    pred_vals = np.asarray(pred_vals)

    nonescaled_y=nonescaled_test_y.flatten()

    print("----------- test set --------------------")

    get_scores("---------Random Forest Regressor----------",pred_vals,nonescaled_y,'r',n,fileName,name_string_test)

    rfr_prediction = rfr.predict(val_X)

    # pred_vals = convert_from_log(rfr_prediction)

    pred_vals = rescle(rfr_prediction, column_min_max)

    pred_vals = np.asarray(pred_vals)

    nonescaled_y = nonescaled_val_y.flatten()

    print("----------- vallidation set --------------------")

    get_scores("---------Random Forest Regressor----------", pred_vals, nonescaled_y, 'r', n, fileName, name_string_val)


if __name__ == '__main__':
    execution()