import skopt
from skopt import gp_minimize, dump
from skopt.space import Real, Categorical, Integer
from skopt.plots import plot_convergence
from skopt.plots import plot_objective, plot_evaluations
from skopt.utils import use_named_args
import matplotlib as mplt
mplt.use('agg')  # Must be before importing matplotlib.pyplot or pylab!
import matplotlib.pyplot as plt
import csv
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from math import sqrt
import atexit
from time import time, strftime, localtime
from datetime import timedelta
from sklearn.metrics import mean_absolute_error


input_size = 1
num_layers = 1
columns = ['Sales', 'DayOfWeek', 'SchoolHoliday', 'Promo']
features = len(columns)
fileName = None
column_min_max = None
Error_file_name = None
Error_plot_name = None

fileNames = ['store85_1', 'store519_1', 'store725_1', 'store749_1', 'store165_1','store925_1','store1089_1','store335_1']

column_min_max_all = [ [[0, 17000], [1, 7]],  [[0, 14000], [1, 7]], [[0, 14000], [1, 7]], [[0, 15000], [1, 7]],[[0, 9000], [1, 7]], [[0, 15000], [1, 7]], [[0, 21000], [1, 7]], [[0, 33000], [1, 7]]]



num_steps = None
lstm_size = None
batch_size = None
init_learning_rate = None
learning_rate_decay = None
init_epoch = None  # 5
max_epoch = None  # 100 or 50
hidden1_nodes = None
hidden2_nodes = None
dropout_rate = None
hidden1_activation = None
hidden2_activation = None
lstm_activation = None

lowest_error = 0.0
start = None
iteration = 0
bestTestPrediction = None
bestValiPrediction = None
bestTestTrueVal = None
bestValiTrueVal = None


lstm_num_steps = Integer(low=2, high=14, name='lstm_num_steps')
size = Integer(low=8, high=128, name='size')
lstm_hidden1_nodes = Integer(low=4, high=64, name='lstm_hidden1_nodes')
lstm_hidden2_nodes = Integer(low=2, high=32, name='lstm_hidden2_nodes')
lstm_learning_rate_decay = Real(low=0.7, high=0.99, prior='uniform', name='lstm_learning_rate_decay')
lstm_max_epoch = Integer(low=60, high=200, name='lstm_max_epoch')
lstm_init_epoch = Integer(low=5, high=50, name='lstm_init_epoch')
lstm_batch_size = Integer(low=5, high=64, name='lstm_batch_size')
lstm_dropout_rate = Real(low=0.1, high=0.9, prior='uniform', name='lstm_dropout_rate')
lstm_init_learning_rate = Real(low=1e-4, high=1e-1, prior='log-uniform', name='lstm_init_learning_rate')
lstm_hidden1_activation = Categorical(categories=[tf.nn.relu, tf.nn.tanh], name='lstm_hidden1_activation')
lstm_hidden2_activation = Categorical(categories=[tf.nn.relu, tf.nn.tanh], name='lstm_hidden2_activation')
lstm_lstm_activation = Categorical(categories=[tf.nn.relu, tf.nn.tanh], name='lstm_lstm_activation')

dimensions = [lstm_num_steps, size, lstm_hidden1_nodes, lstm_hidden2_nodes, lstm_init_epoch, lstm_max_epoch,
              lstm_learning_rate_decay, lstm_batch_size, lstm_dropout_rate, lstm_init_learning_rate,
              lstm_hidden1_activation, lstm_hidden2_activation, lstm_lstm_activation]

default_parameters = [5, 35, 30, 15, 5, 60, 0.99, 8, 0.1, 0.01, tf.nn.relu, tf.nn.relu, tf.nn.relu]


def secondsToStr(elapsed=None):
    if elapsed is None:
        return strftime("%Y-%m-%d %H:%M:%S", localtime())
    else:
        return str(timedelta(seconds=elapsed))


def log(s, elapsed=None):
    line = "=" * 40
    print(line)
    print(secondsToStr(), '-', s)
    if elapsed:
        print("Elapsed time:", elapsed)
    print(line)
    print()


def endlog():
    end = time()
    elapsed = end - start
    log("End Program", secondsToStr(elapsed))


def generate_batches(train_X, train_y, batch_size):
    num_batches = int(len(train_X)) // batch_size
    if batch_size * num_batches < len(train_X):
        num_batches += 1

    batch_indices = range(num_batches)
    for j in batch_indices:
        batch_X = train_X[j * batch_size: (j + 1) * batch_size]
        batch_y = train_y[j * batch_size: (j + 1) * batch_size]
        # assert set(map(len, batch_X)) == {num_steps}
        yield batch_X, batch_y


def segmentation(data):

    seq = [price for tup in data[columns].values for price in tup]

    seq = np.array(seq)

    # split into items of features
    seq = [np.array(seq[i * features: (i + 1) * features])
           for i in range(len(seq) // features)]

    # split into groups of num_steps
    X = np.array([seq[i: i + num_steps] for i in range(len(seq) - num_steps)])

    y = np.array([seq[i + num_steps] for i in range(len(seq) - num_steps)])

    # get only sales value
    y = [[y[i][0]] for i in range(len(y))]

    y = np.asarray(y)

    return X, y


def scale(data):
    for i in range(len(column_min_max)):
        data[columns[i]] = (data[columns[i]] - column_min_max[i][0]) / ((column_min_max[i][1]) - (column_min_max[i][0]))

    return data


def rescle(test_pred):
    prediction = [(pred * (column_min_max[0][1] - column_min_max[0][0])) + column_min_max[0][0] for pred in test_pred]

    return prediction

def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    itemindex = np.where(y_true == 0)
    y_true = np.delete(y_true, itemindex)
    y_pred = np.delete(y_pred, itemindex)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def RMSPE(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    itemindex = np.where(y_true == 0)
    y_true = np.delete(y_true, itemindex)
    y_pred = np.delete(y_pred, itemindex)
    return np.sqrt(np.mean(np.square(((y_true - y_pred) / y_true)), axis=0))


def pre_process():
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
    scaled_train_data = scale(train_data)
    train_X, train_y = segmentation(scaled_train_data)

    # -------------- processing validation data---------------------------------------
    scaled_validation_data = scale(validation_data)
    val_X, val_y = segmentation(scaled_validation_data)

    # -------------- processing test data---------------------------------------
    scaled_test_data = scale(test_data)
    test_X, test_y = segmentation(scaled_test_data)

    # ----segmenting original validation data-----------------------------------------------
    nonescaled_val_X, nonescaled_val_y = segmentation(original_val_data)

    # ----segmenting original test data-----------------------------------------------
    nonescaled_test_X, nonescaled_test_y = segmentation(original_test_data)

    return train_X, train_y, test_X, test_y, val_X, val_y, nonescaled_test_y, nonescaled_val_y


def setupRNN(inputs,model_dropout_rate):

    cell = tf.contrib.rnn.LSTMCell(lstm_size, state_is_tuple=True, activation=lstm_activation,use_peepholes=True)

    val1, _ = tf.nn.dynamic_rnn(cell, inputs, dtype=tf.float32)

    val = tf.transpose(val1, [1, 0, 2])

    last = tf.gather(val, int(val.get_shape()[0]) - 1, name="last_lstm_output")

    # hidden layer
    hidden1 = tf.layers.dense(last, units=hidden1_nodes, activation=hidden2_activation)
    hidden2 = tf.layers.dense(hidden1, units=hidden2_nodes, activation=hidden1_activation)

    dropout = tf.layers.dropout(hidden2, rate=model_dropout_rate, training=True)

    weight = tf.Variable(tf.truncated_normal([hidden2_nodes, input_size]))
    bias = tf.Variable(tf.constant(0.1, shape=[input_size]))

    prediction = tf.nn.relu(tf.matmul(dropout, weight) + bias)

    return prediction


# saver = tf.train.Saver()
# saver.save(sess, "checkpoints_sales/sales_pred.ckpt")


@use_named_args(dimensions=dimensions)
def fitness(lstm_num_steps, size, lstm_hidden1_nodes, lstm_hidden2_nodes, lstm_init_epoch, lstm_max_epoch,
            lstm_learning_rate_decay, lstm_batch_size, lstm_dropout_rate, lstm_init_learning_rate,
            lstm_hidden1_activation, lstm_hidden2_activation, lstm_lstm_activation):
    global  bestValiPrediction,bestValiTrueVal,bestTestPrediction,bestTestTrueVal, iteration, hidden1_activation, hidden2_activation, lstm_activation, lowest_error, num_steps, lstm_size, hidden2_nodes, hidden2_activation, hidden1_activation, hidden1_nodes, lstm_activation, init_epoch, max_epoch, learning_rate_decay, dropout_rate, init_learning_rate

    num_steps = np.int32(lstm_num_steps)
    lstm_size = np.int32(size)
    batch_size = np.int32(lstm_batch_size)
    learning_rate_decay = np.float32(lstm_learning_rate_decay)
    init_epoch = np.int32(lstm_init_epoch)
    max_epoch = np.int32(lstm_max_epoch)
    hidden1_nodes = np.int32(lstm_hidden1_nodes)
    hidden2_nodes = np.int32(lstm_hidden2_nodes)
    dropout_rate = np.float32(lstm_dropout_rate)
    init_learning_rate = np.float32(lstm_init_learning_rate)
    hidden1_activation = lstm_hidden1_activation
    hidden2_activation = lstm_hidden2_activation
    lstm_activation = lstm_lstm_activation

    # log_dir = log_dir_name(lstm_num_steps, size,lstm_hidden1_nodes,lstm_hidden2_nodes,lstm_learning_rate,lstm_init_epoch,lstm_max_epoch,
    #        lstm_learning_rate_decay,lstm_batch_size)

    train_X, train_y, test_X, test_y, val_X, val_y, nonescaled_test_y, nonescaled_val_y = pre_process()

    inputs = tf.placeholder(tf.float32, [None, num_steps, features], name="inputs")
    targets = tf.placeholder(tf.float32, [None, input_size], name="targets")
    model_learning_rate = tf.placeholder(tf.float32, None, name="learning_rate")
    model_dropout_rate = tf.placeholder_with_default(0.0, shape=())
    global_step = tf.Variable(0, trainable=False)

    prediction = setupRNN(inputs,model_dropout_rate)

    model_learning_rate = tf.train.exponential_decay(learning_rate=model_learning_rate, global_step=global_step, decay_rate=learning_rate_decay,
                                               decay_steps=init_epoch, staircase=False)

    with tf.name_scope('loss'):
        model_loss = tf.losses.mean_squared_error(targets, prediction)

    with tf.name_scope('adam_optimizer'):
        train_step = tf.train.AdamOptimizer(model_learning_rate).minimize(model_loss,global_step=global_step)

    train_step = train_step

    # with tf.name_scope('accuracy'):
    #     correct_prediction = tf.sqrt(tf.losses.mean_squared_error(prediction, targets))
    #
    # accuracy = correct_prediction

    sess = tf.Session()

    sess.run(tf.global_variables_initializer())


    for epoch_step in range(max_epoch):


        for batch_X, batch_y in generate_batches(train_X, train_y, batch_size):
            train_data_feed = {
                inputs: batch_X,
                targets: batch_y,
                model_learning_rate: init_learning_rate,
                model_dropout_rate: dropout_rate
            }
            sess.run(train_step, train_data_feed)

    val_data_feed = {
        inputs: val_X,
    }

    vali_pred = sess.run(prediction, val_data_feed)

    vali_pred_vals = rescle(vali_pred)

    vali_pred_vals = np.array(vali_pred_vals)

    vali_pred_vals = (np.round(vali_pred_vals, 0)).astype(np.int32)

    vali_pred_vals = vali_pred_vals.flatten()

    vali_pred_vals = vali_pred_vals.tolist()

    vali_nonescaled_y = nonescaled_val_y.flatten()

    vali_nonescaled_y = vali_nonescaled_y.tolist()

    val_error = sqrt(mean_squared_error(vali_nonescaled_y, vali_pred_vals))

    val_mae = mean_absolute_error(vali_nonescaled_y, vali_pred_vals)
    val_mape = mean_absolute_percentage_error(vali_nonescaled_y, vali_pred_vals)
    val_rmspe = RMSPE(vali_nonescaled_y, vali_pred_vals)



    with open(Error_file_name, "a") as f:
        writer = csv.writer(f)
        writer.writerows(
            zip([fileName], [val_error], [val_mae], [val_mape],[val_rmspe]))

    if iteration == 0:
        lowest_error = val_error

        test_data_feed = {
            inputs: test_X,
        }

        test_pred = sess.run(prediction, test_data_feed)

        test_pred_vals = rescle(test_pred)

        test_pred_vals = np.array(test_pred_vals)

        test_pred_vals = (np.round(test_pred_vals, 0)).astype(np.int32)

        test_pred_vals = test_pred_vals.flatten()

        test_pred_vals = test_pred_vals.tolist()

        test_nonescaled_y = nonescaled_test_y.flatten()

        test_nonescaled_y = test_nonescaled_y.tolist()

        test_error = sqrt(mean_squared_error(test_nonescaled_y, test_pred_vals))

        test_mae = mean_absolute_error(test_nonescaled_y, test_pred_vals)
        test_mape = mean_absolute_percentage_error(test_nonescaled_y, test_pred_vals)
        test_rmspe = RMSPE(test_nonescaled_y, test_pred_vals)

        with open("best_withoutZero_config.csv", "a") as f:
            writer = csv.writer(f)
            writer.writerows(
                zip([fileName], [num_steps], [lstm_size], [hidden2_nodes], [hidden2_activation], [hidden1_activation],
                    [hidden1_nodes], [lstm_activation], [init_epoch], [max_epoch], [learning_rate_decay],
                    [dropout_rate], [batch_size], [init_learning_rate], [val_error],[val_mae], [val_mape],[val_rmspe], [test_error],[test_mae],[test_mape],[test_rmspe]))





    elif val_error < lowest_error:
        # Save the new model to harddisk.
        saver = tf.train.Saver()
        saver.save(sess, "checkpoints_sales/sales_pred.ckpt")

        test_data_feed = {
            inputs: test_X,
        }

        test_pred = sess.run(prediction, test_data_feed)

        test_pred_vals = rescle(test_pred)

        test_pred_vals = np.array(test_pred_vals)

        test_pred_vals = (np.round(test_pred_vals, 0)).astype(np.int32)

        test_pred_vals = test_pred_vals.flatten()

        test_pred_vals = test_pred_vals.tolist()

        test_nonescaled_y = nonescaled_test_y.flatten()

        test_nonescaled_y = test_nonescaled_y.tolist()

        bestValiPrediction = vali_pred_vals
        bestValiTrueVal = vali_nonescaled_y
        bestTestPrediction = test_pred_vals
        bestTestTrueVal =  test_nonescaled_y

        test_error = sqrt(mean_squared_error(test_nonescaled_y, test_pred_vals))

        test_mae = mean_absolute_error(test_nonescaled_y, test_pred_vals)
        test_mape = mean_absolute_percentage_error(test_nonescaled_y, test_pred_vals)
        test_rmspe = RMSPE(test_nonescaled_y, test_pred_vals)

        with open("best_withoutZero_config.csv", "a") as f:
            writer = csv.writer(f)
            writer.writerows(
                zip([fileName], [num_steps], [lstm_size], [hidden2_nodes], [hidden2_activation], [hidden1_activation],
                    [hidden1_nodes], [lstm_activation], [init_epoch], [max_epoch], [learning_rate_decay],
                    [dropout_rate], [batch_size], [init_learning_rate], [val_error],[val_mae], [val_mape],[val_rmspe],[test_error],[test_mae],[test_mape],[test_rmspe]))


        # Update the classification accuracy.
        lowest_error = val_error

    sess.close()
    tf.reset_default_graph()



    iteration += 1
    return val_error


if __name__ == '__main__':

    print("new")

    start = time()

    for i in range(len(fileNames)):
        iteration = 0
        lowest_error = 0.0

        fileName = '{}{}{}'.format('/home/wso2/suleka/salesPred/', fileNames[i],'.csv')
        Error_file_name = '{}{}{}'.format('all_validation_errors/errors_', fileNames[i],'.csv')
        vali_data = '{}{}{}'.format('validation_data/vali_data_', fileNames[i],'.csv')
        predic_data = '{}{}{}'.format('prediction_data/predic_data_', fileNames[i],'.csv')
        Skopt_object_name = '{}{}{}'.format('/home/wso2/suleka/salesPred/skopt_objects/object_', fileNames[i],'.gz')


        column_min_max = column_min_max_all[i]

        search_result = gp_minimize(func=fitness,
                                    dimensions=dimensions,
                                    acq_func='EI',  # Expected Improvement.
                                    n_calls=300,
                                    x0=default_parameters)

        with open(vali_data, "w") as f:
            writer = csv.writer(f)
            writer.writerows(zip(bestValiTrueVal, bestValiPrediction))

        with open(predic_data, "w") as f:
            writer = csv.writer(f)
            writer.writerows(zip(bestTestTrueVal,bestTestPrediction))

        bestTestPrediction = None
        bestValiPrediction = None
        bestTestTrueVal = None
        bestValiTrueVal = None


        dump(search_result,Skopt_object_name , store_objective=True)


        print(search_result.x)
        print(search_result.fun)


    atexit.register(endlog)
    log("Start Program")

