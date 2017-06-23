"""
2017 web retrieval final project.
See https://github.com/aalexx-S/WR-final
"""
import pandas as pd
from keras import backend as K
import argparse
import sys
import Utils
import numpy as np
from ast import literal_eval
from sklearn.preprocessing import StandardScaler
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout
import sampler
import configparser
import classifier
from keras.models import load_model
from keras.layers import advanced_activations
from keras.callbacks import Callback, EarlyStopping, ModelCheckpoint
from keras.optimizers import Adam
def main(config):
    # read and preprocess of data
    df = pd.read_csv(config.get('GENERAL', 'inputfile'))
    df['Amount_max_fraud'] = 1
    df.loc[df.Amount <= 2125.87, 'Amount_max_fraud'] = 0
    df = df.drop(['V28','V27','V26','V25','V24','V23','V22','V20','V15','V13','V8'], axis =1)
    df['V1_'] = df.V1.map(lambda x: 1 if x < -3 else 0)
    df['V2_'] = df.V2.map(lambda x: 1 if x > 2.5 else 0)
    df['V3_'] = df.V3.map(lambda x: 1 if x < -4 else 0)
    df['V4_'] = df.V4.map(lambda x: 1 if x > 2.5 else 0)
    df['V5_'] = df.V5.map(lambda x: 1 if x < -4.5 else 0)
    df['V6_'] = df.V6.map(lambda x: 1 if x < -2.5 else 0)
    df['V7_'] = df.V7.map(lambda x: 1 if x < -3 else 0)
    df['V9_'] = df.V9.map(lambda x: 1 if x < -2 else 0)
    df['V10_'] = df.V10.map(lambda x: 1 if x < -2.5 else 0)
    df['V11_'] = df.V11.map(lambda x: 1 if x > 2 else 0)
    df['V12_'] = df.V12.map(lambda x: 1 if x < -2 else 0)
    df['V14_'] = df.V14.map(lambda x: 1 if x < -2.5 else 0)
    df['V16_'] = df.V16.map(lambda x: 1 if x < -2 else 0)
    df['V17_'] = df.V17.map(lambda x: 1 if x < -2 else 0)
    df['V18_'] = df.V18.map(lambda x: 1 if x < -2 else 0)
    df['V19_'] = df.V19.map(lambda x: 1 if x > 1.5 else 0)
    df['V21_'] = df.V21.map(lambda x: 1 if x > 0.6 else 0)
    df = df.rename(columns={'Class': 'Fraud'})
    y = df.Fraud
    y = y.values
    y = list(y)
    print('df', df.shape)
    df = df.drop(['Fraud'], axis = 1)
    X = df.values
    X = list(X)
    #splitdata
    train_X, train_y, val_X, val_y\
            = Utils.valicut(X, y, float(config.get('VALIDATE', 'ratio')))
    Utils.verbose_print('Standardizing.')
    train_X = np.array(train_X)
    val_X = np.array(val_X)
    index = []
    for i in range(19):
        index.append(i)
    a = train_X[:,index]
    scaler = StandardScaler().fit(a)
    a = scaler.transform(a)
    b = val_X[:,index]
    b = scaler.transform(b)
    train_X[:,index] = a
    val_X[:,index] = b

    # sample
    Utils.verbose_print('Sampling, method = {0}.'.format(config.get('SAMPLER', 'method')))
    smp = sampler.Smp(config)
    train_X, train_y = smp.fit_sample(train_X, train_y)
    Utils.verbose_print('data size: {0}.'.format(len(train_y)))
    print(train_X.shape)
    print(val_X.shape)
    #building model
    train_y = np.array(train_y)
    val_y = np.array(val_y)
    early = EarlyStopping('val_acc', patience=0)
    checkpoint = ModelCheckpoint(filepath='m3.hdf5'.format(i),
                                     verbose=1,
                                     save_best_only=True,
                                     save_weights_only=True,
                                     monitor='val_acc',
                                     mode='max')
    callbacks = [checkpoint]
    model = Sequential()
    model.add(Dense(256, input_dim=train_X.shape[1], activation=advanced_activations.ELU(alpha=1.0), init='uniform'))#, use_bias=True))
    model.add(Dropout(0.6))
    model.add(Dense(128, activation=advanced_activations.ELU(alpha=1.0), init = 'uniform'))
    model.add(Dropout(0.6))
    model.add(Dense(64, activation=advanced_activations.ELU(alpha=1.0), use_bias=True, init = 'uniform'))
    model.add(Dropout(0.6))
    #model.add(Dense(4, activation='sigmoid', use_bias=True))
    #model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid', init = 'uniform'))
    adam = Adam(lr=0.001,decay=1e-6,clipvalue=0.5)
    model.compile(loss='binary_crossentropy',
              optimizer=adam,
              metrics=['accuracy'])
    model.fit(train_X, train_y,
          epochs=50,
          batch_size=2800,validation_data = (val_X, val_y), verbose = 1,callbacks = callbacks)
    model.load_weights("m3.hdf5")
    # validate
    Utils.verbose_print('Validating.')
    #result_y = clf.predict(val_X)
    result_y = model.predict(val_X)
    print(result_y.shape)
    y = []
    for i in range(len(result_y)):
        y.append(round(result_y[i][0]))
    print(y)
    result_y = y
    correction = Utils.correction(val_y, result_y)
    truth_table = Utils.truth_table(val_y, result_y)
    print('Correction:{0}'.format(correction))
    Utils.verbose_print(Utils.print_truth_table(truth_table))
    f1_score, Precision, Recall = Utils.f1_score(truth_table)
    Utils.verbose_print('F1 Score:{0}'.format(f1_score))
    Utils.verbose_print('Precision:{0}'.format(Precision))
    Utils.verbose_print('Recall:{0}'.format(Recall))



if __name__ == '__main__':
    # parse arguments
    parser = argparse.ArgumentParser(description='WR final program.')
    parser.add_argument('-i', '--input', nargs='?')
    parser.add_argument('-s', '--sample', nargs='?')
    parser.add_argument('-v', '--verbose', action='store_true')
    args = parser.parse_args()

    # read config
    config = configparser.ConfigParser()
    config.read('config.ini')
    config_inputfile = config.get('GENERAL', 'inputfile')
    config_sample_method = config.get('SAMPLER', 'method')
    if args.input:
        config.set('GENERAL', 'inputfile', args.input)
    if args.sample:
        config.set('SAMPLER', 'method', args.sample)
    if args.verbose:
        Utils.set_verbose()

    main(config)
