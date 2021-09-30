import pandas as pd
import tensorflow as tf
import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import MinMaxScaler
# from sklearn.svm import SVR,SVC,LinearSVR,LinearSVC
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.naive_bayes import MultinomialNB
# from lightgbm import LGBMClassifier
import time
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler


min_max_scaler = MinMaxScaler()
# np.random.seed(6)
np.random.seed(6)

o = 1
p = 1
patienc=30
m = 0
n = 720
a = 0.8
acc_1 = []
ti_1 = []
acc_2 = []
ti_2 = []
acc_3 = []
ti_3 = []
acc_4 = []
ti_4 = []

acc_f_1 = []
ti_f_1 = []
acc_f_2 = []
ti_f_2 = []
acc_f_3 = []
ti_f_3 = []
acc_f_4 = []
ti_f_4= []



def load_data_det_8(train_data, val_data):


    train_X = train_data[['SF-WAT', 'SA-CFM', 'RA-CFM', 'SA-TEMP', 'MA-TEMP', 'RA-TEMP', 'SA-HUMD', 'RA-HUMD', 'OA-TEMP', 'HWC-EWT','E_ccoil']]
    val_X = val_data[['SF-WAT', 'SA-CFM', 'RA-CFM', 'SA-TEMP', 'MA-TEMP', 'RA-TEMP', 'SA-HUMD', 'RA-HUMD', 'OA-TEMP', 'HWC-EWT', 'E_ccoil']]
    train_X = train_X.values
    val_X = val_X.values

    train_X = train_X.reshape(train_X.shape[0], train_X.shape[1], 1)
    val_X = val_X.reshape(val_X.shape[0], val_X.shape[1], 1)

    train_Y = train_data.iloc[:, 0]
    val_Y = val_data.iloc[:, 0]

    train_Y = np.array(train_Y)
    val_Y = np.array(val_Y)
    train_Y = train_Y.reshape(-1, 1)
    val_Y = val_Y.reshape(-1, 1)

    return train_X, train_Y, val_X, val_Y




lab0 = pd.read_csv(r'D:\study\maching\wxs\AHU\13\0820A.csv')
lab1 = pd.read_csv(r'D:\study\maching\wxs\AHU\13\0821A.csv')
lab2 = pd.read_csv(r'D:\study\maching\wxs\AHU\13\0822A.csv')
lab3 = pd.read_csv(r'D:\study\maching\wxs\AHU\13\0823A.csv')
lab4 = pd.read_csv(r'D:\study\maching\wxs\AHU\13\0824A.csv')
lab5 = pd.read_csv(r'D:\study\maching\wxs\AHU\13\0826A.csv')
lab6 = pd.read_csv(r'D:\study\maching\wxs\AHU\13\0827A.csv')
lab7 = pd.read_csv(r'D:\study\maching\wxs\AHU\13\0831A.csv')
lab8 = pd.read_csv(r'D:\study\maching\wxs\AHU\13\0901A.csv')
lab9 = pd.read_csv(r'D:\study\maching\wxs\AHU\13\0902A.csv')
lab10 = pd.read_csv(r'D:\study\maching\wxs\AHU\13\0906A.csv')
lab11 = pd.read_csv(r'D:\study\maching\wxs\AHU\13\0907A.csv')
lab12 = pd.read_csv(r'D:\study\maching\wxs\AHU\13\0908A.csv')



lab0_1 = lab0.iloc[360:1081,:]
lab1_1 = lab1.iloc[360:1081,:]
lab2_1 = lab2.iloc[360:1081,:]
lab3_1 = lab3.iloc[360:1081,:]
lab4_1 = lab4.iloc[360:1081,:]
lab5_1 = lab5.iloc[360:1081,:]
lab6_1 = lab6.iloc[360:1081,:]
lab7_1 = lab7.iloc[360:1081,:]
lab8_1 = lab8.iloc[360:1081,:]
lab9_1 = lab9.iloc[360:1081,:]
lab10_1 = lab10.iloc[360:1081,:]
lab11_1 = lab11.iloc[360:1081,:]
lab12_1 = lab12.iloc[360:1081,:]






for k in range(0, o):
    for j in range(0, p):
        lab0_2 = lab0_1.iloc[m:m + n, :]
        lab1_2 = lab1_1.iloc[m:m + n, :]
        lab2_2 = lab2_1.iloc[m:m + n, :]
        lab3_2 = lab3_1.iloc[m:m + n, :]
        lab4_2 = lab4_1.iloc[m:m + n, :]
        lab5_2 = lab5_1.iloc[m:m + n, :]
        lab6_2 = lab6_1.iloc[m:m + n, :]
        lab7_2 = lab7_1.iloc[m:m + n, :]
        lab8_2 = lab8_1.iloc[m:m + n, :]
        lab9_2 = lab9_1.iloc[m:m + n, :]
        lab10_2 = lab10_1.iloc[m:m + n, :]
        lab11_2 = lab11_1.iloc[m:m + n, :]
        lab12_2 = lab12_1.iloc[m:m + n, :]






        # 将n0个数据划分为训练集、验证集、测试集(8:1:1)
        lab0_train = lab0_2.iloc[0:int(a * n), :]
        lab0_val = lab0_2.iloc[int(a * n):n, :]

        lab1_train = lab1_2.iloc[0:int(a * n), :]
        lab1_val = lab1_2.iloc[int(a * n):n, :]

        lab2_train = lab2_2.iloc[0:int(a * n), :]
        lab2_val = lab2_2.iloc[int(a * n):n, :]

        lab3_train = lab3_2.iloc[0:int(a * n), :]
        lab3_val = lab3_2.iloc[int(a * n):n, :]

        lab4_train = lab4_2.iloc[0:int(a * n), :]
        lab4_val = lab4_2.iloc[int(a * n):n, :]

        lab5_train = lab5_2.iloc[0:int(a * n), :]
        lab5_val = lab5_2.iloc[int(a * n):n, :]

        lab6_train = lab6_2.iloc[0:int(a * n), :]
        lab6_val = lab6_2.iloc[int(a * n):n, :]


        lab7_train = lab7_2.iloc[0:int(a * n), :]
        lab7_val = lab7_2.iloc[int(a * n):n, :]


        lab8_train = lab8_2.iloc[0:int(a * n), :]
        lab8_val = lab8_2.iloc[int(a * n):n, :]


        lab9_train = lab9_2.iloc[0:int(a * n), :]
        lab9_val = lab9_2.iloc[int(a * n):n, :]

        lab10_train = lab10_2.iloc[0:int(a * n), :]
        lab10_val = lab10_2.iloc[int(a * n):n, :]

        lab11_train = lab11_2.iloc[0:int(a * n), :]
        lab11_val = lab11_2.iloc[int(a * n):n, :]

        lab12_train = lab12_2.iloc[0:int(a * n), :]
        lab12_val = lab12_2.iloc[int(a * n):n, :]






        train_data = pd.concat([lab0_train, lab1_train,lab2_train,lab3_train,lab4_train,lab5_train,lab6_train,
                                       lab7_train,lab8_train,lab9_train,lab10_train,lab11_train,lab12_train], axis=0)

        val_data = pd.concat([lab0_val, lab1_val,lab2_val,lab3_val,lab4_val,lab5_val,lab6_val,
                                       lab7_val,lab8_val,lab9_val,lab10_val,lab11_val,lab12_val], axis=0)




        train_X, train_Y, val_X, val_Y = load_data_det_8(train_data,val_data)
        print('train_X_det1.shape', train_X.shape)
        print('train_Y_det1.shape:', train_Y.shape)
        print('val_Y_det1.shape:', val_Y.shape)

        input_deep = tf.keras.layers.Input(shape=(11, 1))
        hidden1 = tf.keras.layers.Conv1D(filters=32, kernel_size=3, padding='same', activation='relu')(input_deep)
        x = tf.keras.layers.GlobalAvgPool1D()(hidden1)
        x = tf.keras.layers.Dense(int(x.shape[-1]) // 8, activation='relu')(x)
        x = tf.keras.layers.Dense(int(hidden1.shape[-1]), activation='sigmoid')(x)
        x = tf.keras.layers.Multiply()([hidden1, x])
        hidden4 = tf.keras.layers.Conv1D(filters=64, kernel_size=3, padding='same', activation='relu')(x)
        xx = tf.keras.layers.GlobalAvgPool1D()(hidden4)
        xx = tf.keras.layers.Dense(int(xx.shape[-1]) // 8, activation='relu')(xx)
        xx = tf.keras.layers.Dense(int(hidden4.shape[-1]), activation='sigmoid')(xx)
        xx = tf.keras.layers.Multiply()([hidden4, xx])
        hidden7 = tf.keras.layers.Conv1D(filters=128, kernel_size=3, padding='same', activation='relu')(xx)
        hidden10 = tf.keras.layers.Flatten()(hidden7)
        hidden111 = tf.keras.layers.Dense(128, activation='relu')(hidden10)
        dp = tf.keras.layers.Dropout(0.2)(hidden111)
        output = tf.keras.layers.Dense(13, activation='softmax')(dp)
        cnn_se_ahu = tf.keras.models.Model(inputs=input_deep,outputs=[output])
        cnn_se_ahu_det1 = cnn_se_ahu
        cnn_se_ahu_det1.compile(loss="sparse_categorical_crossentropy", optimizer='Nadam', metrics=['accuracy'])
        cnn_se_ahu_det1.summary()
        time1 = time.time()

        ###一般来说机器学习的训练次数会设置到很大，如果模型的表现没有进一步提升，那么训练可以停止了，继续训练很可能会导致过拟合keras.callbacks.EarlyStopping就是用来提前结束训练的。
        callback1 = [tf.keras.callbacks.EarlyStopping(monitor='accuracy', patience=patienc, restore_best_weights=True)]
        history_cnn_se_ahu =cnn_se_ahu_det1.fit(train_X, train_Y,
                                            callbacks=callback1, batch_size=50, epochs=2000, verbose=2)
        cnn_se_ahu_tim = time.time() - time1

        det_cnn_se_ahu = cnn_se_ahu_det1.predict(val_X)
        a_cnn_se_ahu = np.argmax(det_cnn_se_ahu, axis=1)
        a_cnn_se_ahu =  a_cnn_se_ahu.reshape(-1, 1)
        # 输出总的的故障检测与分类精度
        a_cnn_se_ahu_AC = accuracy_score( val_Y, a_cnn_se_ahu)
        acc_1.append(a_cnn_se_ahu_AC)
        ti_1.append(cnn_se_ahu_tim)
        print('a_cnn_se_ahu_AC=', a_cnn_se_ahu_AC)
        print('cnn_se_ahu_tim=', cnn_se_ahu_tim)
        print('\n\n\n\n\n\n\n\n')
        print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
        m += 1000
    cnn_se_ahu_acc_aver = float(sum(acc_1) / len(acc_1))
    cnn_se_ahu_time_aver = float(sum(ti_1) / len(ti_1))
    acc_f_1.append(cnn_se_ahu_acc_aver)
    ti_f_1.append(cnn_se_ahu_time_aver)
    # acc_10.append(level4_acc_aver_500)
    # ti_10.append(level4_time_aver_500)
    print('level1_cnn_se_2000_det1=', acc_1)
    print('level1_cnn_se_2000_time=', ti_1)
    print('level1_cnn_se_acc_aver_2000=',  cnn_se_ahu_acc_aver)
    print('level1_cnn_se_time_aver_2000=', cnn_se_ahu_time_aver)
    m = 0




# result_excel = pd.DataFrame()
# result_excel["level1_acc"] =acc_1
#
# result_excel["level1_time"] = ti_1
#
#
#
#
# writer = pd.ExcelWriter('CNN_SE_AHU_30EPOCH_10CI_8_2.xlsx')
#
# result_excel.to_excel(writer,"流水")
#
#
# writer.save()


print('\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n')


