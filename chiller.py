#########################选取500个    2000-1500   8+1.5+0.5
import pandas as pd
import tensorflow as tf
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import time
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler

min_max_scaler = MinMaxScaler()
# np.random.seed(6)
np.random.seed(6)
ma1 = 1
re1 = 1

ma2 = 1
re2 = 1

ma3 = 1
re3 = 1

ma4 = 1
re4 = 1


o = 10
p = 1
patienc=1
m = 0
n = 5000
a = 0.7
b = 0.85
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

def load_data_det_8(train_data, val_data, test_data):
    #     train_X = train_data[['FWC', 'FWE', 'Unit Status', 'PO_feed', 'PO_net', 'TWCD', 'VE', 'TWI']]
    #     val_X = val_data[['FWC', 'FWE', 'Unit Status', 'PO_feed', 'PO_net', 'TWCD', 'VE', 'TWI']]
    #     test_X = test_data[['FWC', 'FWE', 'Unit Status', 'PO_feed', 'PO_net', 'TWCD', 'VE', 'TWI']]
    ### 选择8个特征
    # train_X = train_data[
    #     ['TEO', 'TCO', 'FWC', 'VE', 'TRC', 'TR_dis', 'PO_feed', 'TWI']]
    # val_X = val_data[
    #     ['TEO', 'TCO', 'FWC', 'VE', 'TRC', 'TR_dis', 'PO_feed', 'TWI']]
    # test_X = test_data[
    #     ['TEO', 'TCO', 'FWC', 'VE', 'TRC', 'TR_dis', 'PO_feed', 'TWI']]

    #########腾哥的特征选择
    train_X = train_data[['FWC', 'FWE', 'TCA', 'Tolerance%', 'PO_feed', 'PO_net', 'TWCD', 'VE']]
    val_X = val_data[['FWC', 'FWE', 'TCA', 'Tolerance%', 'PO_feed', 'PO_net', 'TWCD', 'VE']]
    test_X = test_data[['FWC', 'FWE', 'TCA', 'Tolerance%', 'PO_feed', 'PO_net', 'TWCD', 'VE']]
    train_X = train_X.values
    val_X = val_X.values
    test_X = test_X.values

    train_X = train_X.reshape(train_X.shape[0], train_X.shape[1], 1)
    val_X = val_X.reshape(val_X.shape[0], val_X.shape[1], 1)
    test_X = test_X.reshape(test_X.shape[0], test_X.shape[1], 1)

    train_Y = train_data.iloc[:, 0]
    val_Y = val_data.iloc[:, 0]
    test_Y = test_data.iloc[:, 0]

    train_Y = np.array(train_Y)
    val_Y = np.array(val_Y)
    test_Y = np.array(test_Y)
    train_Y = train_Y.reshape(-1, 1)
    val_Y = val_Y.reshape(-1, 1)
    test_Y = test_Y.reshape(-1, 1)

    return train_X, train_Y, val_X, val_Y, test_X, test_Y




#############################故障等级一###############################################################################################
###故障等级1中，读取原始数据集
cnn_se_lev1_lab0 = pd.read_csv(r'D:\study\maching\brother\tengge\新想法1\数据集孙\level1\lev1_lab0.csv')  # D:\zhuo mian wen jian\maching\brother\tengge\新想法1\数据集孙
cnn_se_lev1_lab1 = pd.read_csv(r'D:\study\maching\brother\tengge\新想法1\数据集孙\level1\lev1_lab1.csv')
cnn_se_lev1_lab2 = pd.read_csv(r'D:\study\maching\brother\tengge\新想法1\数据集孙\level1\lev1_lab2.csv')
cnn_se_lev1_lab3 = pd.read_csv(r'D:\study\maching\brother\tengge\新想法1\数据集孙\level1\lev1_lab3.csv')
cnn_se_lev1_lab4 = pd.read_csv(r'D:\study\maching\brother\tengge\新想法1\数据集孙\level1\lev1_lab4.csv')
cnn_se_lev1_lab5 = pd.read_csv(r'D:\study\maching\brother\tengge\新想法1\数据集孙\level1\lev1_lab5.csv')
cnn_se_lev1_lab6 = pd.read_csv(r'D:\study\maching\brother\tengge\新想法1\数据集孙\level1\lev1_lab6.csv')
cnn_se_lev1_lab7 = pd.read_csv(r'D:\study\maching\brother\tengge\新想法1\数据集孙\level1\lev1_lab7.csv')

# 丢弃包含NAN的数据
cnn_se_lev1_lab0 = cnn_se_lev1_lab0.dropna()
cnn_se_lev1_lab1 = cnn_se_lev1_lab1.dropna()
cnn_se_lev1_lab2 = cnn_se_lev1_lab2.dropna()
cnn_se_lev1_lab3 = cnn_se_lev1_lab3.dropna()
cnn_se_lev1_lab4 = cnn_se_lev1_lab4.dropna()
cnn_se_lev1_lab5 = cnn_se_lev1_lab5.dropna()
cnn_se_lev1_lab6 = cnn_se_lev1_lab6.dropna()
cnn_se_lev1_lab7 = cnn_se_lev1_lab7.dropna()




for k in range(0, o):
    for j in range(0, p):
        cnn_se_lev1_lab0_1 = cnn_se_lev1_lab0.iloc[m:m + n, :]
        cnn_se_lev1_lab1_1 = cnn_se_lev1_lab1.iloc[m:m + n, :]
        cnn_se_lev1_lab2_1 = cnn_se_lev1_lab2.iloc[m:m + n, :]
        cnn_se_lev1_lab3_1 = cnn_se_lev1_lab3.iloc[m:m + n, :]
        cnn_se_lev1_lab4_1 = cnn_se_lev1_lab4.iloc[m:m + n, :]
        cnn_se_lev1_lab5_1 = cnn_se_lev1_lab5.iloc[m:m + n, :]
        cnn_se_lev1_lab6_1 = cnn_se_lev1_lab6.iloc[m:m + n, :]
        cnn_se_lev1_lab7_1 = cnn_se_lev1_lab7.iloc[m:m + n, :]

        # 将n0个数据划分为训练集、验证集、测试集(8:1:1)
        cnn_se_lev1_lab0_train = cnn_se_lev1_lab0_1.iloc[0:int(a * n), :]
        cnn_se_lev1_lab0_val = cnn_se_lev1_lab0_1.iloc[int(a * n):int(b * n), :]
        cnn_se_lev1_lab0_test = cnn_se_lev1_lab0_1.iloc[int(b * n):n, :]

        cnn_se_lev1_lab1_train = cnn_se_lev1_lab1_1.iloc[0:int(a * n), :]
        cnn_se_lev1_lab1_val = cnn_se_lev1_lab1_1.iloc[int(a * n):int(b * n), :]
        cnn_se_lev1_lab1_test = cnn_se_lev1_lab1_1.iloc[int(b * n):n, :]

        cnn_se_lev1_lab2_train = cnn_se_lev1_lab2_1.iloc[0:int(a * n), :]
        cnn_se_lev1_lab2_val = cnn_se_lev1_lab2_1.iloc[int(a * n):int(b * n), :]
        cnn_se_lev1_lab2_test = cnn_se_lev1_lab2_1.iloc[int(b * n):n, :]

        cnn_se_lev1_lab3_train = cnn_se_lev1_lab3_1.iloc[0:int(a * n), :]
        cnn_se_lev1_lab3_val = cnn_se_lev1_lab3_1.iloc[int(a * n):int(b * n), :]
        cnn_se_lev1_lab3_test = cnn_se_lev1_lab3_1.iloc[int(b * n):n, :]

        cnn_se_lev1_lab4_train = cnn_se_lev1_lab4_1.iloc[0:int(a * n), :]
        cnn_se_lev1_lab4_val = cnn_se_lev1_lab4_1.iloc[int(a * n):int(b * n), :]
        cnn_se_lev1_lab4_test = cnn_se_lev1_lab4_1.iloc[int(b * n):n, :]

        cnn_se_lev1_lab5_train = cnn_se_lev1_lab5_1.iloc[0:int(a * n), :]
        cnn_se_lev1_lab5_val = cnn_se_lev1_lab5_1.iloc[int(a * n):int(b * n), :]
        cnn_se_lev1_lab5_test = cnn_se_lev1_lab5_1.iloc[int(b * n):n, :]

        cnn_se_lev1_lab6_train = cnn_se_lev1_lab6_1.iloc[0:int(a * n), :]
        cnn_se_lev1_lab6_val = cnn_se_lev1_lab6_1.iloc[int(a * n):int(b * n), :]
        cnn_se_lev1_lab6_test = cnn_se_lev1_lab6_1.iloc[int(b * n):n, :]

        cnn_se_lev1_lab7_train = cnn_se_lev1_lab7_1.iloc[0:int(a * n), :]
        cnn_se_lev1_lab7_val = cnn_se_lev1_lab7_1.iloc[int(a * n):int(b * n), :]
        cnn_se_lev1_lab7_test = cnn_se_lev1_lab7_1.iloc[int(b * n):n, :]

        # 合并训练集测试集
        cnn_se_lev1_train = pd.concat([cnn_se_lev1_lab0_train, cnn_se_lev1_lab1_train, cnn_se_lev1_lab2_train, cnn_se_lev1_lab3_train,
                                cnn_se_lev1_lab4_train, cnn_se_lev1_lab5_train, cnn_se_lev1_lab6_train, cnn_se_lev1_lab7_train], axis=0)

        cnn_se_lev1_val = pd.concat([cnn_se_lev1_lab0_val, cnn_se_lev1_lab1_val, cnn_se_lev1_lab2_val, cnn_se_lev1_lab3_val, cnn_se_lev1_lab4_val,
                              cnn_se_lev1_lab5_val, cnn_se_lev1_lab6_val, cnn_se_lev1_lab7_val], axis=0)

        cnn_se_lev1_test = pd.concat([cnn_se_lev1_lab0_test, cnn_se_lev1_lab1_test, cnn_se_lev1_lab2_test, cnn_se_lev1_lab3_test,
                               cnn_se_lev1_lab4_test, cnn_se_lev1_lab5_test, cnn_se_lev1_lab6_test, cnn_se_lev1_lab7_test], axis=0)
        print('cnn_se_lev1_train.shape', cnn_se_lev1_train.shape)
        print('cnn_se_lev1_val.shape', cnn_se_lev1_val.shape)
        print('cnn_se_lev1_test.shape', cnn_se_lev1_test.shape)

        train_X_det1, train_Y_det1, val_X_det1, val_Y_det1, test_X_det1, test_Y_det1 = load_data_det_8(cnn_se_lev1_train,cnn_se_lev1_val,cnn_se_lev1_test)
        print('train_X_det1.shape', train_X_det1.shape)
        print(train_X_det1)
        print('train_Y_det1.shape:', train_Y_det1.shape)
        print('val_X_det1.shape', val_X_det1.shape)
        print('val_Y_det1.shape:', val_Y_det1.shape)
        print('test_X_det1.shape', test_X_det1.shape)
        print('test_Y_det1.shape:', test_Y_det1.shape)

        input_deep = tf.keras.layers.Input(shape=(8, 1))
        hidden1 = tf.keras.layers.Conv1D(filters=32, kernel_size=3, padding='same', activation='relu')(input_deep)
        # hidden2 = tf.keras.layers.Conv1D(filters = 32,kernel_size = 3,padding = 'same',activation = 'relu')(hidden1)
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
        ###全连接层
        output = tf.keras.layers.Dense(8, activation='softmax')(dp)

        cnn_se_lev1_2000 = tf.keras.models.Model(inputs=input_deep,outputs=[output])
        cnn_se_lev1_2000_det1 = cnn_se_lev1_2000
        cnn_se_lev1_2000_det1.compile(loss="sparse_categorical_crossentropy", optimizer='Nadam', metrics=['accuracy'])
        cnn_se_lev1_2000_det1.summary()

        time1 = time.time()

        ###一般来说机器学习的训练次数会设置到很大，如果模型的表现没有进一步提升，那么训练可以停止了，继续训练很可能会导致过拟合keras.callbacks.EarlyStopping就是用来提前结束训练的。
        callback1 = [tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=patienc, restore_best_weights=True)]
        history_cnn_se_lev1_2000 =cnn_se_lev1_2000_det1.fit(train_X_det1, train_Y_det1, validation_data=(val_X_det1, val_Y_det1),
                                            callbacks=callback1, batch_size=50, epochs=2000, verbose=2)
        cnn_se_lev1_2000_tim = time.time() - time1

        det_cnn_se_lev1_2000 = cnn_se_lev1_2000_det1.predict(test_X_det1)
        print('det_cnn_se_lev1_2000 =',det_cnn_se_lev1_2000 )
        print('det_cnn_se_lev1_2000.shape =', det_cnn_se_lev1_2000.shape)
        a_cnn_se_lev1_2000 = np.argmax(det_cnn_se_lev1_2000, axis=1)
        print('da_cnn_se_lev1_2000 =',a_cnn_se_lev1_2000 )
        print('a_cnn_se_lev1_2000.shape =', a_cnn_se_lev1_2000.shape)
        a_cnn_se_lev1_2000 =  a_cnn_se_lev1_2000.reshape(-1, 1)
        # 输出总的的故障检测与分类精度
        a_cnn_se_lev1_2000_AC = accuracy_score(test_Y_det1, a_cnn_se_lev1_2000)
        acc_1.append(a_cnn_se_lev1_2000_AC)
        ti_1.append(cnn_se_lev1_2000_tim)

        level1_cfm = confusion_matrix(test_Y_det1, a_cnn_se_lev1_2000)
        level1_conf = pd.DataFrame(level1_cfm).transpose()
        level1_conf.to_csv("level1_confusion_matrix" + str(ma1) + ".csv", index=True)
        print(level1_cfm)
        level1_report = classification_report(test_Y_det1, a_cnn_se_lev1_2000, output_dict=True)
        level1_df = pd.DataFrame(level1_report).transpose()
        level1_df.to_csv("level1_result" + str(re1) + ".csv", index=True)







        print('a_cnn_se_lev1_2000_AC=', a_cnn_se_lev1_2000_AC)
        print('cnn_se_lev1_2000_tim=', cnn_se_lev1_2000_tim)
        print('\n\n\n\n\n\n\n\n')
        print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
        m += 1000
        ma1 += 1
        re1 += 1
    cnn_se_lev1_2000_acc_aver = float(sum(acc_1) / len(acc_1))
    cnn_se_lev1_2000_time_aver = float(sum(ti_1) / len(ti_1))
    acc_f_1.append(cnn_se_lev1_2000_acc_aver)
    ti_f_1.append(cnn_se_lev1_2000_time_aver)
    # acc_10.append(level4_acc_aver_500)
    # ti_10.append(level4_time_aver_500)
    print('level1_cnn_se_2000_det1=', acc_1)
    print('level1_cnn_se_2000_time=', ti_1)
    print('level1_cnn_se_acc_aver_2000=',  cnn_se_lev1_2000_acc_aver)
    print('level1_cnn_se_time_aver_2000=', cnn_se_lev1_2000_time_aver)
    m = 0


print('\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n')




############################故障等级二###############################################################################################
##故障等级2中，读取原始数据集

cnn_se_lev2_lab0 = pd.read_csv(r'D:\study\maching\brother\tengge\新想法1\数据集孙\level2\lev2_lab0.csv')
cnn_se_lev2_lab1 = pd.read_csv(r'D:\study\maching\brother\tengge\新想法1\数据集孙\level2\lev2_lab1.csv')
cnn_se_lev2_lab2 = pd.read_csv(r'D:\study\maching\brother\tengge\新想法1\数据集孙\level2\lev2_lab2.csv')
cnn_se_lev2_lab3 = pd.read_csv(r'D:\study\maching\brother\tengge\新想法1\数据集孙\level2\lev2_lab3.csv')
cnn_se_lev2_lab4 = pd.read_csv(r'D:\study\maching\brother\tengge\新想法1\数据集孙\level2\lev2_lab4.csv')
cnn_se_lev2_lab5 = pd.read_csv(r'D:\study\maching\brother\tengge\新想法1\数据集孙\level2\lev2_lab5.csv')
cnn_se_lev2_lab6 = pd.read_csv(r'D:\study\maching\brother\tengge\新想法1\数据集孙\level2\lev2_lab6.csv')
cnn_se_lev2_lab7 = pd.read_csv(r'D:\study\maching\brother\tengge\新想法1\数据集孙\level2\lev2_lab7.csv')

# 丢弃包含NAN的数据
cnn_se_lev2_lab0 = cnn_se_lev2_lab0.dropna()
cnn_se_lev2_lab1 = cnn_se_lev2_lab1.dropna()
cnn_se_lev2_lab2 = cnn_se_lev2_lab2.dropna()
cnn_se_lev2_lab3 = cnn_se_lev2_lab3.dropna()
cnn_se_lev2_lab4 = cnn_se_lev2_lab4.dropna()
cnn_se_lev2_lab5 = cnn_se_lev2_lab5.dropna()
cnn_se_lev2_lab6 = cnn_se_lev2_lab6.dropna()
cnn_se_lev2_lab7 = cnn_se_lev2_lab7.dropna()

m = 0
for k in range(0, o):
    for j in range(0, p):
        cnn_se_lev2_lab0_1 = cnn_se_lev2_lab0.iloc[m:m + n, :]
        cnn_se_lev2_lab1_1 = cnn_se_lev2_lab1.iloc[m:m + n, :]
        cnn_se_lev2_lab2_1 = cnn_se_lev2_lab2.iloc[m:m + n, :]
        cnn_se_lev2_lab3_1 = cnn_se_lev2_lab3.iloc[m:m + n, :]
        cnn_se_lev2_lab4_1 = cnn_se_lev2_lab4.iloc[m:m + n, :]
        cnn_se_lev2_lab5_1 = cnn_se_lev2_lab5.iloc[m:m + n, :]
        cnn_se_lev2_lab6_1 = cnn_se_lev2_lab6.iloc[m:m + n, :]
        cnn_se_lev2_lab7_1 = cnn_se_lev2_lab7.iloc[m:m + n, :]

        # 将n0个数据划分为训练集、验证集、测试集(8:1:1)
        cnn_se_lev2_lab0_train = cnn_se_lev2_lab0_1.iloc[0:int(a * n), :]
        cnn_se_lev2_lab0_val = cnn_se_lev2_lab0_1.iloc[int(a * n):int(b * n), :]
        cnn_se_lev2_lab0_test = cnn_se_lev2_lab0_1.iloc[int(b * n):n, :]

        cnn_se_lev2_lab1_train = cnn_se_lev2_lab1_1.iloc[0:int(a * n), :]
        cnn_se_lev2_lab1_val = cnn_se_lev2_lab1_1.iloc[int(a * n):int(b * n), :]
        cnn_se_lev2_lab1_test = cnn_se_lev2_lab1_1.iloc[int(b * n):n, :]

        cnn_se_lev2_lab2_train = cnn_se_lev2_lab2_1.iloc[0:int(a * n), :]
        cnn_se_lev2_lab2_val = cnn_se_lev2_lab2_1.iloc[int(a * n):int(b * n), :]
        cnn_se_lev2_lab2_test = cnn_se_lev2_lab2_1.iloc[int(b * n):n, :]

        cnn_se_lev2_lab3_train = cnn_se_lev2_lab3_1.iloc[0:int(a * n), :]
        cnn_se_lev2_lab3_val = cnn_se_lev2_lab3_1.iloc[int(a * n):int(b * n), :]
        cnn_se_lev2_lab3_test = cnn_se_lev2_lab3_1.iloc[int(b * n):n, :]

        cnn_se_lev2_lab4_train = cnn_se_lev2_lab4_1.iloc[0:int(a * n), :]
        cnn_se_lev2_lab4_val = cnn_se_lev2_lab4_1.iloc[int(a * n):int(b * n), :]
        cnn_se_lev2_lab4_test = cnn_se_lev2_lab4_1.iloc[int(b * n):n, :]

        cnn_se_lev2_lab5_train = cnn_se_lev2_lab5_1.iloc[0:int(a * n), :]
        cnn_se_lev2_lab5_val = cnn_se_lev2_lab5_1.iloc[int(a * n):int(b * n), :]
        cnn_se_lev2_lab5_test = cnn_se_lev2_lab5_1.iloc[int(b * n):n, :]

        cnn_se_lev2_lab6_train = cnn_se_lev2_lab6_1.iloc[0:int(a * n), :]
        cnn_se_lev2_lab6_val = cnn_se_lev2_lab6_1.iloc[int(a * n):int(b * n), :]
        cnn_se_lev2_lab6_test = cnn_se_lev2_lab6_1.iloc[int(b * n):n, :]

        cnn_se_lev2_lab7_train = cnn_se_lev2_lab7_1.iloc[0:int(a * n), :]
        cnn_se_lev2_lab7_val = cnn_se_lev2_lab7_1.iloc[int(a * n):int(b * n), :]
        cnn_se_lev2_lab7_test = cnn_se_lev2_lab7_1.iloc[int(b * n):n, :]

        # 合并训练集测试集
        cnn_se_lev2_train = pd.concat([cnn_se_lev2_lab0_train, cnn_se_lev2_lab1_train, cnn_se_lev2_lab2_train, cnn_se_lev2_lab3_train,
                                cnn_se_lev2_lab4_train, cnn_se_lev2_lab5_train, cnn_se_lev2_lab6_train, cnn_se_lev2_lab7_train], axis=0)

        cnn_se_lev2_val = pd.concat([cnn_se_lev2_lab0_val, cnn_se_lev2_lab1_val, cnn_se_lev2_lab2_val, cnn_se_lev2_lab3_val, cnn_se_lev2_lab4_val,
                              cnn_se_lev2_lab5_val, cnn_se_lev2_lab6_val, cnn_se_lev2_lab7_val], axis=0)

        cnn_se_lev2_test = pd.concat([cnn_se_lev2_lab0_test, cnn_se_lev2_lab1_test, cnn_se_lev2_lab2_test, cnn_se_lev2_lab3_test,
                               cnn_se_lev2_lab4_test, cnn_se_lev2_lab5_test, cnn_se_lev2_lab6_test, cnn_se_lev2_lab7_test], axis=0)
        print('cnn_se_lev2_train.shape', cnn_se_lev2_train.shape)
        print('cnn_se_lev2_val.shape', cnn_se_lev2_val.shape)
        print('cnn_se_lev2_test.shape', cnn_se_lev2_test.shape)

        train_X_det2, train_Y_det2, val_X_det2, val_Y_det2, test_X_det2, test_Y_det2 = load_data_det_8(cnn_se_lev2_train,cnn_se_lev2_val,cnn_se_lev2_test)
        print('train_X_det2.shape', train_X_det2.shape)
        print('train_Y_det2.shape:', train_Y_det2.shape)
        print('val_X_det2.shape', val_X_det2.shape)
        print('val_Y_det2.shape:', val_Y_det2.shape)
        print('test_X_det2.shape', test_X_det2.shape)
        print('test_Y_det2.shape:', test_Y_det2.shape)

        input_deep = tf.keras.layers.Input(shape=(8, 1))
        hidden1 = tf.keras.layers.Conv1D(filters=32, kernel_size=3, padding='same', activation='relu')(input_deep)
        # hidden2 = tf.keras.layers.Conv1D(filters = 32,kernel_size = 3,padding = 'same',activation = 'relu')(hidden1)
        x = tf.keras.layers.GlobalAvgPool1D()(hidden1)
        x = tf.keras.layers.Dense(int(x.shape[-1]) // 8, activation='relu')(x)
        x = tf.keras.layers.Dense(int(hidden1.shape[-1]), activation='sigmoid')(x)
        x = tf.keras.layers.Multiply()([hidden1, x])

        hidden4 = tf.keras.layers.Conv1D(filters=64, kernel_size=3, padding='same', activation='relu')(x)
        # hidden5 = tf.keras.layers.Conv1D(filters = 64,kernel_size = 3,padding = 'same',activation = 'relu')(hidden4)
        xx = tf.keras.layers.GlobalAvgPool1D()(hidden4)
        print(xx.shape)
        xx = tf.keras.layers.Dense(int(xx.shape[-1]) // 8, activation='relu')(xx)
        xx = tf.keras.layers.Dense(int(hidden4.shape[-1]), activation='sigmoid')(xx)
        xx = tf.keras.layers.Multiply()([hidden4, xx])

        hidden7 = tf.keras.layers.Conv1D(filters=128, kernel_size=3, padding='same', activation='relu')(xx)
        hidden10 = tf.keras.layers.Flatten()(hidden7)
        hidden111 = tf.keras.layers.Dense(128, activation='relu')(hidden10)
        dp = tf.keras.layers.Dropout(0.2)(hidden111)
        ###全连接层
        output = tf.keras.layers.Dense(8, activation='softmax')(dp)

        cnn_se_lev2_2000 = tf.keras.models.Model(inputs=input_deep,outputs=[output])
        cnn_se_lev2_2000_det2 = cnn_se_lev2_2000
        cnn_se_lev2_2000_det2.compile(loss="sparse_categorical_crossentropy", optimizer='Nadam', metrics=['accuracy'])

        # In[15]:
        # sum = 0
        # for i in range(0,3):
        time1 = time.time()

        ###一般来说机器学习的训练次数会设置到很大，如果模型的表现没有进一步提升，那么训练可以停止了，继续训练很可能会导致过拟合keras.callbacks.EarlyStopping就是用来提前结束训练的。
        callback1 = [tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=patienc, restore_best_weights=True)]
        history_cnn_se_lev2_2000 =cnn_se_lev2_2000_det2.fit(train_X_det2, train_Y_det2, validation_data=(val_X_det2, val_Y_det2),
                                            callbacks=callback1, batch_size=50, epochs=2000, verbose=2)
        cnn_se_lev2_2000_tim = time.time() - time1

        det_cnn_se_lev2_2000 = cnn_se_lev2_2000_det2.predict(test_X_det2)
        a_cnn_se_lev2_2000 = np.argmax(det_cnn_se_lev2_2000, axis=1)
        a_cnn_se_lev2_2000 =  a_cnn_se_lev2_2000.reshape(-1, 1)
        # 输出总的的故障检测与分类精度
        a_cnn_se_lev2_2000_AC = accuracy_score(test_Y_det2, a_cnn_se_lev2_2000)
        acc_2.append(a_cnn_se_lev2_2000_AC)
        ti_2.append(cnn_se_lev2_2000_tim)

        level2_cfm = confusion_matrix(test_Y_det2, a_cnn_se_lev2_2000)
        level2_conf = pd.DataFrame(level2_cfm).transpose()
        level2_conf.to_csv("level2_confusion_matrix" + str(ma2) + ".csv", index=True)
        print(level2_cfm)
        level2_report = classification_report(test_Y_det2, a_cnn_se_lev2_2000, output_dict=True)
        level2_df = pd.DataFrame(level2_report).transpose()
        level2_df.to_csv("level2_result" + str(re2) + ".csv", index=True)



        print('a_cnn_se_lev2_2000_AC=', a_cnn_se_lev2_2000_AC)
        print('cnn_se_lev2_2000_tim=', cnn_se_lev2_2000_tim)
        print('\n\n\n\n\n\n\n\n')
        print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
        m += 1000
        ma2 += 1
        re2 += 1
    cnn_se_lev2_2000_acc_aver = float(sum(acc_2) / len(acc_2))
    cnn_se_lev2_2000_time_aver = float(sum(ti_2) / len(ti_2))
    acc_f_2.append( cnn_se_lev2_2000_acc_aver)
    ti_f_2.append(cnn_se_lev2_2000_time_aver)
    print('level2_cnn_se_2000_det1=', acc_2)
    print('level2_cnn_se_2000_time=', ti_2)
    print('level2_cnn_se_acc_aver_2000=',  cnn_se_lev2_2000_acc_aver)
    print('level2_cnn_se_time_aver_2000=', cnn_se_lev2_2000_time_aver)
    m = 0

print('\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n')



#############################故障等级三###############################################################################################
###故障等级3中，读取原始数据集

cnn_se_lev3_lab0 = pd.read_csv(r'D:\study\maching\brother\tengge\新想法1\数据集孙\level3\lev3_lab0.csv')
cnn_se_lev3_lab1 = pd.read_csv(r'D:\study\maching\brother\tengge\新想法1\数据集孙\level3\lev3_lab1.csv')
cnn_se_lev3_lab2 = pd.read_csv(r'D:\study\maching\brother\tengge\新想法1\数据集孙\level3\lev3_lab2.csv')
cnn_se_lev3_lab3 = pd.read_csv(r'D:\study\maching\brother\tengge\新想法1\数据集孙\level3\lev3_lab3.csv')
cnn_se_lev3_lab4 = pd.read_csv(r'D:\study\maching\brother\tengge\新想法1\数据集孙\level3\lev3_lab4.csv')
cnn_se_lev3_lab5 = pd.read_csv(r'D:\study\maching\brother\tengge\新想法1\数据集孙\level3\lev3_lab5.csv')
cnn_se_lev3_lab6 = pd.read_csv(r'D:\study\maching\brother\tengge\新想法1\数据集孙\level3\lev3_lab6.csv')
cnn_se_lev3_lab7 = pd.read_csv(r'D:\study\maching\brother\tengge\新想法1\数据集孙\level3\lev3_lab7.csv')

# 丢弃包含NAN的数据
cnn_se_lev3_lab0 = cnn_se_lev3_lab0.dropna()
cnn_se_lev3_lab1 = cnn_se_lev3_lab1.dropna()
cnn_se_lev3_lab2 = cnn_se_lev3_lab2.dropna()
cnn_se_lev3_lab3 = cnn_se_lev3_lab3.dropna()
cnn_se_lev3_lab4 = cnn_se_lev3_lab4.dropna()
cnn_se_lev3_lab5 = cnn_se_lev3_lab5.dropna()
cnn_se_lev3_lab6 = cnn_se_lev3_lab6.dropna()
cnn_se_lev3_lab7 = cnn_se_lev3_lab7.dropna()

m = 0
for k in range(0, o):
    for j in range(0, p):
        cnn_se_lev3_lab0_1 = cnn_se_lev3_lab0.iloc[m:m + n, :]
        cnn_se_lev3_lab1_1 = cnn_se_lev3_lab1.iloc[m:m + n, :]
        cnn_se_lev3_lab2_1 = cnn_se_lev3_lab2.iloc[m:m + n, :]
        cnn_se_lev3_lab3_1 = cnn_se_lev3_lab3.iloc[m:m + n, :]
        cnn_se_lev3_lab4_1 = cnn_se_lev3_lab4.iloc[m:m + n, :]
        cnn_se_lev3_lab5_1 = cnn_se_lev3_lab5.iloc[m:m + n, :]
        cnn_se_lev3_lab6_1 = cnn_se_lev3_lab6.iloc[m:m + n, :]
        cnn_se_lev3_lab7_1 = cnn_se_lev3_lab7.iloc[m:m + n, :]

        # 将n0个数据划分为训练集、验证集、测试集(8:1:1)
        cnn_se_lev3_lab0_train = cnn_se_lev3_lab0_1.iloc[0:int(a * n), :]
        cnn_se_lev3_lab0_val = cnn_se_lev3_lab0_1.iloc[int(a * n):int(b * n), :]
        cnn_se_lev3_lab0_test = cnn_se_lev3_lab0_1.iloc[int(b * n):n, :]

        cnn_se_lev3_lab1_train = cnn_se_lev3_lab1_1.iloc[0:int(a * n), :]
        cnn_se_lev3_lab1_val = cnn_se_lev3_lab1_1.iloc[int(a * n):int(b * n), :]
        cnn_se_lev3_lab1_test = cnn_se_lev3_lab1_1.iloc[int(b * n):n, :]

        cnn_se_lev3_lab2_train = cnn_se_lev3_lab2_1.iloc[0:int(a * n), :]
        cnn_se_lev3_lab2_val = cnn_se_lev3_lab2_1.iloc[int(a * n):int(b * n), :]
        cnn_se_lev3_lab2_test = cnn_se_lev3_lab2_1.iloc[int(b * n):n, :]

        cnn_se_lev3_lab3_train = cnn_se_lev3_lab3_1.iloc[0:int(a * n), :]
        cnn_se_lev3_lab3_val = cnn_se_lev3_lab3_1.iloc[int(a * n):int(b * n), :]
        cnn_se_lev3_lab3_test = cnn_se_lev3_lab3_1.iloc[int(b * n):n, :]

        cnn_se_lev3_lab4_train = cnn_se_lev3_lab4_1.iloc[0:int(a * n), :]
        cnn_se_lev3_lab4_val = cnn_se_lev3_lab4_1.iloc[int(a * n):int(b * n), :]
        cnn_se_lev3_lab4_test = cnn_se_lev3_lab4_1.iloc[int(b * n):n, :]

        cnn_se_lev3_lab5_train = cnn_se_lev3_lab5_1.iloc[0:int(a * n), :]
        cnn_se_lev3_lab5_val = cnn_se_lev3_lab5_1.iloc[int(a * n):int(b * n), :]
        cnn_se_lev3_lab5_test = cnn_se_lev3_lab5_1.iloc[int(b * n):n, :]

        cnn_se_lev3_lab6_train = cnn_se_lev3_lab6_1.iloc[0:int(a * n), :]
        cnn_se_lev3_lab6_val = cnn_se_lev3_lab6_1.iloc[int(a * n):int(b * n), :]
        cnn_se_lev3_lab6_test = cnn_se_lev3_lab6_1.iloc[int(b * n):n, :]

        cnn_se_lev3_lab7_train = cnn_se_lev3_lab7_1.iloc[0:int(a * n), :]
        cnn_se_lev3_lab7_val = cnn_se_lev3_lab7_1.iloc[int(a * n):int(b * n), :]
        cnn_se_lev3_lab7_test = cnn_se_lev3_lab7_1.iloc[int(b * n):n, :]

        # 合并训练集测试集
        cnn_se_lev3_train = pd.concat([cnn_se_lev3_lab0_train, cnn_se_lev3_lab1_train, cnn_se_lev3_lab2_train, cnn_se_lev3_lab3_train,
                                cnn_se_lev3_lab4_train, cnn_se_lev3_lab5_train, cnn_se_lev3_lab6_train, cnn_se_lev3_lab7_train], axis=0)

        cnn_se_lev3_val = pd.concat([cnn_se_lev3_lab0_val, cnn_se_lev3_lab1_val, cnn_se_lev3_lab2_val, cnn_se_lev3_lab3_val, cnn_se_lev3_lab4_val,
                              cnn_se_lev3_lab5_val, cnn_se_lev3_lab6_val, cnn_se_lev3_lab7_val], axis=0)

        cnn_se_lev3_test = pd.concat([cnn_se_lev3_lab0_test, cnn_se_lev3_lab1_test, cnn_se_lev3_lab2_test, cnn_se_lev3_lab3_test,
                               cnn_se_lev3_lab4_test, cnn_se_lev3_lab5_test, cnn_se_lev3_lab6_test, cnn_se_lev3_lab7_test], axis=0)
        print('cnn_se_lev3_train.shape', cnn_se_lev3_train.shape)
        print('cnn_se_lev3_val.shape', cnn_se_lev3_val.shape)
        print('cnn_se_lev3_test.shape', cnn_se_lev3_test.shape)

        train_X_det3, train_Y_det3, val_X_det3, val_Y_det3, test_X_det3, test_Y_det3 = load_data_det_8(cnn_se_lev3_train,cnn_se_lev3_val,cnn_se_lev3_test)
        print('train_X_det3.shape', train_X_det3.shape)
        print('train_Y_det3.shape:', train_Y_det3.shape)
        print('val_X_det3.shape', val_X_det3.shape)
        print('val_Y_det3.shape:', val_Y_det3.shape)
        print('test_X_det3.shape', test_X_det3.shape)
        print('test_Y_det3.shape:', test_Y_det3.shape)

        input_deep = tf.keras.layers.Input(shape=(8, 1))
        hidden1 = tf.keras.layers.Conv1D(filters=32, kernel_size=3, padding='same', activation='relu')(input_deep)
        # hidden2 = tf.keras.layers.Conv1D(filters = 32,kernel_size = 3,padding = 'same',activation = 'relu')(hidden1)
        x = tf.keras.layers.GlobalAvgPool1D()(hidden1)
        print(x.shape)
        x = tf.keras.layers.Dense(int(x.shape[-1]) // 8, activation='relu')(x)
        print(x.shape)
        x = tf.keras.layers.Dense(int(hidden1.shape[-1]), activation='sigmoid')(x)
        print(x.shape)
        x = tf.keras.layers.Multiply()([hidden1, x])

        hidden4 = tf.keras.layers.Conv1D(filters=64, kernel_size=3, padding='same', activation='relu')(x)
        # hidden5 = tf.keras.layers.Conv1D(filters = 64,kernel_size = 3,padding = 'same',activation = 'relu')(hidden4)
        xx = tf.keras.layers.GlobalAvgPool1D()(hidden4)
        print(xx.shape)
        xx = tf.keras.layers.Dense(int(xx.shape[-1]) // 8, activation='relu')(xx)
        xx = tf.keras.layers.Dense(int(hidden4.shape[-1]), activation='sigmoid')(xx)
        xx = tf.keras.layers.Multiply()([hidden4, xx])

        hidden7 = tf.keras.layers.Conv1D(filters=128, kernel_size=3, padding='same', activation='relu')(xx)
        hidden10 = tf.keras.layers.Flatten()(hidden7)
        hidden111 = tf.keras.layers.Dense(128, activation='relu')(hidden10)
        dp = tf.keras.layers.Dropout(0.2)(hidden111)
        ###全连接层
        output = tf.keras.layers.Dense(8, activation='softmax')(dp)

        cnn_se_lev3_2000 = tf.keras.models.Model(inputs=input_deep,outputs=[output])
        cnn_se_lev3_2000_det3 = cnn_se_lev3_2000
        cnn_se_lev3_2000_det3.compile(loss="sparse_categorical_crossentropy", optimizer='Nadam', metrics=['accuracy'])

        # In[15]:
        # sum = 0
        # for i in range(0,3):
        time1 = time.time()

        ###一般来说机器学习的训练次数会设置到很大，如果模型的表现没有进一步提升，那么训练可以停止了，继续训练很可能会导致过拟合keras.callbacks.EarlyStopping就是用来提前结束训练的。
        callback1 = [tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=patienc, restore_best_weights=True)]
        history_cnn_se_lev3_2000 =cnn_se_lev3_2000_det3.fit(train_X_det3, train_Y_det3, validation_data=(val_X_det3, val_Y_det3),
                                            callbacks=callback1, batch_size=50, epochs=2000, verbose=2)
        cnn_se_lev3_2000_tim = time.time() - time1

        det_cnn_se_lev3_2000 = cnn_se_lev3_2000_det3.predict(test_X_det3)
        a_cnn_se_lev3_2000 = np.argmax(det_cnn_se_lev3_2000, axis=1)
        a_cnn_se_lev3_2000 =  a_cnn_se_lev3_2000.reshape(-1, 1)
        # 输出总的的故障检测与分类精度
        a_cnn_se_lev3_2000_AC = accuracy_score(test_Y_det3, a_cnn_se_lev3_2000)
        acc_3.append(a_cnn_se_lev3_2000_AC)
        ti_3.append(cnn_se_lev3_2000_tim)

        level3_cfm = confusion_matrix(test_Y_det3, a_cnn_se_lev3_2000)
        level3_conf = pd.DataFrame(level3_cfm).transpose()
        level3_conf.to_csv("level3_confusion_matrix" + str(ma3) + ".csv", index=True)
        print(level3_cfm)
        level3_report = classification_report(test_Y_det3, a_cnn_se_lev3_2000, output_dict=True)
        level3_df = pd.DataFrame(level3_report).transpose()
        level3_df.to_csv("level3_result" + str(re3) + ".csv", index=True)



        print('a_cnn_se_lev3_2000_AC=', a_cnn_se_lev3_2000_AC)
        print('cnn_se_lev3_2000_tim=', cnn_se_lev3_2000_tim)
        print('\n\n\n\n\n\n\n\n')
        print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
        m += 1000
        ma3 += 1
        re3 += 1
    cnn_se_lev3_2000_acc_aver = float(sum(acc_3) / len(acc_3))
    cnn_se_lev3_2000_time_aver = float(sum(ti_3) / len(ti_3))
    acc_f_3.append( cnn_se_lev2_2000_acc_aver)
    ti_f_3.append(cnn_se_lev2_2000_time_aver)
    print('level3_cnn_se_2000_det1=', acc_3)
    print('level3_cnn_se_2000_time=', ti_3)
    print('level3_cnn_se_acc_aver_2000=',  cnn_se_lev3_2000_acc_aver)
    print('level3_cnn_se_time_aver_2000=', cnn_se_lev3_2000_time_aver)
    m = 0

print('\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n')






#############################故障等级四###############################################################################################
###故障等级4中，读取原始数据集

cnn_se_lev4_lab0 = pd.read_csv(r'D:\study\maching\brother\tengge\新想法1\数据集孙\level4\lev4_lab0.csv')
cnn_se_lev4_lab1 = pd.read_csv(r'D:\study\maching\brother\tengge\新想法1\数据集孙\level4\lev4_lab1.csv')
cnn_se_lev4_lab2 = pd.read_csv(r'D:\study\maching\brother\tengge\新想法1\数据集孙\level4\lev4_lab2.csv')
cnn_se_lev4_lab3 = pd.read_csv(r'D:\study\maching\brother\tengge\新想法1\数据集孙\level4\lev4_lab3.csv')
cnn_se_lev4_lab4 = pd.read_csv(r'D:\study\maching\brother\tengge\新想法1\数据集孙\level4\lev4_lab4.csv')
cnn_se_lev4_lab5 = pd.read_csv(r'D:\study\maching\brother\tengge\新想法1\数据集孙\level4\lev4_lab5.csv')
cnn_se_lev4_lab6 = pd.read_csv(r'D:\study\maching\brother\tengge\新想法1\数据集孙\level4\lev4_lab6.csv')
cnn_se_lev4_lab7 = pd.read_csv(r'D:\study\maching\brother\tengge\新想法1\数据集孙\level4\lev4_lab7.csv')

# 丢弃包含NAN的数据
cnn_se_lev4_lab0 = cnn_se_lev4_lab0.dropna()
cnn_se_lev4_lab1 = cnn_se_lev4_lab1.dropna()
cnn_se_lev4_lab2 = cnn_se_lev4_lab2.dropna()
cnn_se_lev4_lab3 = cnn_se_lev4_lab3.dropna()
cnn_se_lev4_lab4 = cnn_se_lev4_lab4.dropna()
cnn_se_lev4_lab5 = cnn_se_lev4_lab5.dropna()
cnn_se_lev4_lab6 = cnn_se_lev4_lab6.dropna()
cnn_se_lev4_lab7 = cnn_se_lev4_lab7.dropna()

m = 0
for k in range(0, o):
    for j in range(0, p):
        cnn_se_lev4_lab0_1 = cnn_se_lev4_lab0.iloc[m:m + n, :]
        cnn_se_lev4_lab1_1 = cnn_se_lev4_lab1.iloc[m:m + n, :]
        cnn_se_lev4_lab2_1 = cnn_se_lev4_lab2.iloc[m:m + n, :]
        cnn_se_lev4_lab3_1 = cnn_se_lev4_lab3.iloc[m:m + n, :]
        cnn_se_lev4_lab4_1 = cnn_se_lev4_lab4.iloc[m:m + n, :]
        cnn_se_lev4_lab5_1 = cnn_se_lev4_lab5.iloc[m:m + n, :]
        cnn_se_lev4_lab6_1 = cnn_se_lev4_lab6.iloc[m:m + n, :]
        cnn_se_lev4_lab7_1 = cnn_se_lev4_lab7.iloc[m:m + n, :]

        # 将n0个数据划分为训练集、验证集、测试集(8:1:1)
        cnn_se_lev4_lab0_train = cnn_se_lev4_lab0_1.iloc[0:int(a * n), :]
        cnn_se_lev4_lab0_val = cnn_se_lev4_lab0_1.iloc[int(a * n):int(b * n), :]
        cnn_se_lev4_lab0_test = cnn_se_lev4_lab0_1.iloc[int(b * n):n, :]

        cnn_se_lev4_lab1_train = cnn_se_lev4_lab1_1.iloc[0:int(a * n), :]
        cnn_se_lev4_lab1_val = cnn_se_lev4_lab1_1.iloc[int(a * n):int(b * n), :]
        cnn_se_lev4_lab1_test = cnn_se_lev4_lab1_1.iloc[int(b * n):n, :]

        cnn_se_lev4_lab2_train = cnn_se_lev4_lab2_1.iloc[0:int(a * n), :]
        cnn_se_lev4_lab2_val = cnn_se_lev4_lab2_1.iloc[int(a * n):int(b * n), :]
        cnn_se_lev4_lab2_test = cnn_se_lev4_lab2_1.iloc[int(b * n):n, :]

        cnn_se_lev4_lab3_train = cnn_se_lev4_lab3_1.iloc[0:int(a * n), :]
        cnn_se_lev4_lab3_val = cnn_se_lev4_lab3_1.iloc[int(a * n):int(b * n), :]
        cnn_se_lev4_lab3_test = cnn_se_lev4_lab3_1.iloc[int(b * n):n, :]

        cnn_se_lev4_lab4_train = cnn_se_lev4_lab4_1.iloc[0:int(a * n), :]
        cnn_se_lev4_lab4_val = cnn_se_lev4_lab4_1.iloc[int(a * n):int(b * n), :]
        cnn_se_lev4_lab4_test = cnn_se_lev4_lab4_1.iloc[int(b * n):n, :]

        cnn_se_lev4_lab5_train = cnn_se_lev4_lab5_1.iloc[0:int(a * n), :]
        cnn_se_lev4_lab5_val = cnn_se_lev4_lab5_1.iloc[int(a * n):int(b * n), :]
        cnn_se_lev4_lab5_test = cnn_se_lev4_lab5_1.iloc[int(b * n):n, :]

        cnn_se_lev4_lab6_train = cnn_se_lev4_lab6_1.iloc[0:int(a * n), :]
        cnn_se_lev4_lab6_val = cnn_se_lev4_lab6_1.iloc[int(a * n):int(b * n), :]
        cnn_se_lev4_lab6_test = cnn_se_lev4_lab6_1.iloc[int(b * n):n, :]

        cnn_se_lev4_lab7_train = cnn_se_lev4_lab7_1.iloc[0:int(a * n), :]
        cnn_se_lev4_lab7_val = cnn_se_lev4_lab7_1.iloc[int(a * n):int(b * n), :]
        cnn_se_lev4_lab7_test = cnn_se_lev4_lab7_1.iloc[int(b * n):n, :]

        # 合并训练集测试集
        cnn_se_lev4_train = pd.concat([cnn_se_lev4_lab0_train, cnn_se_lev4_lab1_train, cnn_se_lev4_lab2_train, cnn_se_lev4_lab3_train,
                                cnn_se_lev4_lab4_train, cnn_se_lev4_lab5_train, cnn_se_lev4_lab6_train, cnn_se_lev4_lab7_train], axis=0)

        cnn_se_lev4_val = pd.concat([cnn_se_lev4_lab0_val, cnn_se_lev4_lab1_val, cnn_se_lev4_lab2_val, cnn_se_lev4_lab3_val, cnn_se_lev4_lab4_val,
                              cnn_se_lev4_lab5_val, cnn_se_lev4_lab6_val, cnn_se_lev4_lab7_val], axis=0)

        cnn_se_lev4_test = pd.concat([cnn_se_lev4_lab0_test, cnn_se_lev4_lab1_test, cnn_se_lev4_lab2_test, cnn_se_lev4_lab3_test,
                               cnn_se_lev4_lab4_test, cnn_se_lev4_lab5_test, cnn_se_lev4_lab6_test, cnn_se_lev4_lab7_test], axis=0)
        print('cnn_se_lev4_train.shape', cnn_se_lev4_train.shape)
        print('cnn_se_lev4_val.shape', cnn_se_lev4_val.shape)
        print('cnn_se_lev4_test.shape', cnn_se_lev4_test.shape)

        train_X_det4, train_Y_det4, val_X_det4, val_Y_det4, test_X_det4, test_Y_det4 = load_data_det_8(cnn_se_lev4_train,cnn_se_lev4_val,cnn_se_lev4_test)
        print('train_X_det4.shape', train_X_det4.shape)
        print('train_Y_det4.shape:', train_Y_det4.shape)
        print('val_X_det4.shape', val_X_det4.shape)
        print('val_Y_det4.shape:', val_Y_det4.shape)
        print('test_X_det4.shape', test_X_det4.shape)
        print('test_Y_det4.shape:', test_Y_det4.shape)

        input_deep = tf.keras.layers.Input(shape=(8, 1))
        hidden1 = tf.keras.layers.Conv1D(filters=32, kernel_size=3, padding='same', activation='relu')(input_deep)
        # hidden2 = tf.keras.layers.Conv1D(filters = 32,kernel_size = 3,padding = 'same',activation = 'relu')(hidden1)
        x = tf.keras.layers.GlobalAvgPool1D()(hidden1)
        print(x.shape)
        x = tf.keras.layers.Dense(int(x.shape[-1]) // 8, activation='relu')(x)
        print(x.shape)
        x = tf.keras.layers.Dense(int(hidden1.shape[-1]), activation='sigmoid')(x)
        print(x.shape)
        x = tf.keras.layers.Multiply()([hidden1, x])

        hidden4 = tf.keras.layers.Conv1D(filters=64, kernel_size=3, padding='same', activation='relu')(x)
        # hidden5 = tf.keras.layers.Conv1D(filters = 64,kernel_size = 3,padding = 'same',activation = 'relu')(hidden4)
        xx = tf.keras.layers.GlobalAvgPool1D()(hidden4)
        print(xx.shape)
        xx = tf.keras.layers.Dense(int(xx.shape[-1]) // 8, activation='relu')(xx)
        xx = tf.keras.layers.Dense(int(hidden4.shape[-1]), activation='sigmoid')(xx)
        xx = tf.keras.layers.Multiply()([hidden4, xx])

        hidden7 = tf.keras.layers.Conv1D(filters=128, kernel_size=3, padding='same', activation='relu')(xx)
        # hidden8 = tf.keras.layers.Conv1D(filters = 128,kernel_size = 3,padding = 'same',activation = 'relu')(hidden7)
        # hidden9 = tf.keras.layers.MaxPool1D(pool_size = 2)(hidden7)
        hidden10 = tf.keras.layers.Flatten()(hidden7)
        hidden111 = tf.keras.layers.Dense(128, activation='relu')(hidden10)
        dp = tf.keras.layers.Dropout(0.2)(hidden111)
        ###全连接层
        output = tf.keras.layers.Dense(8, activation='softmax')(dp)

        cnn_se_lev4_2000 = tf.keras.models.Model(inputs=input_deep,outputs=[output])
        cnn_se_lev4_2000_det4 = cnn_se_lev4_2000
        cnn_se_lev4_2000_det4.compile(loss="sparse_categorical_crossentropy", optimizer='Nadam', metrics=['accuracy'])

        # In[15]:
        # sum = 0
        # for i in range(0,3):
        time1 = time.time()

        ###一般来说机器学习的训练次数会设置到很大，如果模型的表现没有进一步提升，那么训练可以停止了，继续训练很可能会导致过拟合keras.callbacks.EarlyStopping就是用来提前结束训练的。
        callback1 = [tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=patienc, restore_best_weights=True)]
        history_cnn_se_lev4_2000 =cnn_se_lev4_2000_det4.fit(train_X_det4, train_Y_det4, validation_data=(val_X_det4, val_Y_det4),
                                            callbacks=callback1, batch_size=50, epochs=2000, verbose=2)
        cnn_se_lev4_2000_tim = time.time() - time1

        det_cnn_se_lev4_2000 = cnn_se_lev4_2000_det4.predict(test_X_det4)
        a_cnn_se_lev4_2000 = np.argmax(det_cnn_se_lev4_2000, axis=1)
        a_cnn_se_lev4_2000 =  a_cnn_se_lev4_2000.reshape(-1, 1)
        # 输出总的的故障检测与分类精度
        a_cnn_se_lev4_2000_AC = accuracy_score(test_Y_det4, a_cnn_se_lev4_2000)
        acc_4.append(a_cnn_se_lev4_2000_AC)
        ti_4.append(cnn_se_lev4_2000_tim)

        level4_cfm = confusion_matrix(test_Y_det4, a_cnn_se_lev4_2000)
        level4_conf = pd.DataFrame(level4_cfm).transpose()
        level4_conf.to_csv("level4_confusion_matrix" + str(ma4) + ".csv", index=True)
        print(level4_cfm)
        level4_report = classification_report(test_Y_det4, a_cnn_se_lev4_2000, output_dict=True)
        level4_df = pd.DataFrame(level4_report).transpose()
        level4_df.to_csv("level4_result" + str(re4) + ".csv", index=True)




        print('a_cnn_se_lev4_2000_AC=', a_cnn_se_lev4_2000_AC)
        print('cnn_se_lev4_2000_tim=', cnn_se_lev4_2000_tim)
        print('\n\n\n\n\n\n\n\n')
        print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
        m += 1000
        ma4 += 1
        re4 += 1
    cnn_se_lev4_2000_acc_aver = float(sum(acc_4) / len(acc_4))
    cnn_se_lev4_2000_time_aver = float(sum(ti_4) / len(ti_4))
    acc_f_4.append(cnn_se_lev4_2000_acc_aver)
    ti_f_4.append(cnn_se_lev4_2000_time_aver)
    print('level4_cnn_se_2000_det1=', acc_4)
    print('level4_cnn_se_2000_time=', ti_4)
    print('level4_cnn_se_acc_aver_2000=',  cnn_se_lev4_2000_acc_aver)
    print('level4_cnn_se_time_aver_2000=', cnn_se_lev4_2000_time_aver)
    m = 0

print('\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n')

print('level1_cnn_se_2000_det1=', acc_1)
print('level1_cnn_se_2000_time=', ti_1)
print('level1_cnn_se_acc_aver_2000=', cnn_se_lev1_2000_acc_aver)
print('level1_cnn_se_time_aver_2000=', cnn_se_lev1_2000_time_aver)
print('\n\n\n\n\n\n')

print('level2_cnn_se_2000_det1=', acc_2)
print('level2_cnn_se_2000_time=', ti_2)
print('level2_cnn_se_acc_aver_2000=',  cnn_se_lev2_2000_acc_aver)
print('level2_cnn_se_time_aver_2000=', cnn_se_lev2_2000_time_aver)
print('\n\n\n\n\n\n')
print('level3_cnn_se_2000_det1=', acc_3)
print('level3_cnn_se_2000_time=', ti_3)
print('level3_cnn_se_acc_aver_2000=', cnn_se_lev3_2000_acc_aver)
print('level3_cnn_se_time_aver_2000=', cnn_se_lev3_2000_time_aver)
print('\n\n\n\n\n\n')

print('level4_cnn_se_2000_det1=', acc_4)
print('level4_cnn_se_2000_time=', ti_4)
print('level4_cnn_se_acc_aver_2000=',  cnn_se_lev4_2000_acc_aver)
print('level4_cnn_se_time_aver_2000=', cnn_se_lev4_2000_time_aver)




result_excel = pd.DataFrame()
result_excel["level1_acc"] =acc_1
result_excel["level2_acc"] =acc_2
result_excel["level3_acc"] = acc_3
result_excel["level4_acc"] = acc_4


result_excel["level1_time"] = ti_1
result_excel["level2_time"] = ti_2
result_excel["level3_time"] = ti_3
result_excel["level4_time"] = ti_4



writer = pd.ExcelWriter('CCNN_SE__30EPOCH_7_3.xlsx')

result_excel.to_excel(writer,"流水")


writer.save()