import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time

from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.feature_selection import SelectFromModel, chi2, RFE, VarianceThreshold, SelectKBest
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import cross_val_score, KFold
from sklearn.linear_model import Lasso
from sklearn.decomposition import PCA
from sklearn.decomposition import FastICA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

def diagramOfMethods(method, model1, model2, model3, model4, model5, model6):
    fig = plt.figure(1, figsize=(10, 6))
    plt.clf()
    plt.plot(model1[0], model1[1], getMarker(1))
    plt.plot(model2[0], model2[1], getMarker(2))
    plt.plot(model3[0], model3[1], getMarker(3))
    plt.plot(model4[0], model4[1], getMarker(4))
    plt.plot(model5[0], model5[1], getMarker(5))
    plt.plot(model6[0], model6[1], getMarker(6))
    plt.axis('tight')
    plt.xlabel('Amount of features, %')
    plt.ylabel('Accuracy')
    plt.grid()
    plt.legend(['Logistic regression', 'Random forest classifier', 'Extra trees classifier', 'LinearSVC', 'Lasso-regression', 'Ridge-regression'])
    plt.title("Accuracy of  " + method)
    plt.show()

def preparationData(dataset, numDS):
    data = pd.read_csv(dataset)
    print("Initail set: ", data.shape)

    if numDS == 0:
        data = data.loc[data['pdays'] != -1].loc[data['poutcome'] != 'unknown'].loc[data['job'] != 'unknown'].loc[
            data['education'] != 'unknown'].loc[data['contact'] != 'unknown']
        label = LabelEncoder()
        dicts = {}
        for i in data:
            if type(i) != int and type(i) != float:
                label.fit(data[i].drop_duplicates())
                dicts[i] = list(label.classes_)
                data[i] = label.transform(data[i])

        X = data.drop(['y'], axis=1)
        Y = data['y']
        data_nm = pd.DataFrame(MinMaxScaler().fit_transform(data), columns=data.columns)
        X_nm = data_nm.drop(['y'], axis=1)
        Y_nm = data_nm['y']
        X.columns = X_nm.columns = [i for i in range(1, 17)]
        print(X.shape)

    if (numDS == 1):
        data = data.loc[data['job'] != 'unknown'].loc[data['marital'] != 'unknown'].loc[data['education'] != 'unknown'].loc[
            data['default'] != 'unknown'].loc[data['housing'] != 'unknown'].loc[data['loan'] != 'unknown'].loc[data['poutcome'] != 'nonexistant']
        label = LabelEncoder()
        dicts = {}
        for i in data:
            if type(i) != int and type(i) != float:
                label.fit(data[i].drop_duplicates())
                dicts[i] = list(label.classes_)
                data[i] = label.transform(data[i])

        X = data.drop(['y'], axis=1)
        Y = data['y']
        data_nm = pd.DataFrame(MinMaxScaler().fit_transform(data), columns=data.columns)
        X_nm = data_nm.drop(['y'], axis=1)
        Y_nm = data_nm['y']
        X.columns = X_nm.columns = [i for i in range(1, 21)]
        print(X.shape)

    elif numDS == 2:
        data = data.loc[data['job'] != 'unknown'].loc[data['education'] != 'unknown'].loc[
            data['contact'] != 'unknown'].loc[data['pdays'] != -1].loc[data['poutcome'] != 'unknown']
        label = LabelEncoder()
        dicts = {}
        for i in data:
            if type(i) != int and type(i) != float:
                label.fit(data[i].drop_duplicates())
                dicts[i] = list(label.classes_)
                data[i] = label.transform(data[i])

        X = data.drop(['deposit'], axis=1)
        Y = data['deposit']
        data_nm = pd.DataFrame(MinMaxScaler().fit_transform(data), columns=data.columns)
        X_nm = data_nm.drop(['deposit'], axis=1)
        Y_nm = data_nm['deposit']
        X.columns = X_nm.columns = [i for i in range(1, 17)]
        print(X.shape)

    return X, Y, X_nm, Y_nm

def getMarker(numberDataset):
    return 'r-' if numberDataset == 1 \
        else 'g--' if numberDataset == 2 \
        else 'b-.' if numberDataset == 3 \
        else 'c--.' if numberDataset == 4 \
        else 'yx-' if numberDataset == 5 \
        else '*-'

def funVarianceThreshold(arrX, arrY):
    maxAcc = [0]*3
    print('VarianceThreshold')
    datasetNumber = 0
    fig=plt.figure(1, figsize=(10, 6))
    ax=fig.add_subplot(1, 1, 1)
    iterations = 10
    time_score = [0] * iterations
    thresholds = np.arange(0, iterations*0.01, 0.01)
    for X, y in zip(arrX, arrY):
        datasetNumber += 1
        print("\nDataset", datasetNumber)

        n_splits = 10
        kfold = KFold(n_splits=n_splits)
        est_model = LogisticRegression(solver='lbfgs', max_iter=200)

        results = cross_val_score(est_model, X, y, cv=kfold, scoring='accuracy')
        print("accuracy:", round(sum(results)/n_splits*1e+6)/1e+6, "  features:", X.shape[1])

        acc_score = [0] * iterations
        for i in range(0, iterations):
            start_time = time.time()
            selector = VarianceThreshold(threshold=thresholds[i])
            selector.fit(X, y)
            if X.empty:
                break
            finish_time = time.time()
            X_new = X[X.columns[selector.get_support(indices=True)]]

            results = cross_val_score(est_model, X_new, y, cv=kfold, scoring='accuracy')
            print("accuracy:", round(sum(results)/n_splits*1e+6)/1e+6, "  features:", X_new.shape[1], "  time:", round((finish_time - start_time)*1e+6)/1e+6, "  threshold:", thresholds[i])
            acc_score[i] += round(sum(results)/n_splits*1e+6)/1e+6
            if acc_score[i] > maxAcc[datasetNumber - 1]:
                maxAcc[datasetNumber - 1] = acc_score[i ]
            time_score[i] += round((finish_time - start_time)*1e+6)/1e+6
        print("max accuracy - ", maxAcc[datasetNumber - 1])

        marker = getMarker(datasetNumber)
        plt.plot(thresholds, acc_score, marker, linewidth=2)

    plt.legend(['Dataset 1', 'Dataset 2', 'Dataset 3'])
    plt.axis('tight')
    plt.xlabel('Coefficients of thresholds')
    plt.ylabel('Accuracy')
    ax.set_ylim([0.65, 0.9])
    plt.grid()
    plt.title("VarianceThreshold method")
    plt.show()

def funSelectKBest(arrX, arrY):
    maxAcc = [0] * 3
    print('SelectKBest')
    datasetNumber = 0
    fig = plt.figure(1, figsize=(10, 6))
    ax = fig.add_subplot(1, 1, 1)
    for X, y in zip(arrX, arrY):
        datasetNumber += 1
        print("\nDataset", datasetNumber)

        n_splits = 10
        kfold = KFold(n_splits=n_splits)
        est_model = LogisticRegression(solver='lbfgs')
        if X.shape[0] > 10000:
            est_model = LogisticRegression(solver='lbfgs', max_iter=200)

        acc_score = [0] * X.shape[1]
        for i in range(1, X.shape[1] + 1):
            start_time = time.time()
            fit = SelectKBest(chi2, k=i).fit(X, y)
            finish_time = time.time()
            X_new = X[X.columns[fit.get_support(indices=True)]]

            results = cross_val_score(est_model, X_new, y, cv=kfold, scoring='accuracy')
            acc_score[i-1] = round(sum(results)/n_splits*1e+6)/1e+6
            if acc_score[i-1] > maxAcc[datasetNumber - 1]:
                maxAcc[datasetNumber - 1] = acc_score[i - 1]
        print("max accuracy - ", maxAcc[datasetNumber - 1])
            print("accuracy:", round(sum(results)/n_splits*1e+6)/1e+6, "  features:", X_new.shape[1], "  time:", round((finish_time - start_time)*1e+6)/1e+6)

        axisX = np.zeros(X.shape[1])
        for i in range(1, X.shape[1] + 1):
            axisX[i - 1] = i*100/X.shape[1]
        marker = getMarker(datasetNumber)
        plt.plot(axisX, acc_score, marker, linewidth=2)

    plt.legend(['Dataset 1', 'Dataset 2', 'Dataset 3', 'Dataset 4'])
    plt.axis('tight')
    plt.xlabel('Amount of features, %')
    plt.ylabel('Accuracy')
    ax.set_ylim([0.65, 0.9])
    plt.grid()
    plt.title("SelectKBest method")
    plt.show()

estimateModeles = ['LogisticRegression', 'RandomForestClassifier',
                   'ExtraTreesClassifier', 'LinearSVC', 'Lasso', 'Ridge']

def mySelectFromModel(arrX, arrY, modelNum):
    maxAcc = [0] * 3
    print('SelectFromModel with model №', modelNum)
    datasetNumber=0
    fig = plt.figure(1, figsize=(10, 6))
    ax = fig.add_subplot(1, 1, 1)
    for X, y in zip(arrX, arrY):
        minSpeed=1000.0
        maxSpeed=.0
        datasetNumber += 1
        print("\nDataset", datasetNumber)

        n_splits = 10
        est_model = LogisticRegression(solver='lbfgs', max_iter=200)
        kfold = KFold(n_splits=n_splits)
        if modelNum == 0:
            model = LogisticRegression(solver='lbfgs')
        elif modelNum == 1:
            model = RandomForestClassifier()
        elif modelNum == 2:
            model = ExtraTreesClassifier() # n_estimators = 30
        elif modelNum == 3:
            model = LinearSVC(C=0.01, penalty="l1", dual=False)
        elif modelNum == 4:
            model = Lasso(alpha=0.1)
        elif modelNum == 5:
            model = Ridge(alpha=1.0)
        else:
            print("Invalid number of model")
            return

        acc_score = [0] * X.shape[1]
        for i in range(1, X.shape[1] + 1):
            start_time = time.time()
            smf = SelectFromModel(model, max_features=i, threshold=-np.inf).fit(X, y)
            finish_time = time.time()
            time_ = round((finish_time - start_time)*1e+6)/1e+6
            X_new = X[X.columns[smf.get_support(indices=True)]]

            results = cross_val_score(est_model, X_new, y, cv=kfold, scoring='accuracy')
            acc_score[i-1] = round(sum(results)/n_splits*1e+6)/1e+6
            if acc_score[i - 1] > maxAcc[datasetNumber - 1]:
                maxAcc[datasetNumber - 1] = acc_score[i - 1]
            print("accuracy:", round(sum(results)/n_splits*1e+6)/1e+6, "  features:", X_new.shape[1], "  time:", time_)
            if(minSpeed > time_):
                minSpeed = time_
            if(maxSpeed < time_):
                maxSpeed = time_
        print("max accuracy - ", maxAcc[datasetNumber - 1])
        axisX = np.zeros(X.shape[1])
        for i in range(1, X.shape[1] + 1):
            axisX[i - 1] = i * 100 / X.shape[1]
        # return axisX, acc_score

        marker = getMarker(datasetNumber)
        plt.plot(axisX, acc_score, marker, linewidth=2)
        print('For dataset ', datasetNumber, ' minSpeed = ', minSpeed, '; maxSpeed = ', maxSpeed)

    plt.legend(['Dataset 1', 'Dataset 2', 'Dataset 3'])
    plt.axis('tight')
    plt.xlabel('Amount of features, %')
    plt.ylabel('Accuracy')
    ax.set_ylim([0.7, 0.93])
    plt.grid()
    title = "SelectFromModel with model " + estimateModeles[modelNum]
    plt.title(title)
    plt.show()

def myRFE(arrX, arrY, modelNum):
    maxAcc = [0] * 3
    print('RFE with model ', estimateModeles[modelNum])
    datasetNumber=0
    fig = plt.figure(1, figsize=(10, 6))
    ax = fig.add_subplot(1, 1, 1)
    for X, y in zip(arrX, arrY):
        minSpeed=1000.0
        maxSpeed=.0
        datasetNumber += 1
        print("\nDataset", datasetNumber)

        n_splits = 10
        est_model = LogisticRegression(solver='lbfgs', max_iter=200)
        kfold = KFold(n_splits=n_splits)
        if modelNum == 0:
            model = LogisticRegression(solver='lbfgs')
        elif modelNum == 1:
            model = RandomForestClassifier() # n_estimators = 30
        elif modelNum == 2:
            model = ExtraTreesClassifier() # n_estimators = 30
        elif modelNum == 3:
            model = LinearSVC(C=0.01, penalty="l1", dual=False)
        elif modelNum == 4:
            model = Lasso(alpha=0.1)
        elif modelNum == 5:
            model = Ridge(alpha=1.0)
        else:
            print("Invalid number of model")
            return

        acc_score = [0] * X.shape[1]
        for i in range(1, X.shape[1] + 1):
            start_time = time.time()
            fit = RFE(model, i).fit(X, y)
            finish_time = time.time()
            time_ = round((finish_time - start_time)*1e+6)/1e+6
            X_new = X[X.columns[fit.get_support(indices=True)]]

            results = cross_val_score(est_model, X_new, y, cv=kfold, scoring='accuracy')
            acc_score[i - 1] = round(sum(results)/n_splits*1e+6)/1e+6
            if acc_score[i - 1] > maxAcc[datasetNumber - 1]:
                maxAcc[datasetNumber - 1] = acc_score[i - 1]
        print("max accuracy - ", maxAcc[datasetNumber - 1])
            print("accuracy:", round(sum(results)/n_splits*1e+6)/1e+6, "  features:", X_new.shape[1], "  time:", time_)
            if(minSpeed > time_):
                minSpeed = time_
            if(maxSpeed < time_):
                maxSpeed = time_

        axisX = np.zeros(X.shape[1])
        for i in range(1, X.shape[1] + 1):
            axisX[i - 1] = i * 100 / X.shape[1]
        return axisX, acc_score

        marker = getMarker(datasetNumber)
        plt.plot(axisX, acc_score, marker, linewidth=2)
        print('For dataset ', datasetNumber, ' minSpeed = ', minSpeed, '; maxSpeed = ', maxSpeed)

    plt.legend(['Dataset 1', 'Dataset 2', 'Dataset 3', 'Dataset 4'])
    plt.axis('tight')
    plt.xlabel('Amount of features, %')
    plt.ylabel('Accuracy')
    ax.set_ylim([0.75, 0.9])
    plt.grid()
    title = "RFE with model " + estimateModeles[modelNum]
    plt.title(title)
    plt.show()

def funPCA(arrX, arrY):
    maxAcc = [0] * 3
    print('PCA')
    datasetNumber = 0
    fig = plt.figure(1, figsize=(10, 6))
    ax = fig.add_subplot(1, 1, 1)
    for X, y in zip(arrX, arrY):
        datasetNumber += 1
        print("\nDataset", datasetNumber)

        n_splits = 10
        est_model = LogisticRegression(solver='lbfgs')
        kfold = KFold(n_splits=n_splits)

        acc_score = [0] * X.shape[1]
        for i in range(1, X.shape[1] + 1):
            start_time = time.time()
            fit = PCA(n_components=i).fit(X)
            finish_time = time.time()
            X_new = pd.DataFrame(fit.transform(X))

            results = cross_val_score(est_model, X_new, y, cv=kfold, scoring='accuracy')
            acc_score[i - 1] = round(sum(results)/n_splits*1e+6)/1e+6
            if acc_score[i - 1] > maxAcc[datasetNumber - 1]:
                maxAcc[datasetNumber - 1] = acc_score[i - 1]
        print("max accuracy - ", maxAcc[datasetNumber - 1])
            print("accuracy:", round(sum(results)/n_splits*1e+6)/1e+6, "  features:", X_new.shape[1], "  time:", round((finish_time - start_time)*1e+6)/1e+6)

        axisX = np.zeros(X.shape[1])
        for i in range(1, X.shape[1] + 1):
            axisX[i - 1] = i * 100 / X.shape[1]
        marker = getMarker(datasetNumber)
        plt.plot(axisX, acc_score, marker, linewidth=2)

    plt.legend(['Dataset 1', 'Dataset 2', 'Dataset 3'])
    plt.axis('tight')
    plt.xlabel('Amount of features, %')
    plt.ylabel('Accuracy')
    ax.set_ylim([0.6, 0.9])
    plt.grid()
    plt.title("PCA method")
    plt.show()

def funICA(arrX, arrY):  #Independent Component Analysis
    print('ICA')
    datasetNumber = 0
    for X, y in zip(arrX, arrY):
        datasetNumber += 1
        print("\nDataset", datasetNumber)

        n_splits = 10
        est_model = LogisticRegression(solver='lbfgs')
        kfold = KFold(n_splits=n_splits)

        start_time = time.time()
        fit = FastICA(n_components=1).fit(X)
        finish_time = time.time()
        X_new = pd.DataFrame(fit.transform(X))

        results = cross_val_score(est_model, X_new, y, cv=kfold, scoring='accuracy')
        acc_score = round(sum(results)/n_splits*1e+6)/1e+6
        print("accuracy:", acc_score, "  time:", round((finish_time - start_time)*1e+6)/1e+6)

def funLDA(arrX, arrY):  #ILinear Discriminant Analysis
    print('LDA')
    datasetNumber = 0
    for X, y in zip(arrX, arrY):
        datasetNumber += 1
        print("\nDataset", datasetNumber)

        n_splits = 10
        est_model = LogisticRegression(solver='lbfgs')
        kfold = KFold(n_splits=n_splits)

        start_time = time.time()
        fit = LinearDiscriminantAnalysis(n_components=1).fit(X, y)
        finish_time = time.time()
        X_new = pd.DataFrame(fit.transform(X))

        results = cross_val_score(est_model, X_new, y, cv=kfold, scoring='accuracy')
        acc_score = round(sum(results)/n_splits*1e+6)/1e+6
        print("accuracy:", acc_score, "  time:", round((finish_time - start_time)*1e+6)/1e+6)

if __name__ == '__main__':
    x1, y1, x_nm1, y_nm1 = preparationData('data/data1.csv', 0)
    x2, y2, x_nm2, y_nm2 = preparationData('data/data2.csv', 1)
    x3, y3, x_nm3, y_nm3 = preparationData('data/data3.csv', 2)

    # ----------------------- FEATURE SELECTION -----------------------
    funVarianceThreshold([x_nm1, x_nm2, x_nm3], [y_nm1, y_nm2, y_nm3])
    funSelectKBest([x_nm1, x_nm2, x_nm3], [y_nm1, y_nm2, y_nm3])
    acc_score_SFM_LR = mySelectFromModel([x_nm1, x_nm2, x_nm3], [y_nm1, y_nm2, y_nm3], 0) # Logical regression
    acc_score_RFE_LR = myRFE([x_nm1, x_nm2, x_nm3], [y_nm1, y_nm2, y_nm3], 0)
    acc_score_SFM_RFC = mySelectFromModel([x_nm1, x_nm2, x_nm3], [y_nm1, y_nm2, y_nm3], 1) # Random trees classification
    acc_score_RFE_RFC = myRFE([x_nm1, x_nm2, x_nm3], [y_nm1, y_nm2, y_nm3], 1)
    acc_score_SFM_ETC = mySelectFromModel([x_nm1, x_nm2, x_nm3], [y_nm1, y_nm2, y_nm3], 2) # Extra trees classification
    acc_score_RFE_ETC = myRFE([x_nm1, x_nm2, x_nm3], [y_nm1, y_nm2, y_nm3], 2)
    acc_score_SFM_LSVC = mySelectFromModel([x_nm1, x_nm2, x_nm3], [y_nm1, y_nm2, y_nm3], 3) # LinearSVC classification
    acc_score_RFE_LSVC = myRFE([x_nm1, x_nm2, x_nm3], [y_nm1, y_nm2, y_nm3], 3)
    acc_score_SFM_lasso = mySelectFromModel([x_nm1, x_nm2, x_nm3], [y_nm1, y_nm2, y_nm3], 4) # Lasso classification
    acc_score_RFE_lasso = myRFE([x_nm1, x_nm2, x_nm3], [y_nm1, y_nm2, y_nm3], 4)
    acc_score_SFM_ridge = mySelectFromModel([x_nm1, x_nm2, x_nm3], [y_nm1, y_nm2, y_nm3], 5) # Ridge classification
    acc_score_RFE_ridge = myRFE([x_nm1, x_nm2, x_nm3], [y_nm1, y_nm2, y_nm3], 5)

    # ------------ Comparison SFM ------------
    diagramOfMethods('RFE', acc_score_RFE_LR, acc_score_RFE_RFC, acc_score_RFE_ETC,
                     acc_score_RFE_LSVC, acc_score_RFE_lasso, acc_score_RFE_ridge)

    # ----------------------- FEATURE EXTRACTION -----------------------
    funPCA([x_nm1, x_nm2, x_nm3], [y_nm1, y_nm2, y_nm3])
    funICA([x_nm1, x_nm2, x_nm3], [y_nm1, y_nm2, y_nm3])
    funLDA([x_nm1, x_nm2, x_nm3], [y_nm1, y_nm2, y_nm3])