import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("kidney_disease.csv")
df.head(5)

df.shape

df.columns

df['classification'].value_counts()

df.info()

df.isnull().sum()

from sklearn.impute import SimpleImputer

mode = SimpleImputer(missing_values = np.nan, strategy = 'most_frequent')

df_imputer  = pd.DataFrame(mode.fit_transform(df))
df_imputer.columns = df.columns
df_imputer

df_imputer.isnull().sum()

set(df_imputer['age'].tolist())

for i in df_imputer.columns:
    print("***************", i, "*******************")
    print()
    print(set(df_imputer[i].tolist()))
    print()

print(df_imputer['rc'].mode())
print(df_imputer['wc'].mode())
print(df_imputer['pcv'].mode())

df_imputer['classification'] = df_imputer['classification'].apply(lambda x:'ckd' if x == 'ckd\t' else x)

df_imputer['cad'] = df_imputer['cad'].apply(lambda x:'no' if x == '\tno' else x)

df_imputer['dm'] = df_imputer['dm'].apply(lambda x:'no' if x == '\tno' else x)
df_imputer['dm'] = df_imputer['dm'].apply(lambda x:'yes' if x == '\tyes' else x)
df_imputer['dm'] = df_imputer['dm'].apply(lambda x:'yes' if x == 'yes' else x)

df_imputer['rc'] = df_imputer['rc'].apply(lambda x:'5.2' if x == '\t?' else x)

df_imputer['wc'] = df_imputer['wc'].apply(lambda x:'9800' if x == '\t6200' else x)
df_imputer['wc'] = df_imputer['wc'].apply(lambda x:'9800' if x == '\t8400' else x)
df_imputer['wc'] = df_imputer['wc'].apply(lambda x:'9800' if x == '\t?' else x)

df_imputer['pcv'] = df_imputer['pcv'].apply(lambda x:'41' if x == '\t43' else x)
df_imputer['pcv'] = df_imputer['pcv'].apply(lambda x:'41' if x == '\t?' else x)


for i in df_imputer.columns:
    print("***************", i, "*******************")
    print()
    print(set(df_imputer[i].tolist()))
    print()

df_imputer['classification'].value_counts()

temp = df_imputer['classification'].value_counts()
temp_df = pd.DataFrame({'classification': temp.index, 'values':temp.values})
print(sns.barplot(x = 'classification', y = 'values', data =temp_df ))
# Implanced data

df.dtypes

df_imputer.dtypes

df.select_dtypes(exclude = ['object']).columns

for i in df.select_dtypes(exclude = ['object']).columns:
    df_imputer[i] = df_imputer[i].apply(lambda x: float(x))

df_imputer.dtypes

sns.pairplot(df_imputer)

def distplots(col):
    sns.displot(df[col])
    plt.show()
    
for i in list(df_imputer.select_dtypes(exclude = ['object']).columns)[1:]:
    distplots(i)

# outliers Detection & remove
def boxf(col):
    sns.boxplot(df[col])
    plt.show()
    

for i in list(df_imputer.select_dtypes(exclude = ['object']).columns)[1:]:
    boxf(i)

df_imputer.head()

from sklearn import preprocessing

encode = df_imputer.apply(preprocessing.LabelEncoder().fit_transform)
encode

encode.to_csv("Final_pre_processing_data.csv")

plt.figure(figsize=(20,20))
corr = encode.corr()
sns.heatmap(corr, annot = True)

df.columns

x = encode.drop(['id', 'classification'], axis = 1)
y = encode['classification']

x

y

from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from collections import Counter

print(Counter(y))

ros = RandomOverSampler()
X_ros, y_ros = ros.fit_resample(x, y)
print(Counter(y_ros))

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler((-1, 1))
x = scaler.fit_transform(X_ros)
y = y_ros

x

df.shape # 24

from sklearn.decomposition import PCA

pca = PCA(.95)
X_PCA = pca.fit_transform(x)

print(x.shape)
print(X_PCA.shape)

from sklearn.model_selection import train_test_split

x_train, x_test , y_train, y_test = train_test_split(X_PCA, y, test_size = 0.2, random_state = 7)

x_train

x_test

y_train.shape

y_test.shape

from keras import models, layers, callbacks, optimizers
from sklearn.metrics import roc_curve, auc, confusion_matrix, classification_report, accuracy_score
from sklearn.metrics import precision_recall_curve, average_precision_score, f1_score

x_train.shape[1]

def model():
    clf = models.Sequential()
    clf.add(layers.Dense(15, input_shape=(x_train.shape[1],), activation='relu'))
    clf.add(layers.Dropout(0.2))
    clf.add(layers.Dense(15, activation='relu'))
    clf.add(layers.Dropout(0.4))
    clf.add(layers.Dense(1, activation='sigmoid'))
    clf.compile(optimizer=optimizers.Adam(), loss='binary_crossentropy', metrics=['accuracy'])
    
    return clf
    

model = model()

model.summary()

history = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=20, verbose=1)

x_train.shape[1]

x_train.shape[1] * 15  # Dense X * weight

(x_train.shape[1] + 1) * 15  # X * w + bias


# function to plot the roc_curve
def plot_auc(t_y, p_y):
    fpr, tpr, thresholds = roc_curve(t_y, p_y, pos_label=1)
    fig, c_ax = plt.subplots(1, 1, figsize=(9, 9))
    c_ax.plot(fpr, tpr, label='%s (AUC:%0.2f)' % ('classification', auc(fpr, tpr)))
    c_ax.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
    c_ax.legend()
    c_ax.set_xlabel('False Positive Rate')
    c_ax.set_ylabel('True Positive Rate')

# function to plot the precision_recall_curve
def plot_precision_recall_curve_helper(t_y, p_y):
    fig, c_ax = plt.subplots(1, 1, figsize=(9, 9))
    precision, recall, thresholds = precision_recall_curve(t_y, p_y, pos_label=1)
    aps = average_precision_score(t_y, p_y)
    c_ax.plot(recall, precision, label='%s (AP Score:%0.2f)' % ('classification', aps))
    c_ax.plot(recall, precision, color='red', lw=2)
    c_ax.legend()
    c_ax.set_xlabel('Recall')
    c_ax.set_ylabel('Precision')
    

# function to plot the history
def plot_history(history):
    f = plt.figure()
    f.set_figwidth(15)
    
    f.add_subplot(1, 2, 1)
    plt.plot(history.history['val_loss'], label='val loss')
    plt.plot(history.history['loss'], label='train loss')
    plt.legend()
    plt.title("Model Loss")
    
    f.add_subplot(1, 2, 2)
    plt.plot(history.history['val_accuracy'], label='val accuracy')
    plt.plot(history.history['accuracy'], label='train accuracy')
    plt.legend()
    plt.title("Model Accuracy")

    plt.show()

hist = plot_history(history)

plot_auc(y_test, model.predict(x_test, verbose=True))

plot_precision_recall_curve_helper(y_test, model.predict(x_test, verbose=True))

def calc_f1(prec, recall):
    return 2 * (prec * recall) / (prec + recall) if recall and prec else 0

precision, recall, thresholds = precision_recall_curve(y_test, model.predict(x_test, verbose=True))
f1score = [calc_f1(precision[i], recall[i]) for i in range(len(thresholds))]
idx = np.argmax(f1score)
threshold = thresholds[idx]
print('********************************************************************************************************')
print('Precision: ' + str(precision[idx]))
print('Recall: ' + str(recall[idx]))
print('Threshold: ' + str(thresholds[idx]))
print('F1 Score: ' + str(f1score[idx]))






