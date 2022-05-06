import pandas as pd
import seaborn as sns
import dexplot as dxp
import sklearn_pandas
from sklearn.impute import SimpleImputer
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegressionCV
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from imblearn.over_sampling import SMOTE 
from imblearn.combine import SMOTETomek
from imblearn.over_sampling import RandomOverSampler
from sklearn.preprocessing import MinMaxScaler



# read in data
df = pd.read_csv("train_transaction.csv")
df.head(5)
df.shape

uid_df = pd.read_csv("train_uids_full_v3.csv")

df = pd.merge(df,uid_df.iloc[:,1:],on='TransactionID')
df.head(5)

# visualization

def without_hue(plot, feature):
    total = len(feature)
    for p in ax.patches:
        percentage = '{:.1f}%'.format(100 * p.get_height()/total)
        x = p.get_x() + p.get_width() / 2 - 0.05
        y = p.get_y() + p.get_height()
        ax.annotate(percentage, (x, y), size = 25)
    plt.show()

# frequency plot of isFraud
plt.figure(figsize = (17,15))
ax = sns.countplot('isFraud', data=df)
plt.xticks(size = 25)
plt.xlabel('isFraud', size = 25)
plt.yticks(size = 25)
plt.ylabel('Count', size = 25)
without_hue(ax, df.isFraud)


## delete NA columns
percent_missing = df.isnull().sum()*100/len(df)
missing_cols = list(percent_missing[percent_missing>30].index)
df = df.drop(columns=missing_cols)

## visualize remaining categorical variables

cat_list = ["ProductCD", "card1", "card2", "card3", "card4", "card5", "card6","addr1","addr2","P_emaildomain","M6"]
dxp.count('isFraud', data=df, split='ProductCD', normalize='isFraud')
dxp.count('isFraud', data=df, split='card4', normalize='isFraud')
dxp.count('isFraud', data=df, split='card6', normalize='isFraud')
dxp.count('isFraud', data=df, split='P_emaildomain', normalize='isFraud')
dxp.count('isFraud', data=df, split='M6', normalize='isFraud')


## fill NA
df[cat_list] = df[cat_list].astype("object")
imputer = SimpleImputer(strategy="most_frequent")
df[cat_list] = imputer.fit_transform(df[cat_list])
imputer2 = SimpleImputer(strategy="median")
df_num = df.drop(cat_list, axis = 1)
df[df_num.columns]= imputer2.fit_transform(df[df_num.columns])

## feature engineering

# need to remap, then do one-hot coding
map_list = ["P_emaildomain"]

# delete after agg, same as the cat_list
del_list = ["card1", "card2", "card3","card5","addr1","addr2", "card6", "M6", "ProductCD", "card4", "P_emaildomain"]

# directly use one-hot coding

one_hot = ["card6", "M6", "ProductCD", "card4", "P_emaildomain"]

# remap

len(df["P_emaildomain"].unique())
email_list = list(df["P_emaildomain"].unique())
email_list = email_list[1:]

[s for s in email_list if "gmail" in s] 
df.loc[df["P_emaildomain"].isin(['gmail']),"P_emaildomain"] = "gmail.com"

yahood = [s for s in email_list if "yahoo" in s] 
yahood
df.loc[df["P_emaildomain"].isin(yahood),"P_emaildomain"] = "yahoo.com"

hm = [s for s in email_list if "hotmail" in s] 
hm
df.loc[df["P_emaildomain"].isin(hm),"P_emaildomain"] = "hotmail.com"

len(df["P_emaildomain"].unique())

df.loc[~df["P_emaildomain"].isin(['gmail.com',"yahoo.com","hotmail.com","anonymous.com","aol.com"]),"P_emaildomain"] = "Others" 

len(df["P_emaildomain"].unique())

# One Hot Coding
from sklearn.preprocessing import OneHotEncoder
encoder = OneHotEncoder()
one_code_df = pd.DataFrame(encoder.fit_transform(df[one_hot]).toarray())
one_code_df.columns = encoder.get_feature_names(one_hot)
one_code_df
df3 = df.join(one_code_df)

# Aggregation
num_list= list(df3.columns[(~df3.columns.isin(cat_list)) & (~df3.columns.isin(list(one_code_df.columns)))])
del num_list[0:2] # delete transaction id, isFraud
del num_list[189] # delete uid
num_agg = df3.groupby(['uid'])[num_list].agg(['mean', 'std'])
num_agg_col = list()
for i in num_list:
    num_agg_col.append(i+"_mean")
    num_agg_col.append(i+"_std")
num_agg.columns = num_agg_col
num_agg.reset_index(inplace = True)
num_agg


cat_agg = df3.groupby(['uid'])[cat_list].nunique()
cat_agg_col = list()
for i in cat_list:
    cat_agg_col.append(i+"_nunique")
cat_agg_col
cat_agg.columns = cat_agg_col
cat_agg.reset_index(inplace=True)
cat_agg

df4 = pd.merge(pd.merge(df3,cat_agg,on='uid'), num_agg, on = 'uid')
df4 = df4.drop(cat_list, axis = 1).fillna(0)

# Normalization

from sklearn.preprocessing import MinMaxScaler

norm_col = list(df4.columns[(~df4.columns.isin(["uid", "isFraud", "TransactionID"])) & (~df4.columns.isin(list(one_code_df.columns)))])
scaler = MinMaxScaler()
normalized_arr = scaler.fit_transform(df4[norm_col])
scaled_df = pd.DataFrame(normalized_arr, columns=norm_col)
predictors_scaled_df = pd.concat([scaled_df, one_code_df], axis=1)

# train test split

X = predictors_scaled_df
y = df4['isFraud'].values
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1, stratify=y)
X_sub = X_train.iloc[:2000,:]
y_sub = y_train[:2000]

## Baseline Model

# KNN

#create new a knn model
knn = KNeighborsClassifier()
#create a dictionary of all values we want to test for n_neighbors
param_grid = {'n_neighbors': np.arange(1, 100)}
#use gridsearch to test all values for n_neighbors
knn_gscv = GridSearchCV(knn, param_grid, cv=5, verbose = 2)
#fit model to data
knn_gscv.fit(X_sub, y_sub)
#check top performing n_neighbors value
knn_gscv.best_params_
knn2 = KNeighborsClassifier(n_neighbors = knn_gscv.best_params_)
# Fit the classifier to the data
knn2.fit(X_train,y_train)
y_pred_k = knn2.predict(X_test)

confusion_matrix(y_test, y_pred_k)

print(accuracy_score(y_test, y_pred_k))

print(classification_report(y_test, y_pred_k))

print(roc_auc_score(y_test, y_pred_k))


# Random Forest

clf = RandomForestClassifier() #Initialize with whatever parameters you want to

param_grid = {
                 'n_estimators': np.arange(90,110,1),
                 'max_depth': np.arange(10,40,1)
             }
grid_clf = GridSearchCV(clf, param_grid, cv=5, verbose = 2)
grid_clf.fit(X_sub, y_sub)
grid_clf. best_params_
rf2 = RandomForestClassifier(max_depth= 11, n_estimators = 101)
# Fit the classifier to the data
rf2.fit(X_train,y_train)
y_pred_rf = rf2.predict(X_test)

confusion_matrix(y_test, y_pred_rf)

print(accuracy_score(y_test, y_pred_rf))

print(classification_report(y_test, y_pred_rf))

print(roc_auc_score(y_test, y_pred_rf))


## SMOTE+Tomek

smt = SMOTETomek(random_state = 42)
X_smt, y_smt = smt.fit_resample(X_train, y_train)

# visualize resampling process

def plot_resampling(X, y, ax, title=None):
    
    ax.scatter(X.iloc[:, 2], X.iloc[:, 5], c=y, alpha=0.8, edgecolor="k")
    if title is None:
        title = f"Resampling with {sampler.__class__.__name__}"
    ax.set_title(title, fontsize= 30)
    ax.tick_params(axis='x', labelsize=20)
    ax.tick_params(axis='y', labelsize=20)
    sns.color_palette("husl", 9)
    sns.despine(ax=ax, offset=10)

fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(15, 7))
plot_resampling(X_train, y_train, ax=axs[0], title="Before Resampling")
plot_resampling(X_smt, y_smt,ax=axs[1], title="After Resampling")
fig.tight_layout()

# PCA
pca = PCA(n_components = 0.95).fit(X_smt)
pca.n_components_
PC_values = np.arange(pca.n_components_) + 1
plt.plot(PC_values, pca.explained_variance_ratio_, 'o-', linewidth=2, color='red')
plt.title('Scree Plot')
plt.xlabel('Principal Component')
plt.ylabel('Variance Explained')
plt.show()

plt.rcParams["figure.figsize"] = (12,6)

fig, ax = plt.subplots()
xi = np.arange(1, 42, step=1)
yp = np.cumsum(pca.explained_variance_ratio_)

plt.ylim(0.0,1.1)
plt.plot(xi, yp, marker='o', linestyle='--', color='b')

plt.xlabel('Number of Components')
plt.xticks(np.arange(0, 42, step=1)) #change from 0-based array index to 1-based human-readable label
plt.ylabel('Cumulative variance (%)')
plt.title('The number of components needed to explain variance')

plt.axhline(y=0.95, color='r', linestyle='-')
plt.text(0.5, 0.85, '95% cut-off threshold', color = 'red', fontsize=16)

ax.grid(axis='x')
plt.show()

X_smt_pc = pca.transform(X_smt)
X_test_smt_pc = pca.transform(X_test)
pca_col =list()
for i in np.arange(1,42,1):
    pca_col.append("PC_{}".format(i))

X_smt_pc = pd.DataFrame(X_smt_pc, columns =pca_col )
X_test_smt_pc = pd.DataFrame(X_test_smt_pc, columns =pca_col)

import random
random.seed(10)
sub_index = random.sample(range(1, len(X_smt_pc)), 2000)
X_smt_sub = X_smt_pc.iloc[sub_index,:]
y_smt_sub = y_smt[sub_index]

# KNN
#create new a knn model
knn = KNeighborsClassifier()
#create a dictionary of all values we want to test for n_neighbors
param_grid = {'n_neighbors': np.arange(1, 100)}
#use gridsearch to test all values for n_neighbors
knn_gscv = GridSearchCV(knn, param_grid, cv=5, verbose = 2)
#fit model to data
knn_gscv.fit(X_smt_sub, y_smt_sub)
knn_gscv.best_params_
knn3 = KNeighborsClassifier(n_neighbors = knn_gscv.best_params_)
# Fit the classifier to the data
knn3.fit(X_smt_pc,y_smt)
y_pred_k_1 = knn3.predict(X_test_smt_pc)

confusion_matrix(y_test, y_pred_k_1)

print(accuracy_score(y_test, y_pred_k_1))

print(classification_report(y_test, y_pred_k_1))

print(roc_auc_score(y_test, y_pred_k_1))

# Random Forest
X_smt_sub2 = X_smt.iloc[sub_index,:]
clf = RandomForestClassifier() #Initialize with whatever parameters you want to

param_grid = {
                 'n_estimators': np.arange(100,120,1),
                 'max_depth': np.arange(30,40,1)
             }
grid_clf = GridSearchCV(clf, param_grid, cv=5, verbose = 2)
grid_clf.fit(X_smt_sub2,y_smt_sub)
grid_clf. best_params_
rf2 = RandomForestClassifier(max_depth= 36, n_estimators = 102)
# Fit the classifier to the data
rf2.fit(X_smt,y_smt)
y_pred_r1 = rf2.predict(X_test)

confusion_matrix(y_test, y_pred_r1)

print(accuracy_score(y_test, y_pred_r1))

print(classification_report(y_test, y_pred_r1))

print(roc_auc_score(y_test, y_pred_r1))



## Random Over Sampler

ros = RandomOverSampler(random_state=0)
X_ros, y_ros = ros.fit_resample(X_train, y_train)

# Visualization

fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(15, 7))
plot_resampling(X_train, y_train, ax=axs[0], title="Before Resampling")
plot_resampling(X_ros, y_ros,ax=axs[1], title="After Resampling")
fig.tight_layout()


# pca
pca = PCA(n_components = 0.95).fit(X_ros)
pca.n_components_
pca_col =list()
for i in np.arange(1,49,1):
    pca_col.append("PC_{}".format(i))
X_ros_pc = pca.transform(X_ros)
X_test_ros_pc = pca.transform(X_test)
X_ros_pc = pd.DataFrame(X_ros_pc, columns =pca_col )
X_test_ros_pc = pd.DataFrame(X_test_ros_pc, columns =pca_col)

X_ros_sub = X_ros_pc.iloc[sub_index,:]
y_ros_sub = y_ros[sub_index]

# KNN
#create new a knn model
knn = KNeighborsClassifier()
#create a dictionary of all values we want to test for n_neighbors
param_grid = {'n_neighbors': np.arange(1, 100)}
#use gridsearch to test all values for n_neighbors
knn_gscv = GridSearchCV(knn, param_grid, cv=5, verbose = 2)
#fit model to data
knn_gscv.fit(X_ros_sub, y_ros_sub)
knn_gscv.best_params_
knn4 = KNeighborsClassifier(n_neighbors = 15)
# Fit the classifier to the data
knn4.fit(X_ros_pc,y_ros)
y_pred_k2 = knn4.predict(X_test_ros_pc)

confusion_matrix(y_test, y_pred_k2)

print(accuracy_score(y_test, y_pred_k2))

print(classification_report(y_test, y_pred_k2))

print(roc_auc_score(y_test, y_pred_k2))

# Random Forest
X_ros_sub2 = X_ros.iloc[sub_index,:]
clf = RandomForestClassifier() #Initialize with whatever parameters you want to

param_grid = {
                 'n_estimators': np.arange(100,120,1),
                 'max_depth': np.arange(30,40,1)
             }
grid_clf = GridSearchCV(clf, param_grid, cv=5, verbose = 2)
grid_clf.fit(X_ros_sub2, y_ros_sub)
grid_clf.best_params_
rf2 = RandomForestClassifier(max_depth= 37, n_estimators = 108)
# Fit the classifier to the data
rf2.fit(X_ros,y_ros)
y_pred_r2 = rf2.predict(X_test)

confusion_matrix(y_test, y_pred_r2)

print(accuracy_score(y_test, y_pred_r2))

print(classification_report(y_test, y_pred_r2))

print(roc_auc_score(y_test, y_pred_r2))


## SMOTE

sm = SMOTE(random_state=42)
X_sm, y_sm = sm.fit_resample(X_train, y_train)

# visualization
fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(15, 7))
plot_resampling(X_train, y_train, ax=axs[0], title="Before Resampling")
plot_resampling(X_sm, y_sm,ax=axs[1], title="After Resampling")
fig.tight_layout()

# PCA
pca = PCA(n_components = 0.95).fit(X_sm)
pca.n_components_
X_sm_pc = pca.transform(X_sm)
X_test_sm_pc = pca.transform(X_test)
pca_col =list()

for i in np.arange(1,pca.n_components_+1,1):
    pca_col.append("PC_{}".format(i))

X_sm_pc = pd.DataFrame(X_sm_pc, columns =pca_col )
X_test_sm_pc = pd.DataFrame(X_test_sm_pc, columns =pca_col)
X_sm_sub = X_sm_pc.iloc[sub_index,:]
y_sm_sub = y_sm[sub_index]

# KNN
knn = KNeighborsClassifier()
#create a dictionary of all values we want to test for n_neighbors
param_grid = {'n_neighbors': np.arange(1, 100)}
#use gridsearch to test all values for n_neighbors
knn_gscv = GridSearchCV(knn, param_grid, cv=5, verbose = 2)
#fit model to data
knn_gscv.fit(X_sm_sub, y_sm_sub)
knn_gscv.best_params_
knn3 = KNeighborsClassifier(n_neighbors = 87)
# Fit the classifier to the data
knn3.fit(X_sm_pc,y_sm)

y_pred_k_3 = knn3.predict(X_test_sm_pc)

confusion_matrix(y_test, y_pred_k_3)

print(accuracy_score(y_test, y_pred_k_3))

print(classification_report(y_test, y_pred_k_3))

print(roc_auc_score(y_test, y_pred_k_3))

# Random Forest
X_sm_sub2 = X_sm.iloc[sub_index,:]
clf = RandomForestClassifier() #Initialize with whatever parameters you want to

param_grid = {
                 'n_estimators': np.arange(100,120,1),
                 'max_depth': np.arange(30,40,1)
             }
grid_clf = GridSearchCV(clf, param_grid, cv=5, verbose = 2)
grid_clf.fit(X_sm_sub2,y_sm_sub)
grid_clf. best_params_

rf2 = RandomForestClassifier(max_depth= 35, n_estimators = 108)
# Fit the classifier to the data
rf2.fit(X_sm,y_sm)
y_pred_r3 = rf2.predict(X_test)

confusion_matrix(y_test, y_pred_r3)

print(accuracy_score(y_test, y_pred_r3))

print(classification_report(y_test, y_pred_r3))

print(roc_auc_score(y_test, y_pred_r3))





