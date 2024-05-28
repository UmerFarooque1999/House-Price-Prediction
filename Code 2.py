import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.base import BaseEstimator, TransformerMixin ### Always import these 2 when you are developing custom class for pipeline
from sklearn.metrics import mean_squared_error, accuracy_score

### IMPORTING DATA
housing_data = pd.read_csv('data.csv')
#print(housing_data.info())

### VISUALIZING THE DATA
histogram_fig, ax = plt.subplots()
histogram = ax.hist(housing_data, bins=50, alpha=0.5)
ax.set_xticklabels(housing_data.columns)
#plt.show()
#plt.close()
box_plot = plt.boxplot(housing_data)
#plt.show()
#plt.close()

### SPLITTING THE DATA
split = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=7)
for train_index, test_index in split.split(housing_data, y=housing_data['CHAS']):
    strat_train = housing_data.loc[train_index]
    strat_test = housing_data.loc[test_index]
train_set_with_outliers = pd.DataFrame(strat_train)
test_set = pd.DataFrame(strat_test)

### OUTLIER REMOVAL FROM TRAINNIG SET
def outlier_finder(df, f):
    Q1 = df[f].quantile(0.25)
    Q3 = df[f].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    outlier_index = df.index[(df[f] < lower) | (df[f] > upper)]
    return outlier_index
def outlier_remover(df, f):
    index_list = list()
    for feature in f:
        index_list.extend(outlier_finder(df, feature))
    index_list = sorted(set(index_list))
    df.drop(index=index_list, inplace=True)
    #print('outlier indices: ', index_list)
    return df
train_set_without_outliers_1 = outlier_remover(train_set_with_outliers, ['CRIM', 'ZN', 'B'])
print(train_set_without_outliers_1.info())

### SEPARATING X AND Y
X_train = np.array(train_set_without_outliers_1.copy().iloc[:, 0:-1])
y_train = np.array(train_set_without_outliers_1.iloc[:, -1])
### CORRELATION AND FEATURE ENGINEERING
class feature_edition(BaseEstimator, TransformerMixin):
    def fit(self, df, y=None):
        return self
    def transform(self, df, y=None):
        new_TAXRM_column = df[:, 9] / df[:, 5]
        new_LSTAT_CRIM_column = df[:, 0] * df[:, 12]
        new_df = np.column_stack((df, new_TAXRM_column, new_LSTAT_CRIM_column))
        return new_df
        
editor = feature_edition()
train_set_without_outliers = editor.fit_transform(X_train)
#correlation = train_set_without_outliers.corr()
#print("Correlation: \n", correlation, "\n#########################################")

### PREPROCESSING THE DATA
imputer = SimpleImputer()
X_train_imputed = imputer.fit_transform(train_set_without_outliers)
standard_scaler = StandardScaler()
X_train_imputed_Standardized = standard_scaler.fit_transform(X_train_imputed)
#print("X_train:\n", X_train_imputed_Standardized.shape)

### SELECTING THE MODEL
clf = RandomForestRegressor(n_estimators=5, max_depth=3, random_state=7)
clf2 = RandomForestRegressor(random_state=7) 
clf.fit(X_train_imputed_Standardized, y_train)
y_pred = clf.predict(X_train_imputed_Standardized)
error = mean_squared_error(y_train, y_pred)
print("Mean squared Error without the pipeline is: {}".format(error))
params = {"n_estimators": [1, 3, 5], "max_depth": [2, 3, 5, 7]}
Grid = GridSearchCV(clf2, param_grid=params, cv=3)
### CREATING THE PIPELINE FOR TEST SET AND FUTURE
Pipeline_1 = Pipeline([
    ("feature addition", editor),
    ("Imputation", imputer),
    ("Standardization", standard_scaler),
    ("Model", Grid)
])


### EVALUATING THE MODEL ON TRAINING DATA 
X_train_pipeline = np.array(train_set_without_outliers_1.drop(columns='MEDV'))
y_train_pipeline = np.array(train_set_without_outliers_1['MEDV'])

Pipeline_1.fit(X_train_pipeline, y_train_pipeline)
y_pred_pipeline = Pipeline_1.predict(X_train_pipeline)
#Grid.fit(X_train_pipeline, y_train_pipeline)
#y_pred_pipeline = Grid.predict(X_train_pipeline)
error2 = mean_squared_error(y_train_pipeline, y_pred_pipeline)
print("Mean Squared Error through pipeline is: {}".format(error2))
print("Root Mean Squared Error through pipeline is: {}".format(np.sqrt(error2)))
print("Parameters of pipeline: ", Pipeline_1.named_steps["Model"].best_params_)

### EVALUATING THE MODEL ON TEST DATA
test_set
X_test = np.array(test_set.copy().iloc[:, 0:-1])
y_test = np.array(test_set.copy().iloc[:, -1])
y_test_pred = Pipeline_1.predict(X_test)
error3 = mean_squared_error(y_test, y_test_pred)
print("Root Mean Squared Error of test data through pipeline is: {}".format(np.sqrt(error3)))





