import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns 
sns.set()

from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import confusion_matrix as cm

import os
starting_directory = os.getcwd()
os.chdir("archive_dataframes")                            


df = pd.read_csv(os.listdir()[0])                       

for i in range(1,10):
    df_dummy = pd.read_csv(os.listdir()[i])               
    list_of_dataframes = [df, df_dummy]
    df = pd.concat(list_of_dataframes, ignore_index=True)    



os.chdir(starting_directory)
print(df.shape)
df.head()



X = df.iloc[:,1:-1]           
y = df.iloc[:,-1]                 
print("X:",X.shape, "   y:", y.shape)



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
print(X_train.shape, X_test.shape)



num_components = range(1,41)
explained_var_ratio = []



for n in num_components:  
    pca = PCA(n_components=n)
    pca.fit(X_train)
    explained_var_ratio.append(np.sum(pca.explained_variance_ratio_))



plt.plot(num_components, explained_var_ratio, marker='o')
plt.xlabel('Number of Components')
plt.ylabel('Explained Variance Ratio')
plt.title('Elbow Method for PCA')
plt.show()



pca = PCA(n_components=20)
X_train  = pca.fit_transform(X_train)
X_test = pca.transform(X_test)
print(X_train.shape, X_test.shape)



scalar  = StandardScaler()
X_train = scalar.fit_transform(X_train)
X_test = scalar.transform(X_test)
pd.DataFrame(X_train).head()



for i in range(1,10):
    knn_model = KNeighborsClassifier(n_neighbors=i, metric='euclidean')                 # Model creation
    knn_model.fit(X_train,y_train)                                                      # Model Training
    y_pred = knn_model.predict(X_test)                                                  # Predicting the Results

    print(f"for number of neighbors = {i}, the accuracy of the model is {round(sum(100*(y_pred==y_test))/len(y_pred),2)}%")

# %%
for i in range(1,10):
    knn_model = KNeighborsClassifier(n_neighbors=i, metric='manhattan')                 # Model creation
    knn_model.fit(X_train,y_train)                                                      # Model Training
    y_pred = knn_model.predict(X_test)                                                  # Predicting the Results

    from sklearn.metrics import confusion_matrix
    print(f"for number of neighbors = {i}, the accuracy of the model is {round(sum(100*(y_pred==y_test))/len(y_pred),2)}%")

# %%
for i in range(1,10):
    knn_model = KNeighborsClassifier(n_neighbors=i, metric='chebyshev')                 # Model creation
    knn_model.fit(X_train,y_train)                                                      # Model Training
    y_pred = knn_model.predict(X_test)                                                  # Predicting the Results

    from sklearn.metrics import confusion_matrix
    print(f"for number of neighbors = {i}, the accuracy of the model is {round(sum(100*(y_pred==y_test))/len(y_pred),2)}%")

# %%
for i in range(1,10):
    knn_model = KNeighborsClassifier(n_neighbors=i, metric='cosine')                    # Model creation
    knn_model.fit(X_train,y_train)                                                      # Model Training
    y_pred = knn_model.predict(X_test)                                                  # Predicting the Results
    if i == 3:
        y_pred2 = y_pred

    from sklearn.metrics import confusion_matrix
    print(f"for number of neighbors = {i}, the accuracy of the model is {round(sum(100*(y_pred==y_test))/len(y_pred),2)}%")

# %%
for i in range(1,10):
    knn_model = KNeighborsClassifier(n_neighbors=i, metric='hamming')                   # Model creation
    knn_model.fit(X_train,y_train)                                                      # Model Training
    y_pred = knn_model.predict(X_test)                                                  # Predicting the Results

    from sklearn.metrics import confusion_matrix
    print(f"for number of neighbors = {i}, the accuracy of the model is {round(sum(100*(y_pred==y_test))/len(y_pred),2)}%")

# %% [markdown]
# ### Decision Tree Model

# %%
clf_tree = DecisionTreeClassifier(criterion='gini', max_depth=4, random_state=1)        # Model creation
clf_tree.fit(X_train, y_train)                                                          # Model Training
y_pred = knn_model.predict(X_test)                                                      # Predicting the Results

 
print(f"the accuracy of the model is {round(sum(100*(y_test==y_pred))/len(y_pred),2)}%")

# %% [markdown]
# ### Logistic Regression

# %%
log_model = LogisticRegression(solver='liblinear')
log_model.fit(X_train,y_train)
y_pred = log_model.predict(X_test) 

print(f"the accuracy of the model is {round(sum(100*(y_pred==y_test))/len(y_pred),2)}%")

# %% [markdown]
# ### Random Forest

# %%
forest_model = RandomForestClassifier(n_estimators = 100, random_state=5)
forest_model.fit(X_train,y_train)                                  # Model Training
y_pred = forest_model.predict(X_test) 
y_pred1 = y_pred

print(f"the accuracy of the model is {round(sum(100*(y_pred==y_test))/len(y_pred),2)}%")

# %%


# %% [markdown]
# #### We calculated these accuracy values for traning set sampling of 70%, 80%, 90%. In every time random forest model and KNN models gave consistant results around 90% results.

# %%


# %% [markdown]
# #### The best results were obtained by KNN model with distance matric "cosine". For KNN we get maximum accuracy when number of neighbors is 1, But we came to conclusion that it could be because overfitting fit. Apart from that maximum efficiency was obtainded when number of neighbors is 3, with accuracy of model is in decreasing motion with decrease in taining set size.

# %%


# %% [markdown]
# #### Here is the resulting confusion matrises for these 2 models

# %%
# y_pred1 is from random forest model, y_pred2 is from Knn model

cm1 = cm(y_test, y_pred1) 
cm2 =cm(y_test, y_pred2)

# %%
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(20, 8))
sns.heatmap(cm1, annot=True, fmt="d", cmap="Blues", cbar=False, square=True, ax=ax[0])
sns.heatmap(cm2, annot=True, fmt="d", cmap="Greens", cbar=False, square=True, ax=ax[1])

# Set labels, title, and ticks
class_namess = np.unique(np.concatenate((y_test, y_pred)))
ax[0].set_xlabel('Predicted Labels')
ax[0].set_ylabel('True Labels')
ax[0].set_title('Forest Model Confusion Matrix')

ax[1].set_xlabel('Predicted Labels')
ax[1].set_ylabel('True Labels')
ax[1].set_title('KNN model Confusion Matrix')

plt.setp(ax[:], xticks=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], xticklabels=class_namess)
plt.setp(ax[:], yticks=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], yticklabels=class_namess)

# Rotate x-axis labels if needed
for a in ax:
    a.tick_params(axis='y', labelrotation=45)
    a.tick_params(axis='x', labelrotation=45)

# Showing the plot
plt.show()


# %%


# %% [markdown]
# ### Checking the model

# %%
def prediction(b):
    b = pca.transform(pd.DataFrame([b]))
    b = scalar.transform(b)
    return forest_model.predict(b) 


# %%
prediction( df.iloc[0,1:-1].tolist() )

# %%



