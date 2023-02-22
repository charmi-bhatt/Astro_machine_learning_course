#This code is divided into 5 sections: data preprocessing and then one for each of the 4 machine learning techniques used. 
#Section are named as follows:
#1. Data pre-processing 
#2. Correlation Matrix 
#3. PCA 
#4. t-SNE
#5. Supervised Learning - Regression (Linear, Gradient Boosting and Random Forest)

#Data used includes 11 parameters across 133 sightlines. 
#Those parameters are physical propoerties of the sightline and equivalent width of 8 diffuse interstelar bands (DIBs). 


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor



#1. Data Pre-processing

raw_data = pd.read_csv(r"C:\Users\Charmi Bhatt\OneDrive\Desktop\AstroML\Project\scott_data.txt", delim_whitespace=(True)) #, header= [0:])
raw_data = raw_data.fillna(0)
EBV = raw_data.iloc[:,2] #interstellar extinction
pd.to_numeric(EBV)
logNH = raw_data.iloc[:,3] #Atomic Hydrogen abundance 
pd.to_numeric(logNH)
logNH2 = raw_data.iloc[:,6] #Molecular Hydrogen abundance 
pd.to_numeric(logNH2)

#8 DIBs with last 4 digits representing their wavelength in Angstroms. 
DIB_5487 = raw_data.iloc[:,9]
pd.to_numeric(DIB_5487)
DIB_5705 = raw_data.iloc[:,12]
pd.to_numeric(DIB_5705)
DIB_5780 = raw_data.iloc[:,15]
pd.to_numeric(DIB_5780)
DIB_5797 = raw_data.iloc[:,18]
pd.to_numeric(DIB_5797)
DIB_6196 = raw_data.iloc[:,21]
pd.to_numeric(DIB_6196)
DIB_6204 = raw_data.iloc[:,24]
pd.to_numeric(DIB_6204)
DIB_6283 = raw_data.iloc[:,27]
pd.to_numeric(DIB_6283)
DIB_6614 = raw_data.iloc[:,30]
pd.to_numeric(DIB_6614)

#creating pandas DataFrame of the data we need. Shape:(133,11)
data = np.array([EBV, logNH, logNH2, DIB_5487, DIB_5705, DIB_5780, DIB_5797, DIB_6196, DIB_6204, DIB_6283, DIB_6614,]).transpose()
data = pd.DataFrame({'E(B-V)':data[:,0], 'log(N(H))':data[:,1], 'log(N(H2))':data[:,2], 'DIB_5487': data[:,3], 'DIB_5705':data[:,4], 'DIB_5780':data[:,5], 'DIB_5797':data[:,6], 'DIB_6196':data[:,7], 'DIB_6204' :data[:,8], 'DIB_6283':data[:,9], 'DIB_6614':data[:,10]})


#2. Correlation matrix
# It displays how 11 parameters are correlated to each other
corr_matrix = data.corr()

#plotting correlation using heatmap
sns.set (rc = {'figure.figsize':(9, 7)})
sns.heatmap(corr_matrix, annot=True, linewidths=.5)
plt.show()


#3. PCA 
#It helps us reduce dimensionality of the data

#standardize data before appyling PCA algorithm 
scaler = StandardScaler()
scaler.fit(data) #demean and std dev = 1
data_rescaled = scaler.transform(data)


pca = PCA(n_components = 0.95) #n_components = 0.95 = components required to capture 95% variance
pca.fit(data_rescaled) 

print("Explained variance ratio: ")
print(pca.explained_variance_ratio_)

print("PCA components: ")
print(pca.components_)

print("PCA explained varinace: ")
print(pca.explained_variance_)

print("PCA noise varinace: ")
print(pca.noise_variance_)

#We got 4 components that capture 95% variance in the data
#Visual representation of Cumulative variance by each PCA component
plt.rcParams["figure.figsize"] = (12,6)
fig, ax = plt.subplots()
xi = np.arange(1, 5, step=1)
y = np.cumsum(pca.explained_variance_ratio_)
plt.ylim(0.0,1.1)
plt.plot(xi, y, marker='o', linestyle='--', color='b')

plt.xlabel('Number of Components')
plt.xticks(np.arange(0, 5, step=1)) #change from 0-based array index to 1-based human-readable label
plt.ylabel('Cumulative variance (%)')
plt.title('The number of components needed to explain variance')

plt.axhline(y=0.95, color='r', linestyle='-')
plt.text(0.5, 0.85, '95% cut-off threshold', color = 'red', fontsize=16)

plt.show()

#Transform data into PCA components
data_pca = pca.transform(data_rescaled)
# Plot the transformed data
plt.scatter(data_pca[:, 0], data_pca[:, 1])
plt.xlabel('PCA 1')
plt.ylabel('PCA 2')
plt.show()

#Using t-SNE as it can handle non linear relations between different features in the dataset, unlike PCA which can only do linear relations. 
# Create a t-SNE object.
tsne = TSNE(n_components=2, perplexity=20, n_iter=4000)
# Fit the t-SNE object to the data
data_tsne = tsne.fit_transform(data_rescaled)

# Plot the transformed data
plt.scatter(data_tsne[:, 0], data_tsne[:, 1])
plt.xlabel('t-SNE 1')
plt.ylabel('t-SNE 2')
plt.show()



#5. Supervised Learning - Regression

# Here we aim to train the machine to estimate log(N(H)) given the equivalent width of 8 diffuse interstellar bands. 
#Thus, for now, we remove all other parameters except equivalent widths from the dataset and name the new data "Known".  
#Since we are training the machine to estimate log(N(H)), it is called "to_be_predicted" dataset. 
 
to_be_predicted = data['log(N(H))'] #similar methods can be applied to log(N(H2))
known = data.drop(['E(B-V)','log(N(H))','log(N(H2))'] , axis=1)

#testing the dimesnions of new datasets
print("To be predicted dataset dimenisons:", to_be_predicted.shape)
print("Known dataset dimenisons: ", known.shape)


#Pairplot visualization 
sns.pairplot(data)


#Splitting data into training (80%) and testing(20%) datasets
train_data, test_data = train_test_split(known, test_size=0.2)
train_labels, test_labels = train_test_split(to_be_predicted, test_size=0.2)


# Fit a linear regression model to the data
reg = LinearRegression()
reg.fit(train_data, train_labels)
# Predict the labels of the test data
predicted_labels = reg.predict(test_data)
# Show the mean squared error
print("Linear regression")
print("Mean squared error: ", mean_squared_error(test_labels, predicted_labels))
# Show the average percentage error
print("Average percentage error: ", np.mean(np.abs((test_labels - predicted_labels)/test_labels))*100)


# Fit a Gradient boosting regressor to the data
gbr = GradientBoostingRegressor()
gbr.fit(train_data, train_labels)
# Predict the labels of the test data
predicted_labels = gbr.predict(test_data)
# Show the mean squared error
print("Gradient boosting")
print("Mean squared error: ", mean_squared_error(test_labels, predicted_labels))
# Show the average percentage error
print("Average percentage error: ", np.mean(np.abs((test_labels - predicted_labels)/test_labels))*100)


# Fit a random forest regressor to the data
rfr = RandomForestRegressor()
rfr.fit(train_data, train_labels)
# Predict the labels of the test data
predicted_labels = rfr.predict(test_data)
# Show the mean squared error
print("Random forest")
print("Mean squared error: ", mean_squared_error(test_labels, predicted_labels))
# Show the average percentage error
print("Average percentage error: ", np.mean(np.abs((test_labels - predicted_labels)/test_labels))*100)

#Conclusion: It can be seen from last piece of the code that how equivalent width of few diffuse interstellar bands can be used to estimate hydrogen abundance 
#in line of sight with average percentage error below 2% using Random Forest regression model.
