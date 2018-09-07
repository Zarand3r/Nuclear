import pandas as pd
from sklearn.datasets import make_classification
from sklearn.datasets import make_blobs
from sklearn.datasets import make_gaussian_quantiles
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import numpy as np
import sys

# set seed for tsne
np.random.seed(6)

nuclear_data = pd.read_csv("reduced.csv")
sub_nuc_data = nuclear_data[['AreaShape_Area','AreaShape_Compactness','AreaShape_Eccentricity','AreaShape_Extent','AreaShape_FormFactor','AreaShape_MaximumRadius','AreaShape_MeanRadius','AreaShape_Perimeter','Intensity_IntegratedIntensityEdge_Enhanced_Inverted','Intensity_IntegratedIntensity_Enhanced_Inverted','Intensity_MaxIntensity_Enhanced_Inverted','Intensity_MeanIntensity_Enhanced_Inverted','AreaShape_Solidity','AreaShape_MajorAxisLength']]
# sub_nuc_data = nuclear_data[['AreaShape_Area','AreaShape_Compactness','AreaShape_Eccentricity','AreaShape_Extent','AreaShape_FormFactor','AreaShape_MaximumRadius','AreaShape_Perimeter','Intensity_MaxIntensity_Enhanced_Inverted','AreaShape_Solidity']]
# sub_nuc_data = nuclear_data[['AreaShape_Area','AreaShape_MaximumRadius','Intensity_MaxIntensity_Enhanced_Inverted']]


#print sub_nuc_data
# print nuclear_data
frac_nuc_data  = sub_nuc_data.sample(n=800)

# X, y = make_blobs()
X = frac_nuc_data
y = frac_nuc_data['AreaShape_MaximumRadius']   #Intensity_MaxIntensity_Enhanced_Inverted  #AreaShape_Area  #Intensity_MeanIntensity_Enhanced_Inverted
#y = frac_nuc_data['AreaShape_MeanRadius']   #Intensity_MaxIntensity_Enhanced_Inverted  #AreaShape_Area  #Intensity_MeanIntensity_Enhanced_Inverted

# X, y = make_blobs(n_samples=1000, n_features=6, centers=4, cluster_std=3.6)   #2.7


####### PCA code
pca = PCA()   #n_components = 2
pca_results = pca.fit_transform(X)

####### Tsne code
perplexity = 10   #10  #20
tsne = TSNE( verbose=1, perplexity=perplexity, n_iter=2000)  # perplexity = 40,20   n_components=2,
tsne_results = tsne.fit_transform(X)

#sys.exit()
#################
h = plt.figure(2)
plt.scatter(pca_results[:,0],pca_results[:,1],c = y,s=3)  #,c = y  #,pca_results[:,2]
plt.title("PCA")

g = plt.figure(3)
plt.scatter(tsne_results[:,0],tsne_results[:,1],c = y,s=3)  #,c = y
plt.title("tSNE")
plt.show()









#V = pca.components_
#pca_recovered = pca.inverse_transform(pca_results)


#plt.scatter(X,Y, c=C) # 'ro', label='PCA'
#plt.show()

#X_train, X_test, Y_train, Y_test = sklearn.cross_validation.train_test_split(X, Y, test_size = 0.33, random_state = 5)
#print(X_train.shape)
#print(X_test.shape)
#print(Y_train.shape)
#print(Y_test.shape)
