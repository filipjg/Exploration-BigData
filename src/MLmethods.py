import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import SpectralClustering, AgglomerativeClustering, KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, adjusted_rand_score
from sklearn import mixture
from sklearn.metrics.cluster import fowlkes_mallows_score
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from sklearn import metrics
from scipy.spatial.distance import cdist


""" Pre-process the data by scaling """

def preProcessing(X): 
    
    scaler = StandardScaler()
    scaledX = scaler.fit_transform(X)
    
    return scaledX

""" PCA - for reduction of dimension """

def pcaDimensionReduction(X, scaledX, y, xCircle, yCircle, numberOfComponentsPCA):
    
    pca = PCA(numberOfComponentsPCA)
    pcaDimensions = pca.fit_transform(scaledX)
    pcaExplainedVariance = pca.explained_variance_ratio_
   
    """ Creating a data frame for the top x pca components 
        and visualization of most important features using PCA and t-SNE """
 
    tSNE = TSNE(n_components=2)
    xTSNE = tSNE.fit_transform(scaledX)
    pcaVisualization = PCA(n_components=2)
    pcaVisualizationDimensions = pcaVisualization.fit_transform(scaledX)
    pcaExplainedVarianceVisualization = pcaVisualization.explained_variance_ratio_
       
    return pcaExplainedVariance, pcaDimensions, xTSNE, pcaVisualizationDimensions, pcaExplainedVarianceVisualization
   
    
""" Hirachical or agglomerative clustering based on the dimension reduction from PCA """

def hirachicalClustering(pcaDimensions, scaledX, xTSNE):

    clusteringAgglomerative = AgglomerativeClustering(distance_threshold=0, n_clusters=None)
    clusteringAgglomerative =  clusteringAgglomerative.fit(pcaDimensions)
    
    return clusteringAgglomerative
    

""" Spectral clustering - compare this results to k-Mean """

def spectralClustering(y, X, scaledX, xCircle, xTSNE, yCircle, numberOfClusters):
    
    spectralCircle = SpectralClustering(n_clusters=numberOfClusters, affinity='nearest_neighbors', random_state=10)
    spectralClusteringCircle = spectralCircle.fit_predict(xCircle)
    adjustedRandScoreSpectral = adjusted_rand_score(yCircle, spectralClusteringCircle)

    spectralClusteringRandomData = SpectralClustering(n_clusters=numberOfClusters, affinity='nearest_neighbors', random_state=10)
    spectralClusteringRandomData = spectralClusteringRandomData.fit_predict(scaledX)

    return spectralClusteringCircle, adjustedRandScoreSpectral, spectralClusteringRandomData
    

""" K-means clustering """

def kMeans(y, X, scaledX, xCircle, yCircle, numberOfClusters, randomState):
       
    """ Circle for comparison with spectral clustering """
    kMeanCircle = KMeans(n_clusters=2, random_state=10)
    kMeanCircleFit = kMeanCircle.fit_predict(xCircle)
    adjustedRandScorekMean = adjusted_rand_score(yCircle, kMeanCircleFit)
    
    kMeanRandomData = KMeans(n_clusters=numberOfClusters, random_state=randomState)
    kMeanRandomDataFit = kMeanRandomData.fit_predict(X)
        
    return kMeanCircleFit, adjustedRandScorekMean, kMeanRandomDataFit
    
""" Gaussian mixture model clustering """

def gaussianMixtureModel(y, X, scaledX, clusterVector):
    
    
    numberOfClusters = len(clusterVector)
    GaussianMixtureModelSilhouetteScore  = []
    GaussianMixtureModelSilhouetteScoreAverage = []
    
    """ Bayes Information Criterion """
    BICLowest = np.infty
    BIC = []
    numberOfCompontentsRange = range(2, numberOfClusters+2)
    
    covariance = ['full']
    
    for covariance in covariance:
        for n_components in numberOfCompontentsRange:
            
            GaussianMixtureModelCluster = mixture.GaussianMixture(n_components=n_components, covariance_type=covariance)
            GaussianMixtureModelCluster.fit(scaledX)
            GaussianMixtureModelClusterFit = GaussianMixtureModelCluster.fit_predict(scaledX)
            
            GaussianMixtureModelClusterFitAdjustedRand = adjusted_rand_score(y, GaussianMixtureModelClusterFit)
            GaussianMixtureModelClusterFitFMI = fowlkes_mallows_score(y, GaussianMixtureModelClusterFit)
            
            GaussianMixtureModelClusterSilhouette = silhouette_score(scaledX, GaussianMixtureModelClusterFit)
            GaussianMixtureModelSilhouetteScoreAverage.append(GaussianMixtureModelClusterSilhouette)
        
            BIC.append(GaussianMixtureModelCluster.bic(scaledX))
            if BIC[-1] < BICLowest:
                BIClowest = BIC[-1]
                bestGaussianMixtureModelCluster = GaussianMixtureModelCluster
    
    BIC = np.array(BIC)

    return GaussianMixtureModelCluster, BIC, GaussianMixtureModelClusterSilhouette, GaussianMixtureModelSilhouetteScoreAverage
    

def clusterAmountComparison(X, y, scaledX, xCircle, yCircle, clusterVector, randomState, numberOfCluster):
    numberOfComponentsPCA = 3
    
    distortions = []
    kMeanSilhouetteScoreAverageVector = []
    kMeanDaviesBouldinVector = []
    
    spectralClusterSilhouetteScoreVector = []
    spectralClusterDaviesBouldinVector = []
    
    agglomerativeClusterSilhouetteVector = []
    agglomerativeClusterDaviesBouldinVector = []
    
    kMeanFMVector = []
    spectralClusterFMVector = []
    agglomerativeClusterClusterFMVector = []
    
    
    for clusterNumber in clusterVector:

        """ K-means clustering """
        kMeanCluster = KMeans(n_clusters=clusterNumber, random_state=randomState)
        kMeanClusterFit = kMeanCluster.fit_predict(scaledX)
      
        kMeanSilhouetteScoreAverage = silhouette_score(scaledX, kMeanClusterFit, metric = 'euclidean')
        kMeanSilhouetteScoreAverageVector.append(kMeanSilhouetteScoreAverage)
        kMeanDaviesBouldin = metrics.davies_bouldin_score(scaledX, kMeanClusterFit)
        kMeanDaviesBouldinVector.append(kMeanDaviesBouldin)
        
        kMeanRandScore = adjusted_rand_score(y, kMeanClusterFit)
        kMeanFM = fowlkes_mallows_score(y, kMeanClusterFit)
        kMeanFMVector.append(kMeanFM)
        
        """ Spectral clustering """
        spectralCluster = SpectralClustering(n_clusters=clusterNumber, affinity='nearest_neighbors', random_state=randomState)
        spectralCluster = spectralCluster.fit_predict(scaledX)
       
        spectralClusterSilhouetteScore = silhouette_score(scaledX, spectralCluster)
        spectralClusterSilhouetteScoreVector.append(spectralClusterSilhouetteScore)
        
        spectralClusterDaviesBouldin = metrics.davies_bouldin_score(scaledX, spectralCluster)
        spectralClusterDaviesBouldinVector.append(spectralClusterDaviesBouldin)
        
        spectralClusterRandScore = adjusted_rand_score(y, spectralCluster)
        spectralClusterFM = fowlkes_mallows_score(y, spectralCluster)
        spectralClusterFMVector.append(spectralClusterFM)
        
        """Agglomerative clustering """
        agglomerativeCluster = AgglomerativeClustering(n_clusters=clusterNumber)
        agglomerativeCluster = agglomerativeCluster.fit_predict(scaledX) 
      
        agglomerativeClusterSilhouette = silhouette_score(scaledX, agglomerativeCluster)
        agglomerativeClusterSilhouetteVector.append(agglomerativeClusterSilhouette)
        
        agglomerativeClusterDaviesBouldin = metrics.davies_bouldin_score(scaledX, agglomerativeCluster)
        agglomerativeClusterDaviesBouldinVector.append(agglomerativeClusterDaviesBouldin)
    
        agglomerativeClusterRandScore = adjusted_rand_score(y, agglomerativeCluster)
        agglomerativeClusterClusterFM = fowlkes_mallows_score(y, agglomerativeCluster)
        agglomerativeClusterClusterFMVector.append(agglomerativeClusterClusterFM)
        
        distortions.append(sum(np.min(cdist(scaledX, kMeanCluster.cluster_centers_,'euclidean'), axis=1)) / scaledX.shape[0])
        
    return kMeanCluster, kMeanSilhouetteScoreAverageVector, kMeanRandScore, kMeanDaviesBouldinVector, kMeanFMVector, spectralCluster, spectralClusterSilhouetteScoreVector, spectralClusterDaviesBouldinVector, spectralClusterRandScore, spectralClusterFMVector, agglomerativeCluster, agglomerativeClusterSilhouetteVector, agglomerativeClusterDaviesBouldinVector, agglomerativeClusterRandScore, agglomerativeClusterClusterFMVector, distortions


       