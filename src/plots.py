import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pandas as pd
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram
from sklearn.cluster import SpectralClustering, AgglomerativeClustering, KMeans, OPTICS, cluster_optics_dbscan

def pcaPlot(X, y, pcaExplainedVariance, pcaVisualizationDimensions, xTSNE, pcaExplainedVarianceVisualization): 
    
    """ Feature importance PCA """
    indices = np.argsort(pcaExplainedVariance)[::-1]
    
    plt.figure()
    plt.title("Feature importances PCA (n_clusters=3)")
    plt.scatter(x=range(len(pcaExplainedVariance)), y=pcaExplainedVariance[indices], c='r')
    plt.xticks([], [])
    plt.xlim([-1, len(pcaExplainedVariance)])
    plt.ylabel("Importance metric")
    plt.xlabel("Feature index")
    
    """ Most informative visualization dimensions using PCA """
    plt.figure()
    plt.scatter(x=pcaVisualizationDimensions[:,0], y=pcaVisualizationDimensions[:,1])
    plt.title("PCA Results: Soil", weight='bold').set_fontsize('14')
    plt.xlabel("Dimension 1", weight='bold').set_fontsize('10')
    plt.ylabel("Dimension 2", weight='bold').set_fontsize('10')
    plt.grid()
    
    """ Most informative visualization dimensions using t-SNE """
    plt.figure()
    plt.scatter(x=xTSNE[:,0], y=xTSNE[:,1])
    plt.title("TSNE Results: Soil", weight='bold').set_fontsize('14')
    plt.xlabel("Dimension 1", weight='bold').set_fontsize('10')
    plt.ylabel("Dimension 2", weight='bold').set_fontsize('10')
    plt.grid()
    
    """ Distribution plot of clusters based on X-number of PCA components """

    FeatureVector = ["PCA" + str(num+1) for num in range(len(pcaExplainedVarianceVisualization))]
    XDataFrame = pd.DataFrame(pcaVisualizationDimensions, columns = FeatureVector)
    XDataFrame["labels"] = y
    sns.pairplot(XDataFrame, vars=XDataFrame.columns[:-1], hue="labels")
    plt.show()
    return 

       
""" Hirachiocal clustering plots """       

def hirarchicalClusteringPlot(clusteringAgglomerative, X):
    
    def plotDendrogram(model, **kwargs):
        collection = np.zeros(model.children_.shape[0])
        nSamples = len(model.labels_)
        
        for index, leafMerge in enumerate(model.children_):
            collectionIndex = 0
         
            for childIndex in leafMerge:
                if childIndex < nSamples: 
                    collectionIndex += 1
                else:
                    collectionIndex += collection[childIndex - nSamples] 
            collection[index] = collectionIndex 
        stacked_matrix = np.column_stack([model.children_, model.distances_,
                                          collection]).astype(float)
        dendrogram(stacked_matrix, truncate_mode='level', p=5)
        
        
    plt.figure()
    plotDendrogram(clusteringAgglomerative, truncate_mode='level', p=5)
    
    plt.title('Hierarchical Clustering Dendrogram (Ward-linkage)')
    plotDendrogram(clusteringAgglomerative, truncate_mode='level', p=5)
    plt.xlabel("Node based on PCA components")
    
    return 




""" Comparison of K-mean performance with spectral clustering using circle plots """
def circlePlots(xCircle, X, spectralClusteringCircle, kMeanCircle): 
    
    plt.figure()
    Spectral_colors = cm.nipy_spectral(spectralClusteringCircle.astype(float) / 2)
    plt.scatter(xCircle[:,0],y = xCircle[:,1], c=Spectral_colors, cmap='rainbow', alpha=0.7, edgecolors='b')
    plt.title('Spectral clustering')
    plt.grid()

    plt.figure()
    kMeanCircle = cm.nipy_spectral(kMeanCircle.astype(float) / 2)
    plt.scatter(xCircle[:,0],y = xCircle[:,1], c=kMeanCircle, cmap='rainbow', alpha=0.7, edgecolors='b')
    plt.title('K-mean clustering')
    plt.grid()
    
    return

def comparisonPlot(spectralClusteringRandomData, kMeanRandomDataFit, xTSNE, pcaVisualizationDimensions, y):
    
    plt.figure()
    spectralColors = cm.nipy_spectral(spectralClusteringRandomData.astype(float) / 2)
    plt.scatter(xTSNE[:,0], y = xTSNE[:,1], c=spectralColors, cmap='rainbow', alpha=0.7, edgecolors='b')
    plt.title('Spectral clustering')
    plt.grid()
    
    plt.figure()
    kMeanColors = cm.nipy_spectral(kMeanRandomDataFit.astype(float) / 2)
    plt.scatter(xTSNE[:,0], y = xTSNE[:,1], c=kMeanColors, cmap='rainbow', alpha=0.7, edgecolors='b')
    plt.title('K-mean clustering')
    plt.grid()
 
    return
   


def numberOfClusterPlot(clusterVector, distortions, kMeanSilhouetteScoreAverageVector, spectralClusterSilhouetteScoreVector, spectralClusterDaviesBouldinVector, kMeanDaviesBouldinVector, BIC, agglomerativeClusterSilhouetteVector):
    plt.figure()
    plt.title("Silhouette score using Agglomerative clustering")
    plt.xticks(np.array(clusterVector))
    plt.bar(x=np.array(clusterVector), height=agglomerativeClusterSilhouetteVector, color='g', width=0.45, align='center')    
    plt.xlabel("Number of clusters")
    plt.ylabel("Average Silhouette score")
    plt.grid()
    
    plt.figure()
    plt.plot(clusterVector, distortions, 'bx-')
    plt.xlabel('Number of cluster')
    plt.ylabel('Distortion')
    plt.title('The Elbow heuristic for K-mean')
    plt.grid()
    
    plt.figure()
    plt.title("Silhouette score using K-means")
    plt.xticks(np.array(clusterVector))
    plt.bar(x = np.array(clusterVector), height=kMeanSilhouetteScoreAverageVector, color='g', width=0.45, align='center')    
    plt.xlabel("Number of clusters")
    plt.ylabel("Average Silhouette score")
    plt.grid()
    
    plt.figure()
    plt.title("Davies bouldin score using K-mean clustering")
    plt.xticks(x=np.array(clusterVector))
    plt.plot(np.array(clusterVector), kMeanDaviesBouldinVector)
    plt.xlabel("Number of clusters")
    plt.ylabel("Davies bouldin score")
    plt.grid()
    
    plt.figure()
    plt.title("Silhouette score using Agglomerative clustering")
    plt.xticks(np.array(clusterVector))
    plt.bar(x=np.array(clusterVector), height=agglomerativeClusterSilhouetteVector, color='g', width=0.45, align='center')    
    plt.xlabel("Number of clusters")
    plt.ylabel("Average Silhouette score")
    plt.grid()
    
    plt.figure()
    plt.title("Silhouette score using spectral clustering")
    plt.xticks(np.array(clusterVector))
    plt.bar(x=np.array(clusterVector), height=spectralClusterSilhouetteScoreVector, color='g', width=0.45, align='center')    
    plt.xlabel("Number of clusters")
    plt.ylabel("Average Silhouette score")
    plt.grid()
    
    plt.figure()
    plt.title("Davies bouldin score using spectral clustering")
    plt.xticks(x=np.array(clusterVector))
    plt.plot(np.array(clusterVector), spectralClusterDaviesBouldinVector)
    plt.xlabel("Number of clusters")
    plt.ylabel("Davies bouldin score")
    plt.grid()
    
    plt.figure()
    plt.xticks(np.array(clusterVector))
    plt.ylim([BIC.min() * 1.01 - 0.01 * BIC.max(), BIC.max()])  
    plt.bar(np.array(clusterVector), height=BIC, color='g', width=0.45, align='center')
    plt.xlabel("Number of clusters")
    plt.ylabel("BIC-score")  
    plt.title('Gaussian mixture model - Bayes Information Criteria')
    plt.grid()   
    return 
    
def FowlkesMallowsScore(kMeanFMVector, spectralClusterFMVector, agglomerativeClusterClusterFMVector, clusterVector):
    
    plt.figure()
    plt.plot(clusterVector, kMeanFMVector, color="r")
    plt.plot(clusterVector, spectralClusterFMVector, color="b")
    plt.plot(clusterVector, agglomerativeClusterClusterFMVector, color="black")
    plt.legend(["K-means","Spectral","Hirachical"])
    plt.xlabel("Number of clusters")
    plt.ylabel("Fowlkes Mallows Score")
    plt.title("Fowlkes Mallows Score")
    plt.grid()
    return