
from src.data import generateData
from src.MLmethods import preProcessing, pcaDimensionReduction, hirachicalClustering, spectralClustering, kMeans, gaussianMixtureModel, clusterAmountComparison
from src.plots import pcaPlot, hirarchicalClusteringPlot, circlePlots, comparisonPlot, numberOfClusterPlot, FowlkesMallowsScore
	
import warnings
warnings.filterwarnings('ignore') # setting ignore as a parameter

def main():
    
    """ Inputs """ 
    numberOfFeatures = 80
    numberInformativeFeatures = 50
    numberOfRedundantFeatures = 30
    numberOfRepeatedFeatures = 0
    numberOfClasses = 3
    clusterPerClass = 1
    numberOfCluster = 3
    clusterVector = [2, 3, 4, 5, 6, 7]
    weightsOfClass = None
    numberOfSamples = 2000
    classSeparator = 15.0
    hypercube = False
    shift = 0.0
    scale = 1.0 
    shuffle = True
    randomState = None
    noise = 0.1
    factor = 0.5
    pcaComponents = 50
    
    X, y, xCircle, yCircle = generateData(numberOfFeatures, numberInformativeFeatures, numberOfRedundantFeatures, numberOfRepeatedFeatures, numberOfClasses,
                                         clusterPerClass, weightsOfClass, numberOfSamples, classSeparator, hypercube, shift, scale, shuffle, randomState, noise, factor)

    
    
    """ Pre-processing """
    scaledX = preProcessing(X)  
    
    """ PCA """
    pcaExplainedVariance, pcaDimensions, xTSNE, pcaVisualizationDimensions, pcaExplainedVarianceVisualization = pcaDimensionReduction(X, scaledX, y, xCircle, yCircle, pcaComponents)
    
    """ Hirachical clustering """   
    clusteringAgglomerative = hirachicalClustering(pcaDimensions, scaledX, xTSNE)
    
    """ Spectral clustering """
    spectralClusteringCircle, adjustedRandScoreSpectral, spectralClusteringRandomData = spectralClustering(y, X, scaledX, xCircle, xTSNE, yCircle, numberOfCluster)
    
    """ k-mean clustering """
    kMeanCircleFit, adjustedRandScorekMean, kMeanRandomDataFit = kMeans(y, X, scaledX, xCircle, yCircle, numberOfCluster, randomState)

    """ Gaussian Mixture Model """
    GaussianMixtureModelCluster, BIC, GaussianMixtureModelClusterSilhouette, GaussianMixtureModelSilhouetteScoreAverage = gaussianMixtureModel(y, X, scaledX, clusterVector)

    """ Counting the number of clusters """
    kMeanCluster, kMeanSilhouetteScoreAverageVector, kMeanRandScore, kMeanDaviesBouldinVector, kMeanFMVector, spectralCluster, spectralClusterSilhouetteScoreVector, spectralClusterDaviesBouldinVector, spectralClusterRandScore, spectralClusterFMVector, agglomerativeCluster, agglomerativeClusterSilhouetteVector, agglomerativeClusterDaviesBouldinVector, agglomerativeClusterRandScore, agglomerativeClusterClusterFMVector, distortions = clusterAmountComparison(X, y, scaledX, xCircle, yCircle, clusterVector, randomState, numberOfCluster)

    """ Visualization of results """
    pcaPlot(X, y, pcaExplainedVariance, pcaVisualizationDimensions, xTSNE, pcaExplainedVarianceVisualization) 
    hirarchicalClusteringPlot(clusteringAgglomerative, X)
    circlePlots(xCircle, X, spectralClusteringCircle, kMeanCircleFit)
    comparisonPlot(spectralClusteringRandomData, kMeanRandomDataFit, xTSNE, pcaVisualizationDimensions, y)
    numberOfClusterPlot(clusterVector, distortions, kMeanSilhouetteScoreAverageVector, spectralClusterSilhouetteScoreVector, spectralClusterDaviesBouldinVector, kMeanDaviesBouldinVector, BIC, agglomerativeClusterSilhouetteVector)
    FowlkesMallowsScore(kMeanFMVector, spectralClusterFMVector, agglomerativeClusterClusterFMVector, clusterVector)

if __name__=='__main__':    
    
   main()
