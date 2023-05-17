from sklearn.datasets import make_classification
from sklearn import datasets

def generateData(numberOfFeatures, numberInformativeFeatures, numberOfRedundantFeatures, numberOfRepeatedFeatures, numberOfClasses, 
                 clusterPerClass, weightsOfClass, numberOfSamples, classSeparator, hypercube, shift, scale, shuffle, randomState, noise, factor):
    
    
    X, y = make_classification(n_samples = numberOfSamples,
                               n_features = numberOfFeatures,
                               n_informative = numberInformativeFeatures,
                               n_redundant = numberOfRedundantFeatures,
                               n_repeated = numberOfRepeatedFeatures,
                               n_classes = numberOfClasses,
                               n_clusters_per_class = clusterPerClass,
                               weights=weightsOfClass,
                               class_sep = classSeparator,
                               hypercube=hypercube,
                               shift=shift,
                               scale=scale,
                               shuffle=shuffle,
                               random_state=randomState)
    
    circle, yCircle = datasets.make_circles(n_samples = numberOfSamples,
                                            shuffle = shuffle,
                                            noise = noise,
                                            random_state = randomState,
                                            factor = factor)
    
    return X, y, circle, yCircle
    