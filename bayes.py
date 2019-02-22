
import pandas as pd
import numpy as np
from collections import Counter

class NaiveBayes:
    def __init__(self,dataset):
        self.__dataset = dataset
        self.__classAttribute = list(self.__dataset.keys())[-1]
        self.__classes = list(set(self.__dataset[self.__classAttribute]))
        self.__attributes = list(self.__dataset.keys())[:-1]
        self.__classProbabilites = dict()
        
        
        
        
    def train(self):
        self.__classCounts = dict(Counter(self.__dataset[self.__classAttribute]))
        self.__featureProbabilites = {}
        self.__initClassProbabilities()
        self.__initFeatureProbabilites()
        
        
    
    def __initClassProbabilities(self):
        counts = dict(Counter(self.__dataset[self.__classAttribute]))
        totalNumberOfTuples = sum(counts.values())
        self.__classProbabilites = {key:self.__getProbability(counts[key],totalNumberOfTuples) for key in counts.keys()}
        
    
    

    def __initFeatureProbabilites(self):
        for attribute in self.__attributes:
            data = {}
            attributeCounts = len(self.__getAttributeValues(attribute))#for laplace correction
            for attributeValue in self.__getAttributeValues(attribute):
                probabilities = {}
                for classValue in self.__classes:
                    probability = self.__getProbability(1+self.__getCounts((attribute,attributeValue),(self.__classAttribute,classValue)),attributeCounts+self.__classCounts[classValue])
                    probabilities[classValue] = probability
                data[attributeValue] = probabilities
            self.__featureProbabilites[attribute] = data
        return self.__featureProbabilites
        
    def __getAttributeValues(self,attribute):
        return list(set(self.__dataset[attribute]))
                
    
    def __getCounts(self,tuple1,tuple2):
        return len(dataset[(self.__dataset[tuple1[0]] == tuple1[1]) & (self.__dataset[tuple2[0]] == tuple2[1])])
        
    def __getProbability(self,n,N):
        return n/N

    def __getClassProbabilities(self):
        return self.__classProbabilites
    
    def __getFeatureProbabilities(self):
        return self.__featureProbabilites
    
    def predict(self,featureDictionary):
        probabilitesOfClasses = []
        for classValue in self.__classes:
            probability = 1
            for key,value in featureDictionary.items():
                probability*= self.__featureProbabilites[key][value][classValue]
            
            probability *=self.__classProbabilites[classValue]
        
            probabilitesOfClasses.append(probability)
        
        return self.__classes[np.argmax(probabilitesOfClasses)]
    

