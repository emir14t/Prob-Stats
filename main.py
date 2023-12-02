import Grapher
from Models import *
import pandas as pd
from DataInterpreter import DataInterpreter as di
import numpy as np
from HypothesisTest import Hypothesis

"""
    Macro (change me)
"""
MATRICULE = 2194964


def charger(matricule):
    np.random.seed(matricule)
    myData = pd.read_csv("DevoirD_A23.csv", sep=',')
    myData = myData.sample(n=205, replace=False, random_state=matricule)
    myData.to_csv("DevoirD_A23.csv", index=False)

    return myData


if __name__ == '__main__':
    charger(MATRICULE)

    # Partie 1 a)
    dataP1 = di('DevoirD_A23.csv', 0)
    dataP1.dataSet = pd.read_csv('DevoirD_A23.csv')
    dataP1.plotBox('IR', 'Box Plot all IR')
    dataP1.plotHistogram('IR', 'Distribution of IR in all materials')
    dataP1.plotNormalProbabilityPlot('IR', 'Normal plot for IR values in the entire set')
    test = Hypothesis()
    test.test(dataP1.dataSet['T'].tolist())
    # dataP1.plotBox('IR', 'Box Plot for material m0 and m1')
    graph = Grapher.BoxPlot('DevoirD_A23.csv')
    graph.data = dataP1.dataSet
    graph.render(axisX='M', axisY='IR')

    # Partie 1 b)

    dataM0 = di('DevoirD_A23.csv', 0)
    dataM0.plotBox('IR', 'Box Plot for material 0')
    dataM0.plotHistogram('IR', 'Distribution of IR in material 0')
    dataM0.plotNormalProbabilityPlot('IR', 'Normal plot for IR values in the m0 set')

    dataM1 = di('DevoirD_A23.csv', 1)
    dataM1.plotBox('IR', 'Box Plot for material 1')
    dataM1.plotHistogram('IR', 'Distribution of IR in material 1')
    dataM1.plotNormalProbabilityPlot('IR', 'Normal plot for IR values in the m1 set')

    # hypothesis test
    dataY0 = dataM0.dataSet['IR'].tolist()
    dataY1 = dataM1.dataSet['IR'].tolist()
    test.tTest(dataY0, dataY1)

    # Partie 2 c)

    # we choose to work with m0 (can also work with m1)
    models = [Model1(), Model2(), Model3(), Model4(), Model5(), Model6()]
    for model in models:
        dataM0.addModel(model)

    dataM0.evaluateAllModels()
    dataM0.testResidues()
