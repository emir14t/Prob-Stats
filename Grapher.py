from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd


class Grapher (ABC):

    def __init__(self, dataFilePath):
        self.plot = plt
        self.sns = sn
        self.data = pd.read_csv(dataFilePath, delimiter=',')

    @abstractmethod
    def render(self, axisX, axisY, title='', color=''):
        pass

    def plotGraph(self, axisX='', axisY='', title=''):
        self.plot.xlabel = axisX
        self.plot.ylabel = axisY
        self.plot.title(title)
        self.plot.show()


class BoxPlot (Grapher):

    def __init__(self, dataFilePath):
        super().__init__(dataFilePath)

    def render(self, axisX, axisY, title='', color=''):
        self.sns.boxplot(x=axisX, y=axisY, data=self.data)
        self.plotGraph(axisX, axisY, title)


class Histogram (Grapher):

    def __init__(self, dataFilePath):
        super().__init__(dataFilePath)

    def render(self, axisX, axisY, title='', color=''):
        self.sns.displot(data=self.data, x=axisX, bins=10, kde=True)
        self.plotGraph(axisX, "count", title)


class NormalProbPlot (Grapher):

    def __init__(self, dataFilePath):
        super().__init__(dataFilePath)

    def render(self, axisX, axisY, title='', color=''):
        self.sns.scatterplot(x=axisX, y=axisY, data=self.data)
        self.plotGraph(axisX, axisY, title)

