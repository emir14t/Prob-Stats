from Grapher import *
from Models import *
from HypothesisTest import Hypothesis


class DataInterpreter:

    def __init__(self, file, classToUse):
        self.dataSet = pd.read_csv(file)
        self.dataSet = self.dataSet.drop(self.dataSet[self.dataSet["M"] != classToUse].index)
        self.filePath = file
        self.className = classToUse
        self.models = []

    def getDataSet(self):
        return self.dataSet

    def plotBox(self, xName, title=''):
        boxPlot = BoxPlot(self.filePath)
        boxPlot.data = self.dataSet
        boxPlot.render(self.className, xName, title)

    def plotHistogram(self, xName, title=''):
        histogram = Histogram(self.filePath)
        histogram.data = self.dataSet
        histogram.render(xName, 'M', title)

    def __findParameters(self, filePath=''):
        stats = Stats(dataloc=filePath)
        stats.analyze(self.dataSet['IR'])
        stats.export()
        return stats.mean, stats.variance

    def plotNormalProbabilityPlot(self, xName, title=''):
        oldDF = self.dataSet.copy(True)
        mean, var = self.__findParameters(f"{title}v1.csv")
        vals = sorted(self.dataSet[xName].tolist())
        yValues = [(i - 0.375)/(len(vals) + 0.25) for i in range(0, len(vals))]
        errors = [(x - mean) / var for x in vals]
        self.dataSet[xName] = errors
        self.dataSet["y"] = yValues
        normalProbPlot = NormalProbPlot(self.filePath)
        normalProbPlot.data = self.dataSet
        normalProbPlot.render(xName, "y", title=title)
        self.dataSet = oldDF

    def addModel(self, model):
        self.models.append(model)

    def evaluateAllModels(self):
        for model in self.models:
            model.initiate(XData=self.dataSet[model.XColumnName].tolist(),
                           YData=self.dataSet['IR'].tolist())
            # models
            Y = model.model(model.XColumnName)

            YUpper = model.upperBoundModel()
            YLower = model.lowerBoundModel()

            # data
            YPred = [Y(x) for x in self.dataSet[model.XColumnName].tolist()]
            YUpperPred = [YUpper(x) for x in self.dataSet[model.XColumnName].tolist()]
            YLowerPred = [YLower(x) for x in self.dataSet[model.XColumnName].tolist()]

            # set data
            tempDF = self.dataSet.copy(True)
            tempDF['Y_hat'] = YPred
            tempDF['Y_upper'] = YUpperPred
            tempDF['Y_lower'] = YLowerPred

            # plot
            sns = sn
            sns.scatterplot(data=tempDF, y='IR', x=model.XColumnName, color='blue')
            sns.scatterplot(data=tempDF, y='Y_hat', x=model.XColumnName, color='red')
            sns.scatterplot(data=tempDF, y='Y_upper', x=model.XColumnName, color='gray')
            sns.scatterplot(data=tempDF, y='Y_lower', x=model.XColumnName, color='gray')

            plt.title(f"{type(model)} regression comp")
            plt.show()

            model.export(f"{type(model)}.csv")
            model.exportVarianceTable(tempDF["IR"].tolist(), tempDF[model.XColumnName].tolist(), f"{type(model)}-varianceTable.csv")

    def testResidues(self):
        Ydata = self.dataSet['IR'].tolist()
        for model in self.models:
            modelFunction = model.model(model.XColumnName)
            Xdata = self.dataSet[model.XColumnName].tolist()

            # hypothesis  test:
            print(f"for model {type(model)}")
            hypothesis = Hypothesis()
            hypothesis.test(Xdata)
            print("\n====================\n")

            tempDF = self.dataSet.copy(True)
            residues = [Ydata[i] - modelFunction(Xdata[i]) for i in range(len(Xdata))]

            tempDF['residues'] = residues

            sns = sn
            sns.scatterplot(data=tempDF, y='residues', x=model.XColumnName, color='violet')

            plt.title(f"Residues of {type(model)}")
            plt.show()

