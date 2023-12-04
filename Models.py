import csv
import math
from abc import ABC, abstractmethod
from StatsVoodoo import Stats, Parameters


class Model(ABC):

    def __init__(self):
        self.stats = Stats()
        self.parameters = None  # uninitialized
        self.XColumnName = ":)"
        self.beta0 = -1  # uninitialized
        self.beta1 = -1  # uninitialized
        self.beta0Bound = -1    # uninitialized
        self.beta1Bound = -1    # uninitialized
        self.beta0Interval = [-1, -1]   # uninitialized
        self.beta1Interval = [-1, -1]   # uninitialized
        self.SSE = -1   # uninitialized
        self.F_0 = -1   # uninitialized

    def initiate(self, XData, YData):
        self.parameters = Parameters(XData=XData, YData=YData)
        self.parameters.evaluate()
        self.beta0Bound = self.parameters.beta0Interval
        self.beta1Bound = self.parameters.beta1Interval

    @abstractmethod
    def model(self, columnName):
        if columnName != self.XColumnName:
            raise Exception('Wrong column name bozo!')

    @abstractmethod
    def upperBoundModel(self):
        pass

    @abstractmethod
    def lowerBoundModel(self):
        pass

    def export(self, filePath):
        data = [
            ["beta0", "beta1", "beta0 interval (+/-)", "beta1 interval (+/-)"],
            [self.beta0, self.beta1, (self.beta0Interval[1] - self.beta0Interval[0]) / 2, (self.beta1Interval[1] - self.beta1Interval[0]) / 2]
        ]
        with open(filePath, 'w') as f:
            writer = csv.writer(f)
            # header
            writer.writerow(data[0])
            # rest of the file
            writer.writerow(data[1])

    def exportVarianceTable(self, targetYVals, XVals, filePath=''):
        """
         __________________________________________________________________________________________________________
        |source de variation | somme des carrés | nb de deg de liberté | moyenne des carrés | F_0                  |
        |--------------------|------------------|----------------------|-------------------|----------------------|
        |regression          |SSR (déjà calculé)|        1             | somme/deg. de lib.| regression / résidus |
        |--------------------|------------------|----------------------|-------------------|----------------------|
        |résidus             |SSE (à calculer)  |       n-2            | somme/deg. de lib | Null                 |
        |--------------------|------------------|----------------------|-------------------|----------------------|
        |total               | SSR + SSE        |       n-1            | Null              | Null                 |
        |--------------------|------------------|----------------------|-------------------|----------------------|

        """
        modelFunction = self.model(self.XColumnName)
        self.SSE = sum([(targetYVals[i] - modelFunction(XVals[i])) ** 2 for i in range(len(XVals))])
        averageY = sum(targetYVals) / len(targetYVals)
        SSR = sum([(modelFunction(x) - averageY) ** 2 for x in XVals])
        self.F_0 = SSR / (self.SSE / (len(XVals) - 2))
        data = [
            ["source de variation", "somme des carrés", "nb de deg de liberté", "moyenne des carrés", "F_0"],
            ["regression", SSR, 1, SSR, self.F_0],
            ["résidus", self.SSE, (len(XVals) - 2), self.SSE / (len(XVals) - 2), None],
            ["total", SSR + self.SSE, (len(XVals) - 1), None, None]
        ]

        with open(filePath, 'w') as f:
            writer = csv.writer(f)
            # header
            writer.writerow(data[0])
            # rest of the file
            for i in range(1, len(data)):
                writer.writerow(data[i])


    def __gt__(self, other):
        # this is to automate sorting the models eventually
        pass


class ModelType1(Model, ABC):
    def __init__(self):
        super().__init__()

    def model(self, columnName):
        """
            Y = ß0 + ß1 * X
            regression gives us directly ß0 and ß1, so nothing to do
        """
        super().model(columnName)
        self.beta0 = self.parameters.beta0
        self.beta1 = self.parameters.beta1

        self.beta0Interval = [self.beta0 - self.beta0Bound, self.beta0 + self.beta0Bound]
        self.beta1Interval = [self.beta1 - self.beta1Bound, self.beta1 + self.beta1Bound]

        def Y(X):
            return self.beta0 + (self.beta1 * X)

        return Y

    def upperBoundModel(self):
        """
            Y = ß0 + ß1 * X
            Y = [ß0 + ß0int] + [ß1 + ß1int] * X
        """

        return lambda X: self.beta0Interval[1] + self.beta1Interval[1] * X

    def lowerBoundModel(self):
        """
            Y = ß0 + ß1 * X
            Y = [ß0 - ß0int] + [ß1 - ß1int] * X
        """

        return lambda X: self.beta0Interval[0] + self.beta1Interval[0] * X


class ModelType2(Model, ABC):
    def __init__(self):
        super().__init__()

    def initiate(self, XData, YData):
        lnYData = [math.log(abs(y), math.e) for y in YData]
        lnXData = [math.log(abs(x), math.e) for x in XData]
        super().initiate(lnXData, lnYData)

    def model(self, columnName):
        """
            Y = ß0 * X^ß1 * e^𝛆
            lnY = lnß0 + (lnX) * ß1

            we obtain B0 = lnß0 and B1 = ß1 through the regression
            ß0 = e^B0, ß1 = B1

            Y = e^lnß0 * X^ß1 * e^𝛆

        """

        super().model(columnName)
        self.beta0 = math.e ** self.parameters.beta0
        self.beta1 = self.parameters.beta1

        self.beta0Interval = [self.beta0 / self.beta0Bound, self.beta0 * self.beta0Bound]
        self.beta1Interval = [self.beta1 - self.beta1Bound, (self.beta1 + self.beta1Bound)]

        def Y(X):
            return self.beta0 * (X ** self.beta1)

        return Y

    def upperBoundModel(self):
        """
        Y = ß0 * X^ß1 * e^𝛆
        lnY = lnß0 + (lnX) * ß1
        lnY = [lnß0 + ln(ß0 interval)] + (lnX) * [ß1 + ß1int]

        Y = (ß0 * ß0int) * X ^ (ß1 + ß1int)
        """

        return lambda X: self.beta0Interval[1] * (X ** self.beta1Interval[1])

    def lowerBoundModel(self):
        """
            Y = ß0 * X^ß1 * e^𝛆
            lnY = lnß0 + (lnX) * ß1
            lnY = [lnß0 - ln(ß0 interval)] + (lnX) * [ß1 - ln(ß1)]

            Y = (ß0 / ß0int) * X ^ (ß1 - ß1int)
        """
        return lambda X: self.beta0Interval[0] * (X ** self.beta1Interval[0])


class ModelType3(Model, ABC):

    def __init__(self):
        super().__init__()

    def initiate(self, XData, YData):
        lnYData = [math.log(abs(y), math.e) for y in YData]
        super().initiate(XData, lnYData)

    def model(self, columnName):
        """
            Y = ß0 * e^(ß1*X + 𝛆)
            lnY = lnß0 + (ß1 * X) + 𝛆
            we obtain B0 = lnß0 and B1 = ß1 through the regression

            ß0 = e^B0, ß1 = B1

        """

        super().model(columnName)
        self.beta0 = math.e ** self.parameters.beta0
        self.beta1 = self.parameters.beta1

        self.beta0Interval = [self.beta0 / self.beta0Bound, self.beta0 * self.beta0Bound]
        self.beta1Interval = [self.beta1 - self.beta1Bound, self.beta1 + self.beta1Bound]

        def Y(X):
            return self.beta0 * (math.e ** (self.beta1 * X))

        return Y

    def upperBoundModel(self):
        """
        lnY = lnß0 + (ß1 * X)
        lnY = [lnß0 + ln(ß0int)] + (ß1 + ß1int) * X

        Y = (ß0 * ß1int) * e^(ß1 + ß1int) * X)
        """
        return lambda X: self.beta0Interval[1] * (math.e ** (self.beta1Interval[1] * X))

    def lowerBoundModel(self):
        """
            lnY = lnß0 + (ß1 * X)
            lnY = [lnß0 - ln(ß0int)] + (ß1 - ß1int) * X

            Y = (ß0 / ß1int) * e^((ß1 - ß1int) * X)
        """
        return lambda X: self.beta0Interval[0] * (math.e ** (self.beta1Interval[0] * X))


# Model implementations

class Model1(ModelType1):
    def __init__(self):
        super().__init__()
        self.XColumnName = "V"  # safety precaution


class Model4(ModelType1):
    def __init__(self):
        super().__init__()
        self.XColumnName = "T"  # safety precaution


class Model2(ModelType2):
    def __init__(self):
        super().__init__()
        self.XColumnName = "V"  # safety precaution


class Model5(ModelType2):
    def __init__(self):
        super().__init__()
        self.XColumnName = "T"  # safety precaution


class Model3(ModelType3):
    def __init__(self):
        super().__init__()
        self.XColumnName = "V"  # safety precaution


class Model6(ModelType3):
    def __init__(self):
        super().__init__()
        self.XColumnName = "T"  # safety precaution
