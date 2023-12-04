import csv
import math
from scipy.stats import t

class Stats:

    def __init__(self, alphaError=0.05, dataloc=''):
        self.location = dataloc
        self.alpha = alphaError
        self.mean = -1  # uninitialized
        self.variance = 0   # uninitialized
        self.stD = 0    # standard deviation, uninitialized
        self.Q1 = -1    # uninitialized
        self.median = -1    # uninitialized
        self.Q3 = -1    # uninitialized
        self.CI = -1    # confidence interval, uninitialized

    def analyze(self, data):
        self.mean = sum(data) / len(data)
        self.variance = sum([(dataPoint - self.mean) ** 2 for dataPoint in data]) / (len(data) - 1) # n-1 degrees of freedom
        self.stD = math.sqrt(self.variance)
        # confidence interval is calculated as Zscore * std/sqrt(n)
        # confidence level of 95% is given by a Zscore of 1.96
        self.CI = 1.96 * (self.stD / math.sqrt(len(data)))
        sortedData = sorted(data)
        if len(data) % 2 == 0:
            self.median = (sortedData[(len(data) // 2) - 1] + sortedData[len(data)//2]) / 2
            self.Q1 = (sortedData[(len(data) // 4) - 1] + sortedData[len(data) //4]) / 2
            self.Q3 = (sortedData[(3 * len(data)) // 4 - 1] + sortedData[(3 * len(data)) // 4]) / 2
        else:
            self.median = sortedData[len(data) // 2]
            self.Q1 = sortedData[len(data) // 4]
            self.Q3 = sortedData[(3 * len(data)) // 4]


        return self

    def export(self):
        data = [
            ["Quartile 1", "Médiane", "Quartile 3", "Moyenne", "Écart-type", "Intervale de confiance (95%)"],
            [self.Q1, self.median, self.Q3, self.mean, self.stD, self.CI]
        ]
        with open(self.location, 'w') as f:
            writer = csv.writer(f)
            # header
            writer.writerow(data[0])
            # rest of the file
            writer.writerow(data[1])

    def __str__(self):
        return {
            "Quartile 1": self.Q1,
            "Médiane": self.median,
            "Quartile 3": self.Q3,
            "Moyenne": self.mean,
            "Écart-type": self.stD,
            "Intervale de confiance(95 %)" : self.CI
        }.__str__()


class Parameters:
    def __init__(self, XData, YData, alphaError=0.05, dataloc=''):
        self.X = XData
        self.Y = YData
        self.alpha = alphaError
        self.filePath= dataloc
        self.beta0 = -1   #uninitialized
        self.beta1 = -1   #uninitialized
        self.beta0Interval = -1   #uninitialized
        self.beta1Interval = -1   #uninitialized
        self.SXX = -1     #uninitialized
        self.SXY = -1     #uninitialized
        self.SYY = -1     #uninitialized
        self.MSE = -1     #uninitialized
        self.SSE = -1     #uninitialized
        self.SSR = -1     #uninitialized

    """
    This methode allows us to calculate the the confidence interval for the
    ß0 and ß1 parameters for a linear regression, as well as making the appropriate calls
    to finding ß0, ß1 and all the other parameter 
    """
    def evaluate(self):
        if len(self.Y) != len(self.X):
            # we want to crash the code here because it's not worth continuing
            raise Exception('Y and X are not the same size, dumbass!')

        self.findSumsOfSquares()
        self.findBeta0And1()
        self.findSquaredErrors()
        self.evaluateConfidenceInterval()

    def findSumsOfSquares(self):
        xMean = Stats().analyze(self.X).mean
        yMean = Stats().analyze(self.Y).mean

        self.SXX = sum([(xi - xMean) ** 2 for xi in self.X])
        self.SYY = sum([(yi - yMean) ** 2 for yi in self.Y])
        self.SXY = sum([(self.X[i] - xMean) * (self.Y[i] - yMean) for i in range(len(self.X))])

    def findBeta0And1(self):
        self.beta1 = self.SXY / self.SXX
        xMean = Stats().analyze(self.X).mean
        yMean = Stats().analyze(self.Y).mean
        self.beta0 = yMean - (xMean * self.beta1)

    def findSquaredErrors(self):
        self.SSR = (self.SXY * self.SXY) / self.SXX  # did it like this cuz its painfully slow otherwise
        self.SSE = sum([(self.Y[i] - (self.beta0 + (self.beta1 * self.X[i]))) ** 2 for i in range(len(self.X))])
        self.MSE = self.SSE / (len(self.X) - 2)     # MSE ≡ ø^2 = SSE / (n - 2)

    def evaluateConfidenceInterval(self):
        tVal = t.ppf(1 - self.alpha/2, len(self.X) - 2)
        xMean = Stats().analyze(self.X).mean
        self.beta0Interval = tVal * math.sqrt(self.MSE * ((1 / len(self.X)) + (xMean * xMean / self.SXX)))
        self.beta1Interval = tVal * math.sqrt(self.MSE / self.SXX)

    def export(self):
        data = [
            ["ß0", "ß1", "ß0 interval", "ß1 interval", "SXX", "SXY", "SYY", "SSE", "SSR", "MSE"],
            [self.beta0, self.beta1, self.beta0Interval, self.beta1Interval,
             self.SXX, self.SXY, self.SYY, self.SSE, self.SSE, self.MSE]
        ]
        with open(self.filePath, 'w') as f:
            writer = csv.writer(f)
            # header
            writer.writerow(data[0])
            # rest of the file
            writer.writerow(data[1])
            print("Wrote data :)")

    def __str__(self):
        return {
            "ß0": self.beta0,
            "ß1": self.beta1,
            "ß0 interval": self.beta0Interval,
            "ß1 interval": self.beta1Interval,
            "SXX": self.SXX,
            "SXY": self.SXY,
            "SYY": self.SYY,
            "SSE": self.SSE,
            "SSR": self.SSR,
            "MSE": self.MSE
        }.__str__()