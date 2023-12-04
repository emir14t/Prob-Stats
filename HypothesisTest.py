import math

from scipy.stats import shapiro, t, f


class Hypothesis:

    def __init__(self, alpha=0.05):
        self.isNullHypothesis = False  # H0 starts out as 0
        self.alpha = alpha

    def test(self, dataSet):
        w, pVal = shapiro(dataSet)
        print(w)
        self.isNullHypothesis = pVal > self.alpha
        if self.isNullHypothesis:
            print(f"You can accept H0, needed a p-value more than {self.alpha}, got {pVal}")
        else:
            print(f"You should reject H0, needed a p-value more than {self.alpha}, got {pVal}")

        return self.isNullHypothesis

    def tTest(self, dataY1, dataY2):
        """

        :param dataY1: Array
        :param dataY2: Array
        :return: number, number
        """
        meanY1 = sum(dataY1)/len(dataY1)
        stDevY1 = math.sqrt(sum([(y - meanY1) ** 2 for y in dataY1]) / (len(dataY1) - 1))
        meanY2 = sum(dataY2) / len(dataY2)
        stDevY2 = math.sqrt(sum([(y - meanY2) ** 2 for y in dataY2]) / (len(dataY2) - 1))

        tVal = abs(meanY1 - meanY2) / math.sqrt(((stDevY1 ** 2) / (len(dataY1) - 1)) + ((stDevY2 ** 2) / (len(dataY2) - 1)))

        # two tailed test
        refTVal = t.ppf(1 - self.alpha/2, len(dataY1) + len(dataY2) - 2)

        if tVal > refTVal:
            print(f"Null hypothesis (H0) must be rejected:")
        else:
            print(f"Null hypothesis (H0) can be considered:")
        print(f"got t-value of {tVal} for reference t-value with 95% certainty {refTVal}")

        return tVal, refTVal

    def fTest(self, value, degOfFreedom1, degOfFreedom2):

        fDist = 1 - f.cdf(value, degOfFreedom1, degOfFreedom2)
        if fDist < self.alpha:
            print("Null hypothesis can be considered")
        else:
            print("Null hypothesis can be rejected")

        print(f"got p-value of {fDist}")
