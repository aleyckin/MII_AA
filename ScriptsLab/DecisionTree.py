import numpy as np


class DecisionTree:
    def __init__(self, min_samples, max_depth, data):
        self.min_samples = min_samples
        self.max_depth = max_depth
        self.data = data
        self.trainInit()

    def getMaxMse(self, samples, mse, param):
        sortedSamples = sorted(samples, key=lambda x: self.data[x][param])

        maxMse = -1e18
        maxVal = -1

        for i in range(self.min_samples - 1, len(samples) - self.min_samples):
            if self.data[sortedSamples[i]][param] == self.data[sortedSamples[i + 1]][param]:
                continue

            predict1 = 0
            predict2 = 0

            for j in range(len(samples)):
                ind = sortedSamples[j]
                if j <= i:
                    predict1 += self.data[ind]["absolute_magnitude"]
                else:
                    predict2 += self.data[ind]["absolute_magnitude"]

            predict1 /= i + 1
            predict2 /= len(samples) - i - 1

            mse1 = 0
            mse2 = 0

            for j in range(len(samples)):
                ind = sortedSamples[j]
                if j <= i:
                    diff = (self.data[ind]["absolute_magnitude"] - predict1) ** 2
                    mse1 += diff
                else:
                    diff = (self.data[ind]["absolute_magnitude"] - predict2) ** 2
                    mse2 += diff

            mse1 /= i + 1
            mse2 /= len(samples) - i - 1

            currVal = mse - mse1 - mse2
            if currVal > maxMse:
                maxMse = currVal
                maxVal = (self.data[sortedSamples[i + 1]][param] + self.data[sortedSamples[i]][param]) / 2

        return {
            'val': maxVal,
            'mse': maxMse
        }

    def getSamples(self, val, param, isLeft):
        res = []
        for i in range(len(self.data)):
            if isLeft and self.data[i][param] <= val:
                res.append(i)

            if not isLeft and self.data[i][param] > val:
                res.append(i)

        return res

    def train(self, node, depth):
        node["predict"] = 0
        node["mse"] = 0
        node["isLeaf"] = False

        for i in node["samples"]:
            node["predict"] += self.data[i]["absolute_magnitude"]

        node["predict"] /= len(node["samples"])

        for i in node["samples"]:
            diff = (self.data[i]["absolute_magnitude"] - node["predict"]) ** 2
            node["mse"] += diff

        node["mse"] /= len(node["samples"])

        if depth == self.max_depth:
            return

        if len(node["samples"]) == self.min_samples:
            return

        val1 = self.getMaxMse(node["samples"], node["mse"], "est_diameter_min")
        val2 = self.getMaxMse(node["samples"], node["mse"], "relative_velocity")

        if val1["val"] == -1 and val2["val"] == -1:
            node["isLeaf"] = True
            return

        maxVal = val1
        param = "est_diameter_min"

        if val2["mse"] > val1["mse"]:
            maxVal = val2
            param = "relative_velocity"

        node["val"] = maxVal["val"]
        node["param"] = param

        node["left"] = {}
        node["left"]["samples"] = self.getSamples(node["val"], param, True)
        self.train(node["left"], depth + 1)

        node["right"] = {}
        node["right"]["samples"] = self.getSamples(node["val"], param, False)
        self.train(node["right"], depth + 1)

    def trainInit(self):
        self.tree = {}
        self.tree["samples"] = []

        for i in range(len(self.data)):
            self.tree["samples"].append(i)

        self.train(self.tree, 1)

    def getAns(self, elem):
        current = self.tree
        res = 0

        depth = 1

        while True:
            res = current["predict"]

            if current["isLeaf"]:
                break

            if depth == self.max_depth:
                break

            if len(current["samples"]) <= self.min_samples:
                break

            if elem[current["param"]] <= current["val"]:
                current = current["left"]
            else:
                current = current["right"]

            depth += 1

        return res
