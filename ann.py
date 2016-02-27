#Ziyan Ding    zding

import math
import sys
import random
from array import array
from datetime import datetime

#global variable 
usage = "[Error]Usage: python ann.py <filename> [h <number of hidden nodes> | p <holdout percentage> ]"

random.seed(datetime.now())
#print(datetime.now())

#implement ANN
class ANN:
    def __init__(self, layerNum, inputNum, outputNum, hiddenNodeNum, holdoutPercentage):
        self.layerNum = layerNum
        #self.inputNum = inputNum + 1 #for a0
        self.inputNum = inputNum
        self.outputNum = outputNum
        self.hiddenNodeNum = hiddenNodeNum
        self.holdoutPercentage = holdoutPercentage

        self.w_x_h = [[0 for j in range(hiddenNodeNum)] for i in range(inputNum)]
        #only one output, so 2d array is not needed
        self.w_h_output = [0 for i in range(hiddenNodeNum)]
        
        #initialize weights with small random number
        for j in range(self.hiddenNodeNum):
            for i in range(self.inputNum):
                self.w_x_h[i][j] = random.random()/50
                

        for i in range(self.hiddenNodeNum):
            self.w_h_output[i] =  random.random()/50

    #sigmoid function set p = 1
    def sigmoidFunction(self, x):
        #print('x is')
        #print(x)
        if x > 700 or x < -700:
            return 0.0
        else:
            result = 1/(1 + math.exp(-x)) 
            return result

    #given a node, predict the classification 
    def predict(self, x1, x2):
        hNode = [0 for j in range(self.hiddenNodeNum)]
        par = 0.0
        for j in range(self.hiddenNodeNum):
            hNode[j] = self.w_x_h[0][j]*x1 + self.w_x_h[1][j]*x2
            par += self.w_h_output[j]*self.sigmoidFunction(hNode[j])

        output = self.sigmoidFunction(par)
        if output > 0.5:
            output = 1
        else:
            output = 0
        return output

    def sigmoidDerivative(self, x):
        result = self.sigmoidFunction(x)*(1-self.sigmoidFunction(x))
        return result

    def backPropLearning(self, info):
        iteration = 1
        while(iteration < 50):
            #n = int(len(info)*(1-self.holdoutPercentage))
            n = int(len(info)*self.holdoutPercentage)
            k = int(len(info) - 1)
            iteration += 1
            for data in info[n:k+1]:
                alpha = 0.1
                inputX = [0.0 for i in range(self.inputNum)]
                in_x_h = [0.0 for i in range(self.hiddenNodeNum)]
                hidden = [0.0 for i in range(self.hiddenNodeNum)]
                output = 0.0

                in_h_output = 0.0
                #holds the error from the output layer
                delta = 0.0 
                #holds the error from the hidden layer
                delta_h = [0.0 for i in range(self.hiddenNodeNum)]
                inputX[0] = data.x1
                inputX[1] = data.x2
                
                #propagate the input forward to compute the output
                #layer = 2, get hidden nodes
                for j in range(self.hiddenNodeNum):
                    #in_x_h[j] = w_x_h[0][j] + w_x_h[1][j] * x1 + w_x_h[2][j] * x2
                    in_x_h[j] = self.w_x_h[0][j] * inputX[0]+ self.w_x_h[1][j] * inputX[1]
                    hidden[j] = self.sigmoidFunction(in_x_h[j])

                #layer = 3, get output node
                for i in range(self.hiddenNodeNum):
                    in_h_output += self.w_h_output[i] * hidden[i]
                output = self.sigmoidFunction(in_h_output)

                #propagate deltas backward from output layer to input layer
                delta = self.sigmoidDerivative(in_h_output)*(data.label - output)
                #layer = 2
                for i in range(self.hiddenNodeNum):
                    delta_h[i] = self.sigmoidDerivative(in_x_h[i])*(self.w_h_output[i] * delta)
                
                #update weights between hidden and output layer
                for i in range(self.hiddenNodeNum):
                    self.w_h_output[i] = self.w_h_output[i] + alpha*hidden[i]*delta

                #update weights between input and hidden layer
                for i in range(self.inputNum):
                    for j in range(self.hiddenNodeNum):
                        self.w_x_h[i][j] = self.w_x_h[i][j] + alpha*inputX[i]*delta_h[j]

    #return error rate
    def trainning(self, info):
        errorNum = 0.0
        errorRate = 0.0
        self.backPropLearning(info)
        lowerBound = 0
        upperBound = int(len(info)*self.holdoutPercentage - 1)
        totalNum = upperBound - lowerBound
        totalNum = upperBound - lowerBound
        for i in range(lowerBound, upperBound):
            predict_output = self.predict(info[i].x1, info[i].x2)
            if predict_output != info[i].label:
                errorNum += 1
        #print('error ' + str(errorNum))
        errorRate = errorNum/totalNum
        print('\nError Rate is ' + str(errorRate) + '\n')
        return errorRate

#define Data to store data
class Data:
    def __init__(self, x1 = 0.0, x2 = 0.0, label = 0.0):
        self.x1 = x1
        self.x2 = x2
        self.label = label
        


#reads given file line by line
def readFile(filename, info):
    f = open(filename, "r")
    fileContent = []   
    for line in f:
        fileContent = line.split()
        #point = Point(fileContent[0], fileContent[1])
        #data = Data(point, fileContent[2])
        data = Data(float(fileContent[0]), float(fileContent[1]), float(fileContent[2]))
        info.append(data)
    

    f.close()
    
def main():
    hiddenNodeNum = 5
    holdoutPercentage = 0.2
    info = []
    #gets command argument
    command = str(sys.argv)
    commandNum = len(sys.argv)
    if commandNum == 2:
        #using default values for hiddenNodeNum and holdoutPercentage
        filename = str(sys.argv[1])       
        readFile(filename, info)
    elif commandNum == 4:
        filename = str(sys.argv[1])
        readFile(filename, info)
        if str(sys.argv[2]) == "p":
            holdoutPercentage = float(str(sys.argv[3]))
        elif str(sys.argv[2]) == "h":
            hiddenNodeNum = int(str(sys.argv[3]))
        else:
            print(usage)
    elif commandNum == 6:
        filename = str(sys.argv[1])
        readFile(filename, info)
        if str(sys.argv[2]) == "p":
            holdoutPercentage = float(str(sys.argv[3]))
            if str(sys.argv[4]) == "h":
                hiddenNodeNum = int(str(sys.argv[5]))
            else:
                print(usage)
        elif str(sys.argv[2]) == "h":
            hiddenNodeNum = int(str(sys.argv[3]))
            if str(sys.argv[4]) == "p":
                holdoutPercentage = float(str(sys.argv[5]))
            else:
                print(usage)
    else:
        print(usage)

    newANN = ANN(3, 2, 1, hiddenNodeNum, holdoutPercentage)
    newANN.trainning(info)      
      

if __name__ == '__main__':
    main()
        
