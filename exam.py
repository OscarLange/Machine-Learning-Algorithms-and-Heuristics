import math
import matplotlib.pyplot as plt
import numpy as np
#---------------------------------- General Math ----------------------------------
def average(inputlist):
    result = 0
    print("Calculating Average-----------------------------------------")
    print("(", end="", sep="")
    for value in inputlist:
        if(value < 0):
            print(value, end="", sep="")
        else:
            print("+",value, end="", sep="")
        result = result + value
    result = result / len(inputlist)
    print(")/", len(inputlist)," = ",result, sep="")
    return result

def sd(inputlist):
    avg = average(inputlist)
    result = 0
    print("Calculating Standart Deviation-----------------------------------------")
    print("(", end="", sep="")
    for value in inputlist:
        print("(", value, "-", avg, ")²+", end="", sep="")
        result = result + ((value - avg)**2)
    result = result / (len(inputlist))
    print(")/", len(inputlist)-1," = ",result, sep="")
    print("sqrt(", result, ")=", math.sqrt(result), sep="")
    return math.sqrt(result)

def cov(list1, list2):
    avg1 = average(list1)
    avg2 = average(list2)
    list1 = list(map(lambda x: x-avg1, list1))
    list2 = list(map(lambda x: x-avg2, list2))
    cov = sum(list(map(lambda x, y: x*y, list1, list2)))/len(list1)
    print("Cov=", cov, sep="")
    return cov

#cov([5,8,10,1],[10,9,1,6])
#sd([-0.31,-0.28,0.02,-0.11,-0.2,-0.06,-0.11,-0.05,-0.15,-0.13])
#---------------------------------- Statistical Tests ----------------------------------

def ttest(inputlist):
    avg = average(inputlist)
    std = sd(inputlist)
    sqrtK = math.sqrt(len(inputlist))
    result = avg/ (std/sqrtK)
    print("Calculating T Test-----------------------------------------")
    print(avg, "/(",std, "/", sqrtK, "=", result, sep="")

def ttestDiff(baselist, improvedlist):
    output = []
    print("Calculating Difference-----------------------------------------")
    print("[", end="", sep="")
    for index in range(len(baselist)):
        output.append(baselist[index] - improvedlist[index])
        print(baselist[index] - improvedlist[index], ",", end="", sep="")
    print("]",sep="")
    ttest(output)

#ttestDiff([0.71,0.69,0.83,0.72,0.91,0.74,0.72,0.83,0.67,0.79,0.82,0.72,0.71,0.84,0.8],[0.69,0.67,0.74,0.63,0.89,0.7,0.78,0.86,0.66,0.81,0.78,0.72,0.78,0.85,0.78])
#ttest([0,-0.06,0.05,-0.16,-0.11,0.04,0.11,-0.01,0.01,-0.02])

#---------------------------------- Decision Trees ----------------------------------

def informationGain(rootmatrix, newmatrix):
    print("CALCULATING ROOT-----------------------------------------")
    result1 = info(rootmatrix)
    print("CALCULATING Base-----------------------------------------")
    result2 = info(newmatrix)
    print("CALCULATING Information Gain-----------------------------------------")
    print(result1, "-", result2, "=", result1 - result2)
    print("CALCULATING Information Gain Ratio-----------------------------------------")
    gainRatio(newmatrix, result1-result2)

def gainRatio(inputmatrix, informationGain):
    allSums = [[]]
    for rows in inputmatrix:
        sum = 0
        for value in rows:
            sum += value
        allSums[0].append(sum)
    print(informationGain, "/ info([", end="", sep="")
    for index in range(len(allSums[0])):
        if(index < len(allSums[0]) - 1):
            print(allSums[0][index], ",", end="", sep="")
        else:
            print(allSums[0][index], "]=", end="", sep="")
    information = info(allSums)
    print(informationGain, "/", information, "=", informationGain/information, sep="")

def info(inputmatrix):
    allSums = []
    majorSum = 0
    for rows in inputmatrix:
        sum = 0
        for value in rows:
            sum += value
            majorSum += value
        allSums.append(sum)
    
    result = 0
    for index in range(len(inputmatrix)):
        print("(" , allSums[index], "/" , majorSum, ")*entropy(", end="", sep="")
        for index2 in range(len(inputmatrix[index])):
            if(index2 < len(inputmatrix[index]) - 1):
                print(inputmatrix[index][index2], "/", allSums[index], ",", end="", sep="")
            else:
                print(inputmatrix[index][index2], "/", allSums[index], end="", sep="")
            
            if(index < len(inputmatrix)-1):
                print(")+", end="", sep="")
            else:
                print(")")
        
        result = result + (entropy(inputmatrix, allSums, index) * (allSums[index]/majorSum))
        print("Result", result)
    return result

def entropy(inputmatrix, allSums, index):
    result = 0
    for value in inputmatrix[index]:
        if(value == 0):
            return 0
        tmp = math.log2(value/allSums[index]) * value/allSums[index]
        result = result - tmp
    return result

#informationGain([[5,5]], [[1,0],[1,3],[4,2]])

#---------------------------------- Evaluation ----------------------------------

def decimalToPercent(inputlist):
    for index in range(len(inputlist)):
        inputlist[index] = str(round(inputlist[index]*100, 4))+"%"

    return inputlist

def gainAndLiftCurve(inputlist, percentSteps):
    stepSize = len(inputlist)*percentSteps
    positiveInstances = []
    i = 0
    majorSum = 0
    while i < len(inputlist):
        j = 0
        sum = 0
        while j < stepSize:
            if(inputlist[i]):
                sum += 1
                majorSum += 1
            j = j+1
            i += 1
        positiveInstances.append(sum)
    current = 0
    for index in range(len(positiveInstances)):
        current += ((positiveInstances[index] / majorSum)) 
        positiveInstances[index] = current
        print(positiveInstances[index],"|", end="", sep="")
    
    steps = []
    i = percentSteps
    while i <= 1:
        steps.append(i)
        i += percentSteps

    lift = []
    for index in range(len(positiveInstances)):
        lift.append(positiveInstances[index]/steps[index])
    
    steps = decimalToPercent(steps)
    positiveInstances = decimalToPercent(positiveInstances)

    fig, (ax1, ax2) = plt.subplots(2)
    ax1.plot(steps, positiveInstances, '--bo')
    ax1.set_xlabel('Percentage of the Data Set')
    ax1.set_ylabel('Percentage of positive instances')
    ax1.set_xticks(steps)
    for i,j in zip(steps,positiveInstances):
        ax1.annotate(str(j),xy=(i,j))

    ax2.plot(steps, lift, '--bo')
    ax2.set_xlabel('Percentage of the Data Set')
    ax2.set_ylabel('Lift')
    ax2.set_xticks(steps)
    for i,j in zip(steps,lift):
        ax2.annotate(str(j),xy=(i,j))

    plt.show()

def roc(inputmatrix, cutOff):
    prediction = []
    posVal = 0
    negVal = 0
    for row in inputmatrix:
        if(row[1]):
            prediction.append(True)
            #positive value
            posVal += 1
        else:
            #negative value
            negVal += 1
            prediction.append(False)
    cur_tpr = 0
    cur_fpr = 0
    tpr = []
    fpr = []
    tpr.append(cur_tpr)
    fpr.append(cur_fpr)
    for value in prediction:
        if(value):
            cur_tpr += 1
            tpr.append(cur_tpr/posVal)
            if(cur_fpr > 0):
                fpr.append(cur_fpr/negVal)
            else:
                fpr.append(cur_fpr)
        else:
            cur_fpr += 1
            if(cur_tpr > 0):
                tpr.append(cur_tpr/posVal)
            else:
                tpr.append(cur_tpr)
            fpr.append(cur_fpr/negVal)

    plt.plot(fpr, tpr)
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.title('ROC')
    for i,j in zip(fpr,tpr):
        print(i, "|", j)
    plt.show()

#roc([[0.91,True],[0.89,False],[0.88,False],[0.81,True],[0.79,False],[0.77,True],[0.68,True],[0.54,False],[0.25,False],[0.17,False]], 0.76)

#gainAndLiftCurve([True,True,True,True,True,False,True,False,True,True,False,False,True,False,True,False,False,False,False,False], 0.1)

#---------------------------------- Clustering and Ensemble ----------------------------------

def euclideanDistance(x1, y1, x2, y2):
    return math.sqrt((x1-x2)**2 + (y1-y2)**2)

def sortPoints(valuematrix, centroidmatrix):
    sortedPoints = []
    for centroid in centroidmatrix:
        sortedPoints.append([])
    index = 1
    for row in valuematrix:
        maxDist = float("inf")
        currCentroid = 0
        for centroidIndex in range(len(centroidmatrix)):
            centroid = centroidmatrix[centroidIndex]
            tmpDist = euclideanDistance(row[0], row[1], centroid[0], centroid[1])
            print("Point=(" , row[0], "|" , row[1], ") and centroid=(",centroid[0], "|", centroid[1],")have distance = ",tmpDist ,sep="")
            if(tmpDist < maxDist):
                maxDist = tmpDist
                currCentroid = centroidIndex    
        print("-----------------------------------Point_", index, "--------------------------------------------")
        print("Point=(" , row[0], "|" , row[1], ") is closest to centroid_",currCentroid ,"=(",centroidmatrix[currCentroid][0], "|", centroidmatrix[currCentroid][1], ")", sep="")
        print("-------------------------------------------------------------------------------")
        sortedPoints[currCentroid].append((row[0], row[1]))
        index += 1
    return sortedPoints

def newcentroids(valuematrix, centroidmatrix):
    print("---------------------------------Sorting Points----------------------------------------------")
    sortedPoints = sortPoints(valuematrix, centroidmatrix)
    print("---------------------------------Calculating Centroids----------------------------------------------")
    centroidmatrix = []
    for row in sortedPoints:
        total = [0,0]
        for valueTuple in row:
            total[0] += valueTuple[0]
            total[1] += valueTuple[1]
        if(total[0] != 0):
            total[0] /= len(row)
        if(total[1] != 0):
            total[1] /= len(row)
        print("New centroid=(",total[0], "|", total[1],")",sep="")
        centroidmatrix.append(total)

def pdfEM(x, my, sd):
    return (1/(sd* math.sqrt(2*math.pi)))*math.exp(-(((x-my)**2)/(2*sd**2)))

def probEMX(x, myA, sdA, pA, myB, sdB, pB ):
    pdfA = pdfEM(x, myA, sdA)
    print("fA=",pdfA,sep="",end="")
    pdfB = pdfEM(x, myB, sdB)
    print("|fB=",pdfB,sep="")
    prX = pdfA * pA + pdfB * pB
    print("Pr[",x,"]=",prX,sep="")
    prAX = (pdfA*pA)/prX
    print("|Pr[A|",x,"]=",prAX,"|Pr[B|",x,"]=",1-prAX,sep="")
    return prAX

def probEM(x_values, myA, sdA, pA, myB, sdB, pB):
    matrix = []
    for value in x_values:
        print("-----------------------------------Point_", value, "--------------------------------------------")
        matrix.append(probEMX(value, myA, sdA, pA, myB, sdB, pB))
        print("------------------------------------------------------------------------------------------")
    return matrix

def updateMy(weights, x_values):
    avg_wA = sum(weights)
    print("Sum of weights A:",avg_wA,sep="")
    avg_wXA = sum(list(map(lambda x, y: x*y, weights, x_values)))
    print("Sum of weighted x with pa:",avg_wXA,sep="")
    myA = avg_wXA/avg_wA
    print("New my of A:",myA,sep="")

    weightsB = list(map(lambda x: 1-x, weights))
    avg_wB = sum(weightsB)
    print("Sum of weights B:",avg_wB,sep="")
    avg_wXB = sum(list(map(lambda x, y: x*y, weightsB, x_values)))
    print("Sum of weighted x with pa:",avg_wXB,sep="")
    myB = avg_wXB/avg_wB
    print("New my of B:",myB,sep="")
    return (myA,myB)

def updatesd(weights, x_values, myA, myB):
    avg_wA = sum(weights)
    print("Sum of weights A:",avg_wA,sep="")
    avg_wXA = sum(list(map(lambda w, x: w*(x-myA)**2, weights, x_values)))
    print("Sum of weighted x with pa:",avg_wXA,sep="")
    sdA = math.sqrt(avg_wXA/avg_wA)
    print("New sd of A:",sdA,sep="")

    weightsB = list(map(lambda w: 1-w, weights))
    avg_wB = sum(weightsB)
    print("Sum of weights B:",avg_wB,sep="")
    avg_wXB = sum(list(map(lambda w, x: w*(x-myB)**2, weightsB, x_values)))
    print("Sum of weighted x with pa:",avg_wXB,sep="")
    sdB = math.sqrt(avg_wXB/avg_wB)
    print("New sd of B:",sdB,sep="")
    return (sdA,sdB)

def updateProb(weights):
    weightsB = list(map(lambda w: 1-w, weights))
    pA = sum(weights) / (sum(weights) + sum(weightsB))
    print("New pA:",pA,",New pB:", 1-pA, sep="")
    return [pA, 1-pA]


def EM(x_values, myA, sdA, pA, myB, sdB, pB, rounds):
    i = 0
    while(i < rounds):
        print("--------------------------------Expectation Step------------------------------------------")
        weights = probEM(x_values, myA, sdA, pA, myB, sdB, pB)
        print("--------------------------------Maximization Step-----------------------------------------")
        new_my = updateMy(weights, x_values)
        myA = new_my[0]
        myB = new_my[1]
        new_sd = updatesd(weights,x_values, myA, myB)
        sdA = new_sd[0]
        sdB = new_sd[1]
        new_prob = updateProb(weights)
        pA = new_prob[0]
        pB = new_prob[1]
        i += 1

def determinante3x3(matrix):
    print("(" , matrix[0][0], ")*(" , matrix[1][1], "*", matrix[2][2], "-", matrix[2][1], "*", matrix[1][2], ")-", sep="")
    print("(" , matrix[0][1], ")*(" , matrix[1][0], "*", matrix[2][2], "-", matrix[2][1], "*", matrix[0][2], ")+", sep="")
    print("(" , matrix[0][2], ")*(" , matrix[1][0], "*", matrix[1][2], "-", matrix[1][1], "*", matrix[0][2], ")",end="", sep="")

def svd(U, S, V, c):
    print(np.dot(np.dot(U,S), V) + c)

def correlationCoefficient(r1, r2):
    result = cov(r1,r2)/(sd(r1)*sd(r2))
    print("Weighted Correlation=", result, sep="")
    return result

def weightedCorrelations(r1, r2, sig, div):
    corR1 = []
    corR2 = []
    correlatedItems = 0
    for index in range(len(r1)):
        if(r1[index] != None and r2[index] !=None):
            corR1.append(r1[index])
            corR2.append(r2[index])
            correlatedItems += 1
    significanceWeights = 0
    if(correlatedItems >= sig):
        significanceWeights = 1
    else:
        significanceWeights = correlatedItems/div
    print("Significance weight=", significanceWeights, sep="")
    corCoeff = correlationCoefficient(corR1, corR2)
    weight = significanceWeights * corCoeff
    print("Final weight=", weight, sep="")

#weightedCorrelations([9,7,2,9,5,None,8,8,8,6],[10,10,None,9,1,3,None,6,1,6], 7, 7)

#determinante3x3([["10/3-\u03BB","0","2"],["0","-\u03BB","0"],["2","0","10/3-\u03BB"]])
#svd([0.35,0],[[5.22,0],[0,3.12]],[[-0.13], [0.34]], 2)