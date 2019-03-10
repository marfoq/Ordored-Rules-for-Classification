using CSV
using JuMP
using CPLEX
using DataFrames
using MultiJuMP

include("functions.jl")

dataSet = "titanic"
# dataSet = "b_cancer"
# dataSet = "tic_tac_toe"

dataFolder = "../data/"
resultsFolder = "../res/"

# Create the features tables (or load them if they already exist)
# Note: each line corresponds to an individual, the 1st column of each table contain the class
train, test, Part_A, Part_B, Part_C = createFeatures(dataFolder, dataSet)

# Create the rules (or load them if they already exist)
# Note: each line corresponds to a rule, the first column corresponds to the class
rules = createRules(dataSet, resultsFolder, train,false)

orderedRules = sortRules(dataSet, resultsFolder, train, rules,false)

recall = getPrecision(orderedRules, test)
accuracy = getPrecision(orderedRules, train)

println("Recall: ", recall)
println("Accuracy: ", accuracy)


# Bonus Qustion 4 : Corss-Valdiation

println("======Training on Part_A and Part_B and Evaluating on Part_C")
rules = createRules(dataSet, resultsFolder, vcat(Part_A,Part_B),true)

orderedRules = sortRules(dataSet, resultsFolder, vcat(Part_A,Part_B), rules,true)

recall_1 = getPrecision(orderedRules, Part_C)
accuracy_1 = getPrecision(orderedRules, vcat(Part_A,Part_B))

println("Recall: ", recall_1)
println("Accuracy: ", accuracy_1)

println("=====Training on Part_A and Part_C and Evaluating on Part_B")
rules = createRules(dataSet, resultsFolder, vcat(Part_A,Part_C),true)

orderedRules = sortRules(dataSet, resultsFolder, vcat(Part_A,Part_C), rules,true)

recall_2 = getPrecision(orderedRules, Part_B)
accuracy_2 = getPrecision(orderedRules, vcat(Part_A,Part_C))

println("Recall: ", recall_2)
println("Accuracy: ", accuracy_2)

println("=====Training on Part_B and Part_C and Evaluating on Part_A")
rules = createRules(dataSet, resultsFolder, vcat(Part_B,Part_C),true)

orderedRules = sortRules(dataSet, resultsFolder, vcat(Part_B,Part_C), rules,true)

recall_3 = getPrecision(orderedRules, Part_A)
accuracy_3 = getPrecision(orderedRules, vcat(Part_B,Part_C))

println("Recall: ", recall_3)
println("Accuracy: ", accuracy_3)

println("==== Final Scores =====")
println("Vaildation_Recall: ", (recall_3+recall_2+recall_1)/3)
println("Validation_Accuracy: ", (accuracy_3+accuracy_2+accuracy_1)/3)


