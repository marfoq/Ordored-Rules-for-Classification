# Tolerance
epsilon = 0.0001

# Take in table "data" the column named "header" and create in table "features" columns which correspond to its binarization according to the values in "intervals"
#
# Attributes:
# - header: column of table "data" that will be binarized
# - intervals: array of values which delimits the binarization (ex : [2, 4, 6, 8] will lead to 3 columns respectively equal to 1 if the value of column "header" is in [2, 3], [4, 5] and [6, 7])
#
# Example:
#  createColumns(:Age, [1, 17, 50, Inf], data, features) will create 3 binary columns in features named "Age1-16", "Age17-49", "Age50-Inf"
function createColumns(header::Symbol, intervals, data::DataFrames.DataFrame, features::DataFrames.DataFrame)
    for i in 1:size(intervals, 1) - 1
        lb = intervals[i]
        ub = intervals[i+1]
        features[Symbol(header, lb, "-", (ub-1))] = ifelse.((data[header] .>= lb) .& (data[header] .< ub), 1, 0) 
        #features[Symbol(header, lb, "-", (ub-1))] = ifelse.(data[header] .< ub, 1, 0) 
    end
end



# Create the train and test tables related to a data set
#
# Note 1: the input file name is: dataFolder/dataSet.csv
# Note 2: the first column of the output tables must correspond to the class of each individual
function createFeatures(dataFolder::String, dataSet::String)

    rawDataPath = dataFolder * dataSet * ".csv"

    if !isfile(rawDataPath)
        println("Error in createFeatures: Input file not found: ", rawDataPath)
        return
    end

    rawData = CSV.read(rawDataPath,  header=true)

    
    trainDataPath = dataFolder * dataSet * "_train.csv"
    testDataPath = dataFolder * dataSet * "_test.csv"

    partA_DataPath = dataFolder * dataSet * "_partA.csv"
    partB_DataPath = dataFolder * dataSet * "_partB.csv"
    partC_DataPath = dataFolder * dataSet * "_partC.csv"

    # If the train or the test file do not exist
    if !isfile(trainDataPath) || !isfile(testDataPath)

        println("=== Creating the features")

        # Create the table that will contain the features
        features = DataFrames.DataFrame()
        
        # Create the features of the titanic data set
        if dataSet == "titanic"
		
            features[Symbol("Label")] = rawData[:Survived]

            # Sex
            features[Symbol("Sex")] = ifelse.(rawData[:Sex] .== "female", 1, 0) 

	     	# Pclass
	     	features[Symbol("Pclass_1")] =  ifelse.( rawData[:Pclass] .== 1, 1, 0)
	     	features[Symbol("Pclass_2")] =  ifelse.( rawData[:Pclass] .== 2, 1, 0)
	     	features[Symbol("Pclass_3")] =  ifelse.( rawData[:Pclass] .== 3, 1, 0)


            # Age
	     	features[Symbol("Age_0_5")] =  ifelse.(rawData[:Age] .<= 5, 1, 0)
	     	features[Symbol("Age_5_10")] =  ifelse.(5 .< rawData[:Age] .<= 10, 1, 0)
	     	features[Symbol("Age_10_20")] =  ifelse.(10 .< rawData[:Age] .<= 20, 1, 0)
	     	features[Symbol("Age_20_30")] =  ifelse.(20 .< rawData[:Age] .<= 30, 1, 0)
	     	features[Symbol("Age_30_40")] =  ifelse.(30 .< rawData[:Age] .<= 40, 1, 0)
	     	features[Symbol("Age_40_50")] =  ifelse.(40 .< rawData[:Age] .<= 50, 1, 0)
	     	features[Symbol("Age_50_60")] =  ifelse.(50 .< rawData[:Age] .<= 60, 1, 0)
	     	features[Symbol("Age_60_70")] =  ifelse.(60 .< rawData[:Age] .<= 70, 1, 0)
	     	features[Symbol("Age_70_80")] =  ifelse.(70 .< rawData[:Age] .<= 80, 1, 0)
	     	features[Symbol("Age_80_100")] =  ifelse.(rawData[:Age] .> 80, 1, 0)

	     	# Siblings/Spouses
	     	features[Symbol("SibSp_0")] = ifelse.(rawData[Symbol("Siblings/Spouses Aboard")] .== 0, 1, 0)
	     	features[Symbol("SibSp_1")] = ifelse.(rawData[Symbol("Siblings/Spouses Aboard")] .== 1, 1, 0)
	     	features[Symbol("SibSp_2_10")] = ifelse.(rawData[Symbol("Siblings/Spouses Aboard")] .>= 2, 1, 0)

	     	# Parents/Children
	     	features[Symbol("ParCh_0")] = ifelse.(rawData[Symbol("Parents/Children Aboard")] .== 0, 1, 0)
	     	features[Symbol("ParCh_1_3")] = ifelse.(1 .<= rawData[Symbol("Parents/Children Aboard")] .<= 3, 1, 0)
	     	features[Symbol("ParCh_4_10")] = ifelse.(rawData[Symbol("Parents/Children Aboard")] .>= 4, 1, 0)

	     	# Fare
	     	features[Symbol("FareBand_0_8")] =  ifelse.(-0.001 .< rawData[:Fare] .<= 7.925, 1, 0)
	     	features[Symbol("FareBand_8_15")] =  ifelse.(7.925 .< rawData[:Fare] .<= 14.454, 1, 0)
	     	features[Symbol("FareBand_15_32")] =  ifelse.(14.454 .< rawData[:Fare] .<= 31.138, 1, 0)
	     	features[Symbol("FareBand_32_512")] =  ifelse.(31.138 .< rawData[:Fare] .<= 512.329, 1, 0)


        end

       if dataSet == "b_cancer"
		
            features[Symbol("Label")] = ifelse.(rawData[:diagnosis] .== "B", 1, 0)

            # area_mean
	     	features[Symbol("area_mean_1")] =  ifelse.(143.499 .<= rawData[:area_mean] .<= 420.3, 1, 0)
	     	features[Symbol("area_mean_2")] =  ifelse.(420.3 .< rawData[:area_mean] .<= 551.1, 1, 0)
	     	features[Symbol("area_mean_3")] =  ifelse.(551.1 .< rawData[:area_mean] .<= 782.7, 1, 0)
	     	features[Symbol("area_mean_4")] =  ifelse.(782.7 .< rawData[:area_mean] .<= 2501.0, 1, 0)

            # area_se
	     	features[Symbol("area_se_1")] =  ifelse.(6.8 .<= rawData[:area_se] .<= 17.85, 1, 0)
	     	features[Symbol("area_se_2")] =  ifelse.(17.85 .< rawData[:area_se] .<= 24.53, 1, 0)
	     	features[Symbol("area_se_3")] =  ifelse.(24.53 .< rawData[:area_se] .<= 45.19, 1, 0)
	     	features[Symbol("area_se_4")] =  ifelse.(45.19 .< rawData[:area_se] .<= 542.2, 1, 0)
        
            # texture_mean
	     	features[Symbol("texture_mean_1")] =  ifelse.(9.7 .<= rawData[:texture_mean] .<= 16.17, 1, 0)
	     	features[Symbol("texture_mean_2")] =  ifelse.(16.17 .< rawData[:texture_mean] .<= 18.84, 1, 0)
	     	features[Symbol("texture_mean_3")] =  ifelse.(18.84 .< rawData[:texture_mean] .<= 21.8, 1, 0)
	     	features[Symbol("texture_mean_4")] =  ifelse.(21.8 .< rawData[:texture_mean] .<= 39.28, 1, 0)

            # concavity_worst
	     	features[Symbol("concavity_worst_1")] =  ifelse.(-0.001 .<= rawData[:concavity_worst] .<= 0.114, 1, 0)
	     	features[Symbol("concavity_worst_2")] =  ifelse.(0.114 .< rawData[:concavity_worst] .<= 0.227, 1, 0)
	     	features[Symbol("concavity_worst_3")] =  ifelse.(0.227 .< rawData[:concavity_worst] .<= 0.383, 1, 0)
	     	features[Symbol("concavity_worst_4")] =  ifelse.(0.383 .< rawData[:concavity_worst] .<= 1.252, 1, 0)

            # concavity_mean
	     	features[Symbol("concavity_mean_1")] =  ifelse.(-0.001 .<= rawData[:concavity_mean] .<= 0.0296, 1, 0)
	     	features[Symbol("concavity_mean_2")] =  ifelse.(0.0296 .< rawData[:concavity_mean] .<= 0.0615, 1, 0)
	     	features[Symbol("concavity_mean_3")] =  ifelse.(0.0615 .< rawData[:concavity_mean] .<= 0.131, 1, 0)
	     	features[Symbol("concavity_mean_4")] =  ifelse.(0.131 .< rawData[:concavity_mean] .<= 0.427, 1, 0)


        end


       if dataSet == "tic_tac_toe"
		
            features[Symbol("Label")] = ifelse.(rawData[:class] .== "True", 1, 0)


            # TL
	     	features[Symbol("TL_x")] =  ifelse.(rawData[:TL] .== "x", 1, 0)
	     	features[Symbol("TL_o")] =  ifelse.(rawData[:TL] .== "o", 1, 0)
	     	features[Symbol("TL_b")] =  ifelse.(rawData[:TL] .== "b", 1, 0)

            # TM
	     	features[Symbol("TM_x")] =  ifelse.(rawData[:TM] .== "x", 1, 0)
	     	features[Symbol("TM_o")] =  ifelse.(rawData[:TM] .== "o", 1, 0)
	     	features[Symbol("TM_b")] =  ifelse.(rawData[:TM] .== "b", 1, 0)

            # TR
	     	features[Symbol("TR_x")] =  ifelse.(rawData[:TR] .== "x", 1, 0)
	     	features[Symbol("TR_o")] =  ifelse.(rawData[:TR] .== "o", 1, 0)
	     	features[Symbol("TR_b")] =  ifelse.(rawData[:TR] .== "b", 1, 0)

            # ML
	     	features[Symbol("ML_x")] =  ifelse.(rawData[:ML] .== "x", 1, 0)
	     	features[Symbol("ML_o")] =  ifelse.(rawData[:ML] .== "o", 1, 0)
	     	features[Symbol("ML_b")] =  ifelse.(rawData[:ML] .== "b", 1, 0)

            # MM
	     	features[Symbol("MM_x")] =  ifelse.(rawData[:MM] .== "x", 1, 0)
	     	features[Symbol("MM_o")] =  ifelse.(rawData[:MM] .== "o", 1, 0)
	     	features[Symbol("MM_b")] =  ifelse.(rawData[:MM] .== "b", 1, 0)

            # MR
	     	features[Symbol("MR_x")] =  ifelse.(rawData[:MR] .== "x", 1, 0)
	     	features[Symbol("MR_o")] =  ifelse.(rawData[:MR] .== "o", 1, 0)
	     	features[Symbol("MR_b")] =  ifelse.(rawData[:MR] .== "b", 1, 0)

            # BL
	     	features[Symbol("BL_x")] =  ifelse.(rawData[:BL] .== "x", 1, 0)
	     	features[Symbol("BL_o")] =  ifelse.(rawData[:BL] .== "o", 1, 0)
	     	features[Symbol("BL_b")] =  ifelse.(rawData[:BL] .== "b", 1, 0)

            # BM
	     	features[Symbol("BM_x")] =  ifelse.(rawData[:BM] .== "x", 1, 0)
	     	features[Symbol("BM_o")] =  ifelse.(rawData[:BM] .== "o", 1, 0)
	     	features[Symbol("BM_b")] =  ifelse.(rawData[:BM] .== "b", 1, 0)

            # BR
	     	features[Symbol("BR_x")] =  ifelse.(rawData[:BR] .== "x", 1, 0)
	     	features[Symbol("BR_o")] =  ifelse.(rawData[:BR] .== "o", 1, 0)
	     	features[Symbol("BR_b")] =  ifelse.(rawData[:BR] .== "b", 1, 0)

        end


        # Shuffle the individuals
        features = features[shuffle(1:size(features, 1)),:] 
        trainLimit = round.(Int, size(features, 1) * 2/3)
        Limit = round.(Int, size(features, 1) * 1/3)

        train = features[1:trainLimit, :]
        test = features[(trainLimit+1):end, :]
        
        CSV.write(trainDataPath, train)
        CSV.write(testDataPath, test)

        
       	# Divide the data into tree equal size parts (A,B and C)
       	features = features[shuffle(1:size(features, 1)),:] 
        Part_A = features[1:Limit, :]
        Part_B = features[(Limit+1):2*Limit, :]
        Part_C = features[(2*Limit+1):end, :]

        CSV.write(partA_DataPath, Part_A)
        CSV.write(partB_DataPath, Part_B)
        CSV.write(partC_DataPath, Part_C)
        # If the train and test file already exist
    else
        println("=== Loading the features")
        train = CSV.read(trainDataPath)
        test = CSV.read(testDataPath)
        Part_A = CSV.read(partA_DataPath)
        Part_B = CSV.read(partB_DataPath)
        Part_C = CSV.read(partC_DataPath)
    end
    
    println("=== ... ", size(train, 1), " individuals in the train set")
    println("=== ... ", size(test, 1), " individuals in the test set")
    println("=== ... ", size(train, 2), " features")

    println("=== ... ", size(Part_A, 1), " individuals in the Part_A")
    println("=== ... ", size(Part_B, 1), " individuals in the Part_B")
    println("=== ... ", size(Part_C, 1), " individuals in the Part_C")

    
    return train, test, Part_A, Part_B, Part_C
end 


# Create the association rules related to a training set
#
# - train: individuals of the training set (each line is an individual, each column a feature except the first which is the class)
# - output: table of rules (each line is a rule, the first column corresponds to the rules class)
function createRules(dataSet::String, resultsFolder::String, train::DataFrames.DataFrame, forceCompute::Bool)

    rulesPath = resultsFolder * dataSet * "_rules.csv"
    rules = []

    if !isfile(rulesPath) || forceCompute

        println("=== Generating the rules")
        
        # Transactions
        t = train[:, 2:end]

        # Class of the transactions
        transactionClass = train[:, 1:1]

        # Number of features
        d = size(t, 2)

        # Number of transactions
        n = size(t, 1)

        mincovy = 0.01
        iterlim = 5
        RgenX = 0.1 / n
        RgenB = 0.1 / (n * d)
        
        ##################
        # Find the rules for each class
        ##################
        for y = 0:1
    	    sx = n
    	    iter = 1

            while sx >= n*mincovy
            	if iter == 1
    	        	m = Model(solver = CplexSolver())
	
                	# Create Varibles
                	@variable(m, 0<= x[i in 1:n] <= 1)
                	@variable(m, b[i in 1:d], Bin)

                	# Create Constraints
                	@constraint(m, [i in 1:n, j in 1:d], x[i] <= 1 + (t[i, j]-1)*b[j] )
                	@constraint(m, [i in 1:n], x[i] >= 1 + sum((t[i, j]-1)*b[j] for j = 1:d))
                	@constraint(m, sum(x[i] for i in 1:n) <= sx )

                	@objective(m, Max, sum(x[i] for i =1:n if transactionClass[i,1] ==y) - RgenX*sum(x[i] for i =1:n) - RgenB*sum(b[j] for j =1:d) )

                	solve(m)

	        		s = sum(getvalue(x[i]) for i =1:n if transactionClass[i,1] ==y)
 
                	iter = iter +1 
	     		end

            	rule = convert(DataFrame, hcat(append!([y], round.(Int, getvalue(b)))...))
            	if size(rules,1)>0
            		rules = append!(rules,rule)
        		else
 	    			rules = rule
        		end 
          
    
            	vb = getvalue(b)
            	#Add contraint (*)
            	@constraint(m, sum(b[j] for j in 1:d if vb[j] == 0) + sum(1-b[j] for j in 1:d if vb[j] ==1) >= 1)


            	if iter < iterlim
    	        	solve(m)
    	        	if (sum(getvalue(x[i]) for i =1:n if transactionClass[i,1] ==y)< s)
                    	sx = min(sx-1, sum(getvalue(x[i]) for i =1:n if transactionClass[i,1] ==y))
                    	iter = 1
	        		else
                    	iter = iter +1
	        		end
	    		else
	        		sx = sx -1
                	iter =1

	    		end
         	end 
        end        
        
        CSV.write(rulesPath, rules)

    else
        println("=== Loading the rules")
        rules = CSV.read(rulesPath)
    end
    
    println("=== ... ", size(rules, 1), " rules obtained") 

    return rules
end

# Sort the rules and keep the 
function sortRules(dataSet::String, resultsFolder::String, train::DataFrames.DataFrame, rules::DataFrames.DataFrame,  forceCompute::Bool)

    orderedRulesPath = resultsFolder * dataSet * "_ordered_rules.csv"

    if !isfile(orderedRulesPath) || forceCompute

        println("=== Sorting the rules")
        
        # Transactions
        t = train[:, 2:end]

        # Class of the transactions
        transactionClass = train[:, 1:1]

        # Number of features
        d = size(t, 2)

        # Number of transactions
        n = size(t, 1)
        
        # Add the two null rules
        insert!.(rules.columns, 1, append!([0], zeros(d)))
        insert!.(rules.columns, 1, append!([1], zeros(d)))
        rules = unique(rules)

#        append!(rules, DataFrames.DataFrame(append!([0], zeros(d))))
#        append!(rules, DataFrames.DataFrame(append!([1], zeros(d))))
        #        rules = [rules; zeros(2, d)]
        #        ruleClass = [ruleClass 0 1]

        # Number of rules
        L = size(rules)[1]

        Rrank = 1/L

        ################
        # Compute the v_il and p_il constants
        # p_il = :
        #  0 if rule l does not apply to transaction i
        #  1 if rule l applies to transaction i and   correctly classifies it
        # -1 if rule l applies to transaction i and incorrectly classifies it
        ################
        p = zeros(n, L)

        # For each transaction and each rule
        for i in 1:n
            for l in 1:L

                # If rule l applies to transaction i
                # i.e., if the vector t_i - r_l does not contain any negative value
                if !any(x->(x<-epsilon), [sum(t[i, k]-rules[l, k+1]) for k in 1:d])

                    # If rule l correctly classifies transaction i
                    if transactionClass[i, 1] == rules[l, 1]
                        p[i, l] = 1
                    else
                        p[i, l] = -1 
                    end
                end
            end
        end

        v = abs.(p)

        ################
        # Create and solve the model
        ###############
        m  =   Model(solver=CplexSolver(CPX_PARAM_TILIM=600))
#         m  =   Model(solver=CplexSolver())

        # u_il: rule l is the highest which applies to transaction i
        @variable(m, u[1:n, 1:L], Bin)

        # r_l: rank of rule l
        @variable(m, 1 <= r[1:L] <= L, Int)

        # rstar: rank of the highest null rule
        @variable(m, 1 <= rstar <= L)
        @variable(m, 1 <= rB <= L)

        # g_i: rank of the highest rule which applies to transaction i
        @variable(m, 1 <= g[1:n] <= L, Int)

        # s_lk: rule l is assigned to rank k
        @variable(m, s[1:L,1:L], Bin)

        # Rank of null rules
        rA = r[1]
        rB = r[2]

        # rstar == rB?
        @variable(m, alpha, Bin)

        # rstar == rA?
        @variable(m, 0 <= beta <= 1)

        # Maximize the classification accuracy
        @objective(m, Max, sum(p[i, l] * u[i, l] for i in 1:n for l in 1:L)
                   + Rrank * rstar)

        # Only one rule is the highest which applies to transaction i
        @constraint(m, [i in 1:n], sum(u[i, l] for l in 1:L) == 1)

        # g constraints
        @constraint(m, [i in 1:n, l in 1:L], g[i] >= v[i, l] * r[l])
        @constraint(m, [i in 1:n, l in 1:L], g[i] <= v[i, l] * r[l] + L * (1 - u[i, l]))

        # Relaxation improvement
        @constraint(m, [i in 1:n, l in 1:L], u[i, l] >= 1 - g[i] + v[i, l] * r[l])
        @constraint(m, [i in 1:n, l in 1:L], u[i, l] <= v[i, l]) 

        # r constraints
        @constraint(m, [k in 1:L], sum(s[l, k] for l in 1:L) == 1)
        @constraint(m, [l in 1:L], sum(s[l, k] for k in 1:L) == 1)
        @constraint(m, [l in 1:L], r[l] == sum(k * s[l, k] for k in 1:L))

        # rstar constraints
        @constraint(m, rstar >= rA)
        @constraint(m, rstar >= rB)
        @constraint(m, rstar - rA <= (L-1) * alpha)
        @constraint(m, rA - rstar <= (L-1) * alpha)
        @constraint(m, rstar - rB <= (L-1) * beta)
        @constraint(m, rB - rstar <= (L-1) * beta)
        @constraint(m, alpha + beta == 1)

        # u_il == 0 if rstar > rl (also improve relaxation)
        @constraint(m, [i in 1:n, l in 1:L], u[i, l] <= 1 - (rstar - r[l])/ (L - 1))

        solve(m)

        ###############
        # Write the rstar highest ranked rules and their corresponding class
        ###############

        # Number of rules kept in the classifier
        # (all the rules ranked lower than rstar are removed)
        relevantNbOfRules=L-round(Int, getvalue(rstar))+1

        # Sort the rules and their class by decreasing rank
        rulesOrder = getvalue(r[:])
        orderedRules = rules[sortperm(L-rulesOrder), :]
        orderedRules = orderedRules[1:relevantNbOfRules, :]

        CSV.write(orderedRulesPath, orderedRules)

    else
        println("=== Loading the sorting rules")
        orderedRules = CSV.read(orderedRulesPath)
    end 

    return orderedRules

end

function getPrecision(orderedRules::DataFrames.DataFrame, transactions::DataFrames.DataFrame)
	println("-------------------------------- Test time -----------------------------------")
    # Number of transactions
    n = size(transactions, 1)

    accuracy = 0
    
    # For all transaction i
    for i in 1:n
        
        # Get the first rule satisfied by transaction i
        ruleId = findfirst(all(Array{Float64, 2}(orderedRules[2:end][:])  .<= Array{Float64, 2}(transactions[i, 2:end]), 2))
        if orderedRules[ruleId, 1] == transactions[i, 1]
            accuracy += 1
        end
    end

    accuracy /= n

    return accuracy
    
end 
