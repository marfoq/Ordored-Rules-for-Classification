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

            # Age
	     	features[Symbol("Age_0_16")] =  ifelse.(rawData[:Age] .<= 16, 1, 0)
	     	features[Symbol("Age_16_32")] =  ifelse.(16 .< rawData[:Age] .<= 32, 1, 0)
	     	features[Symbol("Age_32_48")] =  ifelse.(32 .< rawData[:Age] .<= 48, 1, 0)
	     	features[Symbol("Age_48_64")] =  ifelse.(48 .< rawData[:Age] .<= 64, 1, 0)
	     	features[Symbol("Age_64_100")] =  ifelse.(rawData[:Age] .> 64, 1, 0)

	     	# Fare
	     	features[Symbol("FareBand_0_8")] =  ifelse.(-0.001 .< rawData[:Fare] .<= 7.925, 1, 0)
	     	features[Symbol("FareBand_8_15")] =  ifelse.(7.925 .< rawData[:Fare] .<= 14.454, 1, 0)
	     	features[Symbol("FareBand_15_32")] =  ifelse.(14.454 .< rawData[:Fare] .<= 31.138, 1, 0)
	     	features[Symbol("FareBand_32_512")] =  ifelse.(31.138 .< rawData[:Fare] .<= 512.329, 1, 0)

	     	# 


        end

        # Shuffle the individuals
        features = features[shuffle(1:size(features, 1)),:] 
        trainLimit = round.(Int, size(features, 1) * 2/3)

        train = features[1:trainLimit, :]
        test = features[(trainLimit+1):end, :]
        
        CSV.write(trainDataPath, train)
        CSV.write(testDataPath, test)

        # If the train and test file already exist
    else
        println("=== Loading the features")
        train = CSV.read(trainDataPath)
        test = CSV.read(testDataPath)
    end
    
    println("=== ... ", size(train, 1), " individuals in the train set")
    println("=== ... ", size(test, 1), " individuals in the test set")
    println("=== ... ", size(train, 2), " features")
    
    return train, test
end 


# Create the association rules related to a training set
#
# - train: individuals of the training set (each line is an individual, each column a feature except the first which is the class)
# - output: table of rules (each line is a rule, the first column corresponds to the rules class)
function createRules(dataSet::String, resultsFolder::String, train::DataFrames.DataFrame)

    rulesPath = resultsFolder * dataSet * "_rules.csv"
    rules = []

    if !isfile(rulesPath)

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
function sortRules(dataSet::String, resultsFolder::String, train::DataFrames.DataFrame, rules::DataFrames.DataFrame)

    orderedRulesPath = resultsFolder * dataSet * "_ordered_rules.csv"

    if !isfile(orderedRulesPath)

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
