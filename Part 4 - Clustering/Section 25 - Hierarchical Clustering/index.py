# Apriori

# import libraries 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

Data = pd.read_csv('Market_Basket_Optimisation.csv',header=None)

transactions = []

for i in range(0,7501):
        transactions.append([str(Data.values[i,j]) for j in range(0,20)])


from apyori import apriori

association_rules = apriori(transactions , min_support = .003 , min_confidence = .2 , min_lift = 2)
association_results = list(association_rules)  

for item in association_results:

    # first index of the inner list
    # Contains base item and add item
    pair = item[0] 
    items = [x for x in pair]
    print("Rule: " + items[0] + " -> " + items[1])

    #second index of the inner list
    print("Support: " + str(item[1]))

    #third index of the list located at 0th
    #of the third index of the inner list

    print("Confidence: " + str(item[2][0][2]))
    print("Lift: " + str(item[2][0][3]))
    print("=====================================")