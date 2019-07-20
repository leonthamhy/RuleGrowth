import numpy as np
import pandas as pd
pd.set_option('mode.chained_Assignment',None)
from math import ceil
import os
import warnings
from tqdm import tqdm

class RuleGrowth:
    
    def __init__(self):
        self.database = None
        self.rules = [] # List containing tuple of valid rules (antecedentSet,consequentSet,support,confidence)
        self.minsup = None
        self.minsupRelative = None
        self.minconf = None
        self.itemlocmap ={} # Dictionary of item's first and last location in each sequence. {Item: {seqID:[first,last]}}
        self.above_threshold = [] # List of items that are >= the min support
        self.maxAntecedent = None
        self.maxConsequent = None
        self.bin = None # Interval to bin rules for sorting
        self.fitted = False
        
    def fit(self,database,minsup,minconf,maxAntecedent=0,maxConsequent=0,bin_interval=0):
        '''
        Generate sequential rules based on RuleGrowth
        
        Arguments:
            database  - List of purchase histories
            
            minsup - Minimum fraction of I => J occurances in all sequences
            
            minconf - Minimum fraction of I => J occurances in sequences with I
            
            maxAntecedent - Maximum number of items in antecedent set (default: 0)
            
            maxConsequent - Maximum number of items in consequent set (default: 0)
            
            bin_interval - Interval to bin rules for sorting (NOT IN USE)
        '''
        
        # Set all attributes
        self.database = database
        self.rules = []
        self.minsup = minsup
        self.minsupRelative = ceil(minsup * len(database))
        self.minconf = minconf
        self.itemlocmap = {}
        self.above_threshold = []
        self.maxAntecedent = maxAntecedent
        self.maxConsequent = maxConsequent
        self.bin = bin_interval
        
        # Check arguments
        assert isinstance(self.minsup,float) and self.minsup<=1 and self.minsup>0, 'Support has to be in range (0,1]'
        assert isinstance(self.minconf,float) and self.minconf<=1 and self.minconf>0, 'Confidence has to be in range (0,1]'
        assert isinstance(self.maxAntecedent,int) and self.maxAntecedent>=0, 'maxAntecedent has to be a positive integer'
        assert isinstance(self.maxConsequent,int) and self.maxConsequent>=0, 'maxConsequent has to be a positive integer'
        assert isinstance(self.bin,(float,int)) and self.bin==0 or int(1/self.bin)==1/self.bin, 'inverse of bin_intervalhas to be an integer. ie. 1has to be divisible by bin_interval'
        if self.bin:
            warnings.warn('bin_interval is not in use and will be ignored.', UserWarning)
            
        # Generate itemlocmap & above_threshold
        def getMinSupItems():
            '''
            Function to generate itemlocmap
            Record items that are above support threshold
            Remove items below threshold of support within database
            '''
            # For each sequence in database
            for seqID,sequence in enumerate(self.database):
                # For each itemset in sequence
                for idx,itemset in enumerate(sequence):
                    # For each item in itemset
                    for item in itemset:
                        # Item not found in itemlocmap yet. Add item into itemlocmap
                        if item not in self.itemlocmap:
                            self.itemlocmap[item] = {seqID:[idx,idx]}
                        # First time item is found in sequence. Add sequence into itemlocmap[item]
                        elif seqID not in self.itemlocmap[item]:
                            self.itemlocmap[item][seqID] = [idx,idx]
                        # Another occurance of item is found. Update last known location of item in sequence.
                        else:
                            self.itemlocmap[item][seqID][1] = idx
                            
            # Generate list of frequentitems
            below_threshold = []
            for item,value in self.itemlocmap.items():
                if len(value) < self.minsupRelative:
                    below_threshold.append(item)
                else:
                    self.above_threshold.append(item)
            # Remove items below min support from database, reduce number of items to loop over
            #self.database = [[np.setdiff1d(itemset,below_threshold) for itemset in seq] for seq in self.database]
            
        getMinSupItems()
        
        # Generate Rules
        def genRules():
            '''
            This function first generates valid rules with both antecedent and consequent of size 1.
            Then it will recursively expand the rules through expandLeft and expandRight
            '''
            # Double for loop to run through all possible I => J combination with no repeats
            for i in range(len(self.above_threshold)):
                for j in range(i+1,len(self.above_threshold)):
                    # items
                    itemI, itemJ = self.above_threshold[i], self.above_threshold[j]
                    # Dictionary of {seqID:[first,last]} for items
                    occurancesI, occurancesJ = self.itemlocmap[itemI], self.itemlocmap[itemJ]
                    # Sequences containing items I & J
                    allseqI, allseqJ = set(occurancesI.keys()), set(occurancesJ.keys())
                    # Sequences containing I => J or J => I
                    allseqIJ, allseqJI = set(), set()
                    
                    # Sequences that contains both I & J
                    allseqboth = set.intersection(allseqI,allseqJ)
                    
                    # Sequences that have I => J or J => I
                    for seqID in allseqboth:
                        if self.itemlocmap[itemI][seqID][0] < self.itemlocmap[itemJ][seqID][1]:
                            allseqIJ.add(seqID)
                        if self.itemlocmap[itemJ][seqID][0] < self.itemlocmap[itemI][seqID][1]:
                            allseqJI.add(seqID)
                            
                    # Check IJ
                    if len(allseqIJ) >= self.minsupRelative:
                        confIJ = len(allseqIJ) / len(occurancesI)
                        antecedentSet = set([itemI])
                        consequentSet = set([itemJ])
                        # Add those with valid support and confidence
                        if confIJ >= self.minconf:
                            self.rules.append((antecedentSet,consequentSet,len(allseqIJ)/len(self.database),confIJ))
                        # Expand left if possible
                        if not self.maxAntecedent or len(antecedentSet) < self.maxAntecedent:
                            expandLeft(antecedentSet,consequentSet,allseqI,allseqIJ,occurancesJ)
                        # Expand right if possible
                        if not self.maxConsequent or len(consequentSet) < self.maxConsequent:
                            expandRight(antecedentSet,consequentSet,allseqI,allseqJ,allseqIJ,occurancesI,occurancesJ)
                            
                    # Check JI
                    if len(allseqJI) >= self.minsupRelative:
                        confJI = len(allseqJI) / len(occurancesJ)
                        antecedentSet = set([itemJ])
                        consequentSet = set([itemI])
                        # Add those with valid support and confidence
                        if confJI >= self.minconf:
                            self.rules.append((antecedentSet,consequentSet,len(allseqJI)/len(self.database),confJI))
                        # Expand left if possible
                        if not self.maxAntecedent or len(antecedentSet) < self.maxAntecedent:
                            expandLeft(antecedentSet,consequentSet,allseqJ,allseqJI,occurancesI)
                        # Expand right if possible
                        if not self.maxConsequent or len(consequentSet) < self.maxConsequent:
                            expandRight(antecedentSet,consequentSet,allseqJ,allseqI,allseqJI,occurancesJ,occurancesI)
                            
                            
        def expandLeft(antecedentSet,consequentSet,allseqI,allseqIJ,occurancesJ):
            '''
            This function builds on an existing rulebyadding an item to the antecedentset
            '''
            # A dictionary of items C and sequenceIDs where IuC => J
            possibleC = dict()
            # Total number of possible sequences
            seqsLeft = len(allseqIJ)
            
            # Loop to populate possibleC
            # For each sequenceID where I => J
            for seqID in allseqIJ:
                sequence = self.database[seqID] # Get sequence
                firstJ, lastJ = occurancesJ[seqID] # Get last occurance of J in sequene
                
                # For each itemsetID before the itemset containing the last occurance of J in the sequence
                for itemsetID in range(lastJ):
                    itemset = sequence[itemsetID] # Get itemset
                    # For each item in itemset
                    for item in itemset:
                        if any([i>=item for i in antecedentSet]) or item in consequentSet or item not in self.above_threshold:
                            # Ensure that the item is not already present in either
                            # antecedent or consequent set.
                            # To prevent repeated rules, only items greater in value
                            # than all items inside antecedent set will be considered
                            continue
                        if item not in possibleC:
                            if seqsLeft >= self.minsupRelative:
                                # items that are not able to meet the
                                # minimum requirement are ignored
                                possibleC[item] = set([seqID])
                                
                        elif len(possibleC[item]) + seqsLeft < self.minsupRelative:
                            # Remove items from possibleC when it is no
                            # longer possible to meet support requirement
                            del possibleC[item]
                            continue
                        
                        else:
                            # Add sequenceID
                            possibleC[item].add(seqID)
                    
                # Decrease max possible sequence left
                seqsLeft -= 1
                
            # Loop through possibleC to generate valid rules
            for itemC, seqIDs in possibleC.items():
                # Check if minimum support requirement is met
                if len(seqIDs) >= self.minsupRelative:
                    # SeqIDs of IuC 
                    allseqIC = set.intersection(set(self.itemlocmap[itemC].keys()),allseqI)
                    
                    # Confidence of IuC => J
                    # support(IuC => J) / support(IuC)
                    confIC_J = len(seqIDs) / len(allseqIC)
                    
                    # New antecedent set
                    itemsIC = antecedentSet.copy()
                    itemsIC.add(itemC)
                    
                    # Add rule
                    if confIC_J >= self.minconf:
                        self.rules.append((itemsIC,consequentSet,len(seqIDs)/len(self.database),confIC_J))
                        
                    # Expand left if possible
                    if not self.maxAntecedent or len(itemsIC) < self.maxAntecedent:
                        expandLeft(itemsIC,consequentSet,allseqIC,seqIDs,occurancesJ)

        def expandRight(antecedentSet,consequentSet,allseqI,allseqJ,allseqIJ,occurancesI,occurancesJ):
            '''
            This function builds on an existing rule by adding an item to the consequent set
            '''

            # A dictionary of item C and sequenceIDs where I => JuC
            possibleC = dict()
            # Total number of possible sequences
            seqsLeft = len(allseqIJ)

            # Loop to populate possibleC 
            # For each sequenceID where I => J
            for seqID in allseqIJ:
                sequence = self.database[seqID] # Get sequence
                firstI, lastI = occurancesI[seqID] # Get first occurance of I in sequence

                # For each itemsetID after the itemset containing the first occurance of I in the sequence
                for itemsetID in range(firstI+1,len(sequence)):
                    itemset = sequence[itemsetID] # Get itemset
                    # For each item in itemset
                    for item in itemset:
                        if any([i>=item for i in consequentSet]) or item in antecedentSet or item not in self.above_threshold:
                            # Ensure that the item is not already present in either
                            # antecedent or consequent set
                            # To prevent repeated rules, only item greater in value
                            # than all items inside the consequent set will be considered
                            continue
                        if item not in possibleC:
                            if seqsLeft >= self.minsupRelative:
                                # items that are not able to meet the 
                                # minimum requirement are ignored
                                possibleC[item] = set([seqID])

                        elif len(possibleC[item]) + seqsLeft < self.minsupRelative:
                            # Remove items from possibleC when it is no 
                            # longer possible to meet support requirement
                            del possibleC[item]
                            continue

                        else:
                            # Add sequenceID
                            possibleC[item].add(seqID)

                # Decrease max possible sequence left
                seqsLeft -= 1

            # Loop through possibleC to generate valid rules
            for itemC, seqIDs in possibleC.items():
                # Check if minimum support requirement is met
                if len(seqIDs) >= self.minsupRelative:
                    # SeqIDs of JuC
                    allseqJC = set()
                    # New consequent occurance map
                    occurancesJC = dict()

                    # Loop through the consequent set to find intersection with item C
                    # Update the occurance of consequent set to make sure the last 
                    # occurance is the earliest last occurance among all items
                    for seqID_J in allseqJ:
                        occurancesC = self.itemlocmap[itemC].get(seqID_J,False)
                        if occurancesC:
                            allseqJC.add(seqID_J)
                            firstJ, lastJ = occurancesJ[seqID_J]
                            if occurancesC[1] < lastJ:
                                occurancesJC[seqID_J] = occurancesC
                            else:
                                occurancesJC[seqID_J] = [firstJ,lastJ]

                    # Confidence of I => JuC
                    # support(I => JuC) / support(I)
                    confI_JC = len(seqIDs) / len(allseqI)

                    # New consequent set
                    itemsJC = consequentSet.copy()
                    itemsJC.add(itemC)

                    # Add rule
                    if confI_JC >= self.minconf:
                        self.rules.append((antecedentSet,itemsJC,len(seqIDs)/len(self.database),confI_JC))

                    # Expand left if possible
                    if not self.maxAntecedent or len(antecedentSet) < self.maxAntecedent:
                        expandLeft(antecedentSet,itemsJC,allseqI,seqIDs,occurancesJC)

                    # Expand right if possible
                    if not self.maxConsequent or len(itemsJC) < self.maxConsequent:
                        expandRight(antecedentSet,itemsJC,allseqI,allseqJC,seqIDs,occurancesI,occurancesJC)

        genRules()

        # Sort rules
        def sort_rules():
            '''
            Sort the rules in the following order of importance
            1. Binned confidence
            2. length of antecedent
            3. Confidence
            '''

            def binning(x):
                '''
                Bin confidence by rounding down to nearest 0.05
                binning(0.44) => 0.4
                binning(0.045) => 0.45
                binning(0.51) => 0.5
                '''
                if self.bin == 0:
                    return x
                intervals = 1 / self.bin
                return int(x*intervals)/intervals

            # Sort by confidence
            temp = sorted(self.rules,key=lambda x:x[3],reverse=True)
            # Sort by length of antecedent 
            #temp = sorted(temp,key=lambda x:len(x[0]),reverse=True)
            # Sort by binned confidence
            #temp = sorted(temp,key=lambda x:binning(x[3]),reverse=True)
            # Return sorted rules
            self.rules = temp

        sort_rules()

        # Clear memory
        self.database = None
        self.itemlocmap = {}
        self.above_threshold = []

        # Number of unique consequent items
        print(f'RULEGROWTH FITTED WITH {len(self.rules)} RULES AND {len(set.union(*[rule[1] for rule in self.rules]))} UNIQUE CONSEQUENT ITEMS.')

        # Fitted 
        self.fitted = True
        
    def predict(self,database,n=3,conf=False,diversity_multiplier=1,rules=False,diversity_percentage=False):
        '''
        Predict the top n items based on a given rule
        
        Arguments:
            database             - List of purchase histories
            
            n                    - Maximum number of predictions per user (default: 3)
            
            conf                 - Return confidence after diversity_multiplier (default: False)
            
            diversity_multiplier - Amount to augment unseen items' confidence (default: 1)
            
            rules                - Return highest confidence rule that predicted item (default: False)
            
            diversity_percentage - Return percentage of items predicted per customer that was unseen previously (default: False)
        '''
        
        # Requires rules to predict
        if not self.fitted:
            raise Exception('Needto fit or load rules before predicting')
            
        if not self.rules:
            return 'There are no rules'
        
        try:
            dataset = [set().union(*i) for i in database]
        except:
            return 'Please check database format.'
        
        predictions = [] # List of predictions
        
        # Go through history of purchase for each sequence
        for history in tqdm(dataset):
            pred_items = set() # set of predicted items
            pred_rank = [] # list of predicted item (and its confidence/rules/is_new)
            smallest = diversity_multiplier
            
            # Go through each rule to generate up to n predictions, if possible
            for rule in self.rules:
                # For a set of history that obeys a rule and the consequent set is
                # not already present in the predictions
                if rule[0].issubset(history) and not rule[1].issubset(pred_items):
                    for i in rule[1].difference(pred_items):
                        if i not in history:
                            pred_rank.append((i,rule[3]*diversity_multiplier,f'{rule[0]} => {rule[1]}',1))
                            if rule[3]*diversity_multiplier < smallest:
                                smallest = rule[3] * diversity_multiplier
                        else:
                            pred_rank.append((i,rule[3],f'{rule[0]} => {rule[1]}',0))
                            if rule[3] < smallest:
                                smallest = rule[3]
                                
                    # Add it to predictions
                    pred_items = pred_items.union(rule[1])
                    
                # Stop when enough predictions are found and no rules can be added even after multiplying
                if len(pred_items) >= n and rule[3]*diversity_multiplier < smallest:
                    break
            pred_rank = sorted(pred_rank,key=lambda x:x[1],reverse=True)
            predictions.append(pred_rank[:n])
            
        # Get average new items 
        if diversity_percentage:
            average_lst = []
            for pred in predictions:
                if not len(pred):
                    average_lst.append(0)
                    break
                total = 0
                for p in pred:
                    total += p[3]
                average_lst.append(total/len(pred))
            avg = np.mean(average_lst)
            
        # Return with confidence and rule
        if conf and rules:
            if diversity_percentage:
                return [[(p[0],p[1],p[2]) for p in preds] for preds in predictions], avg
            return [[(p[0],p[1],p[2]) for p in preds] for preds in predictions]
        
        # Return with confidence
        if conf:
            if diversity_percentage:
                return [[(p[0],p[1]) for p in preds] for preds in predictions], avg
            return [[(p[0],p[1]) for p in preds] for preds in predictions]
        
        # Return with rules
        if rules:
            if diversity_percentage:
                return [[(p[0],p[2]) for p in preds] for preds in predictions], avg
            return [[(p[0],p[2]) for p in preds] for preds in predictions]
        
        # Return just predictions
        if diversity_percentage:
            return [[p[0] for p in preds] for preds in predictions], avg
        return [[p[0] for p in preds] for preds in predictions]
    
    def write_rules(self,filename,filetype='csv'):
        '''
        Write rules to .txt or .csv file. Default is csv
        '''
        if not self.rules:
            return 'There are no rules to write'
        
        if filetype == 'txt':
            with open(f'{filename}.txt','w') as text_file:
                for rule in self.rules:
                    text_file.write('{} ==> {} #SUPORT: {} #CONFIDENCE: {}\n'.format(*rule))
            print(rf'Rules written to {os.getcwd()}\{filename}.txt')
            
        elif filetype == 'csv':
            df = pd.DataFrame(self.rules,columns=['antecedent','consequent','support','confidence'])
            df.to_csv(f'{filename}.csv',index=False)
            print(rf'Rules written to {os.getcwd()}\{filename}.csv')
            
        else:
            print(f'{filetype} is not a supported file format')
            
    def save_rules(self,filename):
        '''
        Save rules and settings to a .npy file for prediction later
        '''
        # Can't save empty rule
        if not self.rules:
            return 'There are no rules to save'
        else:
            rules_and_settings = self.rules.copy()
            rules_and_settings.append([self.minsup,self.minconf,self.maxAntecedent,self.maxConsequent])
            np.save(filename,np.array(rules_and_settings))
            print(rf'Rules successfully saved to {os.getcwd()}\{filename}.npy')
            
    def load_rules(self,filename):
        '''
        Load pre-trained rules
        '''
        rules_and_settings = np.load(filename,allow_pickle=True)
        self.rules = list(rules_and_settings[:-1])
        self.minsup,self.minconf,self.maxAntecedent,self.maxConsequent = rules_and_settings[-1]
        
    def __repr__(self):
        return f'RuleGrowth(minsup={self.minsup}, minconf={self.minconf}, maxAntecedent={self.maxAntecedent}, maxConsequent={self.maxConsequent})'
    
    def __str__(self):
        return f'RuleGrowth(minsup={self.minsup}, minconf={self.minconf}, maxAntecedent={self.maxAntecedent}, maxConsequent={self.maxConsequent})'
            













































