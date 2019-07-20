import numpy as np

def convert_from_txt(filename):
    '''
    Loads .txt file into database for processing.
    -1 is used to denote itemsets partition
    -2 is used to denote sequence partition
    '''
    with open(filename,'r') as f:
        # Split lines into itemsets
        data = [x.split('-1')[:-1] for x in f.readlines()]
        # Split items in itemsets
        data = [([np.array(j.strip().split(),dtype=int) for j in i]) for i in data]
    return data
    