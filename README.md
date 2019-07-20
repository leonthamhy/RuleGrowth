# RuleGrowth
Python implementation of RuleGrowth by Prof. Fournier-Viger. [Paper](http://www.philippe-fournier-viger.com/spmf/rulegrowth.pdf)

Java implementation can be found [here](http://www.philippe-fournier-viger.com/spmf/index.php?link=download.php)

# Prerequisites
  * Python 3
  
# Usage
<b>Prepare text file of sequence history</b>
  - Each line represents the history of a single sequence
  - Itemsets are seperated by -1 and -2 marks the end of the sequence
  - eg. 1 2 -1 3 4 -1 -2 is a valid sequence

<b>Convert the text file into a nested list of numpy arrays</b>

`from utils.helpers import *`

`database = convert_from_txt('sample/contextPrefixSpan.txt')`

<b>Initialise RuleGrowth</b>

`rulegrowth = RuleGrowth`

<b>Fit the rulegrowth with the database</b>

`rulegrowth.fit(database)`

The fit function will take in parameters such as the minimum support and confidence and the maximum antecedent and consequent length.


<b>Predicting</b>

The RuleGrowth class can be used to predict other set of sequences after it has been fitted.

`rulegrowth.predict(database)`

<b>Saving and writing rules</b>

The rules generated can saved and written by using the `save_rules` and `write_rules` respectively.


