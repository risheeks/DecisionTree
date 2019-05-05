##############
# Name: Risheek Narayanadevarakere
# email: naraya15@purdue.edu
# Date: 3/2/19

import numpy as np
import pandas as pd
import sys
import os

def entropy(freqs):
    """ 
    entropy(p) = -SUM (Pi * log(Pi))
    >>> entropy([10.,10.])
    1.0
    >>> entropy([10.,0.])
    0
    >>> entropy([9.,3.])
    0.811278
    """
    all_freq = sum(freqs)
    entropy = 0 
    for fq in freqs:
        prob = fq * 1.0 / all_freq
        if abs(prob) > 1e-8:
            entropy += -prob * np.log2(prob)
    return entropy
    
def infor_gain(before_split_freqs, after_split_freqs):
    """
    gain(D, A) = entropy(D) - SUM ( |Di| / |D| * entropy(Di) )
    >>> infor_gain([9,5], [[2,2],[4,2],[3,1]])
    0.02922
    """
    
    gain = entropy(before_split_freqs)
    overall_size = sum(before_split_freqs)
    for freq in after_split_freqs:
        ratio = sum(freq) * 1.0 / overall_size
        gain -= ratio * entropy(freq)
    return gain
    



class Node(object):
    def __init__(self):
        self.attribute = None
        self.info_gain = None
        self.values = [0,0]
        self.clas = None
        self.left = None
        self.right = None

    def printNode(self):
        print('attribute: ' + self.attribute)
        print('info_gain: ' + str(self.info_gain))
#        print('values: (' + str(self.values[0]) + ', ' + str(self.values[1]) + ')')
#        print('class: ' + self.clas)


class Tree(object):
    
    columns = ['null']
    
    def __init__(self, train_file, test_file, model, training_percentage):
        self.root = None
        self.train_file = train_file
        self.test_file = test_file
        self.model = model
        self.training_percentage = training_percentage
        self.attributes = columns.copy()
#        print(columns)

    def printTree(self):
        self.root.printNode()
        for node in self.root.next:
            if node:
                node.printNode()

    def threshold(self, data, attribute):
        split_points = []
        for index, row in data.iterrows():
            for index1, row1 in data.iterrows():
                if row[attribute] == row1[attribute]:
                    continue
                else:
                    average = (row1 + row)/2
                    self.split_data(data, attribute, average)

        return 22

    def split_data(self, data, attribute, threshold):
        
        print('split_data by ' + attribute)
#        print(data)
        dat = list()
        dat1 = list()
        if attribute == 'Pclass':
            dat = list()
            for index, row in data.iterrows():
                if row[attribute] <= 1:
                    dat.append(row)
                else:
                    dat1.append(row)

        elif attribute == 'Age':
            dat = list()
            for index, row in data.iterrows():
                if row[attribute] <= int(threshold):
                    dat.append(row)
                else:
                    dat1.append(row)

        elif attribute == 'Fare':
            dat = list()
            for index, row in data.iterrows():
                if row[attribute] <= threshold:
                    dat.append(row)
                else:
                    dat1.append(row)

        elif attribute == 'Embarked':
            dat = list()
            for index, row in data.iterrows():
                if row[attribute] <= 10:
                    dat.append(row)
                else:
                    dat1.append(row)

        elif attribute == 'relatives':
            dat = list()
            for index, row in data.iterrows():
                if row[attribute] <= threshold:
                    dat.append(row)
                else:
                    dat1.append(row)

        else:
            i = 0
            splits = [None, None]
            for value in set(data[attribute]):
                dat = list()
                for index, row in data.iterrows():
                    if row[attribute] == value:
                        dat.append(row)
                splits[i] = dat
                i += 1
            dat = splits[0]
            dat1 = splits[1]

        new_data = pd.DataFrame(dat)
        new_data1 = pd.DataFrame(dat1)
        return new_data, new_data1

    def ID3(self, attributes, node, data):
        if len(attributes) == 0:
            return
        
        print('ID3 with--')
        print(attributes)
        print(data)
        survivors = 0
        dead = 0
        bsurvivors = 0
        bdead = 0

        for index, row in data.iterrows():
            if row[list(label.columns.values)[0]] == 1:
                bsurvivors += 1
            else:
                bdead += 1
        
        before_split_freqs = [bdead, bsurvivors]
        info_gain_list = []
        for attribute in attributes:
            thres = self.threshold(data, attribute)
            entropy_list = []
            left, right = self.split_data(data, attribute, thres)
            if len(left.index) == 0 or len(right.index) == 0:
                info_gain_list.append(0)
                print(info_gain_list)
                continue
            if attribute == 'Age':
                survivors = 0
                dead = 0
                for index, row in data.iterrows():
                    if row[attribute] <= 22:
                        if row[list(label.columns.values)[0]] == 1:
                            survivors += 1
                        else:
                            dead += 1
                entropy_list.append([dead, survivors])
                survivors = 0
                dead = 0
                for index, row in data.iterrows():
                    if row[attribute] > 22:
                        if row[list(label.columns.values)[0]] == 1:
                            survivors += 1
                        else:
                            dead += 1
                entropy_list.append([dead, survivors])


            elif attribute == 'Fare':
                survivors = 0
                dead = 0
                for index, row in data.iterrows():
                    if row[attribute] <= 18:
                        if row[list(label.columns.values)[0]] == 1:
                            survivors += 1
                        else:
                            dead += 1
                entropy_list.append([dead, survivors])
                survivors = 0
                dead = 0
                for index, row in data.iterrows():
                    if row[attribute] > 18:
                        if row[list(label.columns.values)[0]] == 1:
                            survivors += 1
                        else:
                            dead += 1
                entropy_list.append([dead, survivors])


            elif attribute == 'relatives':
                survivors = 0
                dead = 0
                for index, row in data.iterrows():
                    if row[attribute] <= 2:
                        if row[list(label.columns.values)[0]] == 1:
                            survivors += 1
                        else:
                            dead += 1
                entropy_list.append([dead, survivors])
                survivors = 0
                dead = 0
                for index, row in data.iterrows():
                    if row[attribute] > 2:
                        if row[list(label.columns.values)[0]] == 1:
                            survivors += 1
                        else:
                            dead += 1
                entropy_list.append([dead, survivors])
#                print(entropy_list)


            else:
#                print('unique values: ' + str(len(set(data[attribute]))))
                if len(set(data[attribute])) == 1:
                    continue
                for value in set(data[attribute]):
                    
                    survivors = 0
                    dead = 0
                    total = 0
                    for index, row in data.iterrows():
                        if row[attribute] == value:
                            total += 1
                            if row[list(label.columns.values)[0]] == 1:
                                survivors += 1
                            else:
                                dead += 1
                    entropy_list.append([dead, survivors])
#               print(entropy_list)
#            print(before_split_freqs)
            info_gain_list.append(infor_gain(before_split_freqs, entropy_list))
#        print(info_gain_list.index(max(info_gain_list)))
#        print(attributes[info_gain_list.index(max(info_gain_list))])
        node.attribute = attributes[info_gain_list.index(max(info_gain_list))]
        node.info_gain = max(info_gain_list)
        att = attributes.copy()
        del att[info_gain_list.index(max(info_gain_list))]
        
        left_data, right_data = self.split_data(data, node.attribute)
        left_data.drop([node.attribute], axis = 1, inplace = True, errors = 'ignore')
        right_data.drop([node.attribute], axis = 1, inplace = True, errors = 'ignore')
#        print(node.attribute)
        print(left_data)
        print(right_data)
#        print(info_gain_list)
        node.left = Node()
#        self.ID3(att, node.left, left_data)
        node.right = Node()
#        self.ID3(att, node.right, right_data)

        """
        dat = list()
        for value in set(data[node.attribute]):
            dat = list()
            for index, row in data.iterrows():
                if row[node.attribute] == value:
                    dat.append(row)
            new_data = pd.DataFrame(dat)
        """

#            self.ID3(att, )

#        node.printNode()






if __name__ == "__main__":
	# parse arguments
#    for x in sys.argv:
#        print('arg: ', x)

    train_file = sys.argv[1]
    test_file = sys.argv[2]
    model = sys.argv[3]
    
    data = pd.read_csv(train_file, delimiter = ',',  index_col=None, engine='python')
    label_file = train_file.split('.')[0] + '.label'
    label = pd.read_csv(label_file, delimiter = ',',  index_col=None, engine='python')
    data = pd.concat([data, label], axis=1, sort=False)
    columns = list(data.columns.values)
    
    if model == 'vanilla':
        training_percentage = sys.argv[4]
        row_count = sum(1 for row in data.iterrows())
        rows = int(row_count * int(training_percentage) / 100)
        data = data[:rows]
#        print(data)

        # build decision tree
        tree = Tree(train_file, test_file, model, training_percentage)
        tree.root = Node()
        tree.ID3(tree.attributes[:-1].copy(), tree.root, data.copy())
#        tree.printTree()
        """
        sample = Node()
        sample.attribute = 'sex'
        sample.entropy = 1
        sample.values = [10,10]
        sample.clas = 'survived'
        tree.root = sample;
        tree.ID3(tree.attributes, tree.root)
        
        sample1 = Node()
        sample1.attribute = 'PClass'
        sample1.entropy = 0.5
        sample1.values = [3,2]
        sample1.clas = 'survived'
        sample.next.append(sample1)
        tree.printTree()
        """
#    print(list(label.columns.values)[0]) # for column name

	# predict on testing set & evaluate the testing accuracy
	
