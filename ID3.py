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
    if all_freq == 0:
        return -1
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
        if entropy(freq) == -1:
            return 0
        ratio = sum(freq) * 1.0 / overall_size
        gain -= ratio * entropy(freq)
    return gain
    



class Node(object):
    def __init__(self):
        self.attribute = None
        self.info_gain = None
        self.values = [0,0]
        self.threshold = None
        self.clas = None
        self.left = None
        self.right = None

    def printNode(self):
        print('node--')
        if self.attribute:
            print('attribute: ' + self.attribute)
        if self.info_gain:
            print('info_gain: ' + str(self.info_gain))
        if self.threshold:
            print('threshold: ' + str(self.threshold))
        if self.values:
            print('values: (' + str(self.values[0]) + ', ' + str(self.values[1]) + ')')
        if self.clas:
            print('class: ' + self.clas)


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

    def printTree(self, node):
        node.printNode()
        if node.left:
            self.printTree(node.left)
        if node.right:
            self.printTree(node.right)
    
    def threshold1(self, data, attribute):
        split_points = []
        survivors = 0
        dead = 0
        data.sort_values(attribute)
#        print('attribute: ' + attribute)
        row1 = None
#        for row in np.Dataframe(np.unique(data[attribute])).iterrows():
        for row in data[attribute].unique():
            if row1:
                if row == row1:
                    continue
                else:
                    average = (row1 + row)/2
                    split_points.append(average)
            row1 = row

        
        bdead = 0
        bsurvivors = 0
        for index, row in data.iterrows():
            if row[list(label.columns.values)[0]] == 1:
                bsurvivors += 1
            else:
                bdead += 1
        
        before_split_freqs = [bdead, bsurvivors]

        max_info_gain = 0
        max_info_gain_point = 0
        for point in set(split_points):
            survivors = 0
            dead = 0
            entropy_list = []
            for index, row in data.iterrows():
                if row[attribute] <= point:
                    if row[list(label.columns.values)[0]] == 1:
                        survivors += 1
                    else:
                        dead += 1
            entropy_list.append([dead, survivors])
            survivors = 0
            dead = 0
            for index, row in data.iterrows():
                if row[attribute] > point:
                    if row[list(label.columns.values)[0]] == 1:
                        survivors += 1
                    else:
                        dead += 1
            entropy_list.append([dead, survivors])
            gain = infor_gain(before_split_freqs, entropy_list)
            if gain > max_info_gain:
                max_info_gain = gain
                max_info_gain_point = point

#        print('max info_gain_point for '  + attribute + ': ' + str(max_info_gain_point))

        return max_info_gain_point



    
    def threshold(self, data, attribute):
        split_points = []
        survivors = 0
        dead = 0
#        data.list()
        for index, row in data.iterrows():
            
            for index1, row1 in data.iterrows():
                if row[attribute] == row1[attribute]:
                    continue
                else:
                    average = (row1[attribute] + row[attribute])/2
                    split_points.append(average)
        bdead = 0
        bsurvivors = 0
        for index, row in data.iterrows():
            if row[list(label.columns.values)[0]] == 1:
                bsurvivors += 1
            else:
                bdead += 1

        before_split_freqs = [bdead, bsurvivors]
    
        max_info_gain = 0
        max_info_gain_point = 0
        for point in set(split_points):
            survivors = 0
            dead = 0
            entropy_list = []
            for index, row in data.iterrows():
                if row[attribute] <= point:
                    if row[list(label.columns.values)[0]] == 1:
                        survivors += 1
                    else:
                        dead += 1
            entropy_list.append([dead, survivors])
            survivors = 0
            dead = 0
            for index, row in data.iterrows():
                if row[attribute] > point:
                    if row[list(label.columns.values)[0]] == 1:
                        survivors += 1
                    else:
                        dead += 1
            entropy_list.append([dead, survivors])
            gain = infor_gain(before_split_freqs, entropy_list)
            if gain > max_info_gain:
                max_info_gain = gain
                max_info_gain_point = point

#        print('max info_gain_point for '  + attribute + ': ' + str(max_info_gain_point))

    
        return max_info_gain_point

    def split_data(self, data, attribute, threshold, node):
#        print('splitting by ' + attribute)
        node.threshold = threshold
#        print(data)
        dat = list()
        dat1 = list()
        
        dat = list()
        for index, row in data.iterrows():
            if int(row[attribute]) <= threshold:
                dat.append(row)
            else:
                dat1.append(row)
        new_data = pd.DataFrame(dat)
        new_data1 = pd.DataFrame(dat1)
        return new_data, new_data1
                
    
    def recurseTest(self, row, node):
#        self.root.printNode()
        if node.attribute == 'Sex' or node.attribute == 'isAlone':
            node.threshold = 0.5
        if node.threshold:
            if row[node.attribute] <= node.threshold:
                return self.recurseTest(row, node.left)
            else:
                return self.recurseTest(row, node.right)
        else:
#            node.printNode()
#            print(node.clas)
            return node.clas


    
    def test(self, data):
        survival = None
        correct = 0
        wrong = 0
        for index, row in data.iterrows():
            survival = self.recurseTest(row, tree.root)
#            print(survival)
#            if (survival == 'alive' and row['survived'] == 1) or (survival == 'dead' and row['survived'] == 0):
#                correct += 1
            if (survival == 'alive' and row['survived'] == 0) or (survival == 'dead' and row['survived'] == 1):
                wrong += 1
            else:
                correct += 1
#        print('correct: ' + str(correct))
#        print('wring: ' + str(wrong))
        return (correct/(correct+wrong))
    
    
    def ID3_depth(self, attributes, node, data, max_depth):
        
    
        '''print('data at the start of ID3:')
        print(data)'''
        survivors = 0
        dead = 0
        bsurvivors = 0
        bdead = 0

        for index, row in data.iterrows():
            if row[list(label.columns.values)[0]] == 1:
                bsurvivors += 1
            else:
                bdead += 1
#        print(len(attributes))
        if len(attributes) <= (len(columns)-int(max_depth)-1) or len(attributes) == 0:
            node.values = None
            if bsurvivors >= bdead:
                node.clas = 'alive'
            else:
                node.clas = 'dead'
#            node.printNode()
            return
    
        if bsurvivors == 0 or bdead == 0:
#            print('We\'re done here!!!')
            node.values = None
            if bsurvivors == 0:
                node.clas = 'dead'
            else:
                node.clas = 'alive'
#            node.printNode()
            return
        
        before_split_freqs = [bdead, bsurvivors]
        info_gain_list = []
        for attribute in attributes:
            thres = self.threshold1(data, attribute)
            entropy_list = []
            
            left, right = self.split_data(data, attribute, thres, node)
            if len(left.index) == 0 or len(right.index) == 0:
                info_gain_list.append(0)
                '''print('nothing on split with ' + attribute + ', ' + str(thres))
                print('left: ')
                print(left)
                print('right:')
                print(right)'''
                continue
            if attribute == 'Age':
                survivors = 0
                dead = 0
                for index, row in data.iterrows():
                    if row[attribute] <= thres:
                        if row[list(label.columns.values)[0]] == 1:
                            survivors += 1
                        else:
                            dead += 1
                entropy_list.append([dead, survivors])
                survivors = 0
                dead = 0
                for index, row in data.iterrows():
                    if row[attribute] > thres:
                        if row[list(label.columns.values)[0]] == 1:
                            survivors += 1
                        else:
                            dead += 1
                entropy_list.append([dead, survivors])


            elif attribute == 'Fare':
                survivors = 0
                dead = 0
                for index, row in data.iterrows():
                    if row[attribute] <= thres:
                        if row[list(label.columns.values)[0]] == 1:
                            survivors += 1
                        else:
                            dead += 1
                entropy_list.append([dead, survivors])
                survivors = 0
                dead = 0
                for index, row in data.iterrows():
                    if row[attribute] > thres:
                        if row[list(label.columns.values)[0]] == 1:
                            survivors += 1
                        else:
                            dead += 1
                entropy_list.append([dead, survivors])


            elif attribute == 'relatives':
                survivors = 0
                dead = 0
                for index, row in data.iterrows():
                    if row[attribute] <= thres:
                        if row[list(label.columns.values)[0]] == 1:
                            survivors += 1
                        else:
                            dead += 1
                entropy_list.append([dead, survivors])
                survivors = 0
                dead = 0
                for index, row in data.iterrows():
                    if row[attribute] > thres:
                        if row[list(label.columns.values)[0]] == 1:
                            survivors += 1
                        else:
                            dead += 1
                entropy_list.append([dead, survivors])


            else:
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

            info_gain_list.append(infor_gain(before_split_freqs, entropy_list))

        node.attribute = attributes[info_gain_list.index(max(info_gain_list))]
        node.info_gain = max(info_gain_list)
        att = attributes.copy()
        del att[info_gain_list.index(max(info_gain_list))]
        thres = self.threshold1(data, node.attribute)
#        print('threshold1 = ' + str(thres))
        node.threshold = thres
#        print('node.threshold = ' + str(node.threshold))
        left_data, right_data = self.split_data(data, node.attribute, thres, node)
        left_data.drop([node.attribute], axis = 1, inplace = True, errors = 'ignore')
        right_data.drop([node.attribute], axis = 1, inplace = True, errors = 'ignore')
        '''print('left_data:')
        print(left_data)
        print('right_data:')
        print(right_data)'''
        node.left = Node()
        self.ID3_depth(att, node.left, left_data, max_depth)
        node.right = Node()
        self.ID3_depth(att, node.right, right_data, max_depth)


    def ID3_min_split(self, attributes, node, data, min_split):
        
        '''print('data at the start of ID3:')
        print(data)'''
        survivors = 0
        dead = 0
        bsurvivors = 0
        bdead = 0

        for index, row in data.iterrows():
            if row[list(label.columns.values)[0]] == 1:
                bsurvivors += 1
            else:
                bdead += 1

        if len(attributes) == 0:
            node.values = None
            if bsurvivors >= bdead:
                node.clas = 'alive'
            else:
                node.clas = 'dead'
#            node.printNode()
            return
    
        if bsurvivors == 0 or bdead == 0:
#            print('We\'re done here!!!')
            node.values = None
            if bsurvivors == 0:
                node.clas = 'dead'
            else:
                node.clas = 'alive'
#            node.printNode()
            return
        
        before_split_freqs = [bdead, bsurvivors]
        info_gain_list = []
        for attribute in attributes:
            thres = self.threshold1(data, attribute)
            entropy_list = []
            
            left, right = self.split_data(data, attribute, thres, node)
            if len(left.index) < int(min_split) or len(right.index) < int(min_split):
                info_gain_list.append(0)
                '''print('nothing on split with ' + attribute + ', ' + str(thres))
                print('left: ')
                print(left)
                print('right:')
                print(right)'''
                continue
            if attribute == 'Age':
                survivors = 0
                dead = 0
                for index, row in data.iterrows():
                    if row[attribute] <= thres:
                        if row[list(label.columns.values)[0]] == 1:
                            survivors += 1
                        else:
                            dead += 1
                entropy_list.append([dead, survivors])
                survivors = 0
                dead = 0
                for index, row in data.iterrows():
                    if row[attribute] > thres:
                        if row[list(label.columns.values)[0]] == 1:
                            survivors += 1
                        else:
                            dead += 1
                entropy_list.append([dead, survivors])


            elif attribute == 'Fare':
                survivors = 0
                dead = 0
                for index, row in data.iterrows():
                    if row[attribute] <= thres:
                        if row[list(label.columns.values)[0]] == 1:
                            survivors += 1
                        else:
                            dead += 1
                entropy_list.append([dead, survivors])
                survivors = 0
                dead = 0
                for index, row in data.iterrows():
                    if row[attribute] > thres:
                        if row[list(label.columns.values)[0]] == 1:
                            survivors += 1
                        else:
                            dead += 1
                entropy_list.append([dead, survivors])


            elif attribute == 'relatives':
                survivors = 0
                dead = 0
                for index, row in data.iterrows():
                    if row[attribute] <= thres:
                        if row[list(label.columns.values)[0]] == 1:
                            survivors += 1
                        else:
                            dead += 1
                entropy_list.append([dead, survivors])
                survivors = 0
                dead = 0
                for index, row in data.iterrows():
                    if row[attribute] > thres:
                        if row[list(label.columns.values)[0]] == 1:
                            survivors += 1
                        else:
                            dead += 1
                entropy_list.append([dead, survivors])


            else:
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

            info_gain_list.append(infor_gain(before_split_freqs, entropy_list))

        node.attribute = attributes[info_gain_list.index(max(info_gain_list))]
        node.info_gain = max(info_gain_list)
        att = attributes.copy()
        del att[info_gain_list.index(max(info_gain_list))]
        thres = self.threshold1(data, node.attribute)
#        print('threshold1 = ' + str(thres))
        node.threshold = thres
#        print('node.threshold = ' + str(node.threshold))
        left_data, right_data = self.split_data(data, node.attribute, thres, node)
        left_data.drop([node.attribute], axis = 1, inplace = True, errors = 'ignore')
        right_data.drop([node.attribute], axis = 1, inplace = True, errors = 'ignore')
        '''print('left_data:')
        print(left_data)
        print('right_data:')
        print(right_data)'''
        node.left = Node()
        self.ID3_min_split(att, node.left, left_data, min_split)
        node.right = Node()
        self.ID3_min_split(att, node.right, right_data, min_split)


    
    def ID3(self, attributes, node, data):
        
    
        '''print('data at the start of ID3:')
        print(data)'''
        survivors = 0
        dead = 0
        bsurvivors = 0
        bdead = 0

        for index, row in data.iterrows():
            if row[list(label.columns.values)[0]] == 1:
                bsurvivors += 1
            else:
                bdead += 1

        if len(attributes) == 0:
            node.values = None
            if bsurvivors >= bdead:
                node.clas = 'alive'
            else:
                node.clas = 'dead'
#            node.printNode()
            return
    
        if bsurvivors == 0 or bdead == 0:
#            print('We\'re done here!!!')
            node.values = None
            if bsurvivors == 0:
                node.clas = 'dead'
            else:
                node.clas = 'alive'
#            node.printNode()
            return
        
        before_split_freqs = [bdead, bsurvivors]
        info_gain_list = []
        for attribute in attributes:
            thres = self.threshold1(data, attribute)
            entropy_list = []
            
            left, right = self.split_data(data, attribute, thres, node)
            if len(left.index) == 0 or len(right.index) == 0:
                info_gain_list.append(0)
                '''print('nothing on split with ' + attribute + ', ' + str(thres))
                print('left: ')
                print(left)
                print('right:')
                print(right)'''
                continue
            if attribute == 'Age':
                survivors = 0
                dead = 0
                for index, row in data.iterrows():
                    if row[attribute] <= thres:
                        if row[list(label.columns.values)[0]] == 1:
                            survivors += 1
                        else:
                            dead += 1
                entropy_list.append([dead, survivors])
                survivors = 0
                dead = 0
                for index, row in data.iterrows():
                    if row[attribute] > thres:
                        if row[list(label.columns.values)[0]] == 1:
                            survivors += 1
                        else:
                            dead += 1
                entropy_list.append([dead, survivors])


            elif attribute == 'Fare':
                survivors = 0
                dead = 0
                for index, row in data.iterrows():
                    if row[attribute] <= thres:
                        if row[list(label.columns.values)[0]] == 1:
                            survivors += 1
                        else:
                            dead += 1
                entropy_list.append([dead, survivors])
                survivors = 0
                dead = 0
                for index, row in data.iterrows():
                    if row[attribute] > thres:
                        if row[list(label.columns.values)[0]] == 1:
                            survivors += 1
                        else:
                            dead += 1
                entropy_list.append([dead, survivors])


            elif attribute == 'relatives':
                survivors = 0
                dead = 0
                for index, row in data.iterrows():
                    if row[attribute] <= thres:
                        if row[list(label.columns.values)[0]] == 1:
                            survivors += 1
                        else:
                            dead += 1
                entropy_list.append([dead, survivors])
                survivors = 0
                dead = 0
                for index, row in data.iterrows():
                    if row[attribute] > thres:
                        if row[list(label.columns.values)[0]] == 1:
                            survivors += 1
                        else:
                            dead += 1
                entropy_list.append([dead, survivors])


            else:
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

            info_gain_list.append(infor_gain(before_split_freqs, entropy_list))

        node.attribute = attributes[info_gain_list.index(max(info_gain_list))]
        node.info_gain = max(info_gain_list)
        att = attributes.copy()
        del att[info_gain_list.index(max(info_gain_list))]
        thres = self.threshold1(data, node.attribute)
#        print('threshold1 = ' + str(thres))
        node.threshold = thres
#        print('node.threshold = ' + str(node.threshold))
        left_data, right_data = self.split_data(data, node.attribute, thres, node)
        left_data.drop([node.attribute], axis = 1, inplace = True, errors = 'ignore')
        right_data.drop([node.attribute], axis = 1, inplace = True, errors = 'ignore')
        '''print('left_data:')
        print(left_data)
        print('right_data:')
        print(right_data)'''
        node.left = Node()
        self.ID3(att, node.left, left_data)
        node.right = Node()
        self.ID3(att, node.right, right_data)

        """
        dat = list()
        for value in set(data[node.attribute]):
            dat = list()
            for index, row in data.iterrows():
                if row[node.attribute] == value:
                    dat.append(row)
            new_data = pd.DataFrame(dat)
        """


if __name__ == "__main__":

    train_file = sys.argv[1]
    test_file = sys.argv[2]
    model = sys.argv[3]
    
    data = pd.read_csv(train_file, delimiter = ',',  index_col=None, engine='python')
    testDat = pd.read_csv(test_file, delimiter = ',',  index_col=None, engine='python')
    test_label_file = train_file.split('.')[0] + '.label'
    test_label = pd.read_csv(test_label_file, delimiter = ',',  index_col=None, engine='python')
    testDat = pd.concat([testDat, test_label], axis=1, sort=False)
    label_file = train_file.split('.')[0] + '.label'
    label = pd.read_csv(label_file, delimiter = ',',  index_col=None, engine='python')
    data = pd.concat([data, label], axis=1, sort=False)
    columns = list(data.columns.values)
    dat = data.copy()
    if model == 'vanilla':
        training_percentage = sys.argv[4]
        row_count = sum(1 for row in data.iterrows())
        rows = int(row_count * int(training_percentage) / 100)
        data = data[:rows]

        # build decision tree
        tree = Tree(train_file, test_file, model, training_percentage)
        tree.root = Node()
        tree.ID3(tree.attributes[:-1].copy(), tree.root, data.copy())
#        tree.printTree(tree.root)

	    # predict on testing set & evaluate the testing accuracy
        print('Train set accuracy: ' + str(tree.test(dat)))
        print('Test set accuracy: ' + str(tree.test(testDat)))

    elif model == 'depth':
        training_percentage = sys.argv[4]
        validation_percentage = sys.argv[5]
        max_depth = sys.argv[6]
        row_count = sum(1 for row in data.iterrows())
        rows = int(row_count * int(training_percentage) / 100)
        dat = data.copy()
        data = data[:rows]
        rows = int(row_count * int(validation_percentage) / 100)
        val_data = dat.iloc[(0-rows):]
#        print(len(dat))
        # build decision tree
        tree = Tree(train_file, test_file, model, training_percentage)
        tree.root = Node()
        tree.ID3_depth(tree.attributes[:-1].copy(), tree.root, data.copy(), max_depth)
#        tree.printTree(tree.root)
#        print(columns)
        # predict on testing set & evaluate the testing accuracy
        print('Train set accuracy: ' + str(tree.test(dat)))
        print('Validation set accuracy: ' + str(tree.test(val_data)))
        print('Test set accuracy: ' + str(tree.test(testDat)))

    elif model == 'min_split':
        training_percentage = sys.argv[4]
        validation_percentage = sys.argv[5]
        min_split = sys.argv[6]
        row_count = sum(1 for row in data.iterrows())
        rows = int(row_count * int(training_percentage) / 100)
        dat = data.copy()
        data = data[:rows]
        rows = int(row_count * int(validation_percentage) / 100)
        val_data = dat.iloc[(0-rows):]
        #        print(len(dat))
        # build decision tree
        tree = Tree(train_file, test_file, model, training_percentage)
        tree.root = Node()
        tree.ID3_min_split(tree.attributes[:-1].copy(), tree.root, data.copy(), min_split)
        #        tree.printTree(tree.root)
        #        print(columns)
        # predict on testing set & evaluate the testing accuracy
        print('Train set accuracy: ' + str(tree.test(dat)))
        print('Validation set accuracy: ' + str(tree.test(val_data)))
        print('Test set accuracy: ' + str(tree.test(testDat)))

    elif model == 'prune':
        training_percentage = sys.argv[4]
        row_count = sum(1 for row in data.iterrows())
        rows = int(row_count * int(training_percentage) / 100)
        data = data[:rows]
        
        # build decision tree
        tree = Tree(train_file, test_file, model, training_percentage)
        tree.root = Node()
        tree.ID3(tree.attributes[:-1].copy(), tree.root, data.copy())
        #        tree.printTree(tree.root)
        
        # predict on testing set & evaluate the testing accuracy
        print('Train set accuracy: ' + str(tree.test(dat)))
        print('Test set accuracy: ' + str(tree.test(testDat)))
