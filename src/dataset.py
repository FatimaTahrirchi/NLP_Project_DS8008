"""
Created on Tue Apr 20 01:54:02 2021

@author: fatemeh tahrirchi

"""

import os
import sys
import csv


csv.field_size_limit(sys.maxsize)
DATA_FOLDER = "datasets"

class YelpReview(object):
    """
    credit goes to Xiang Zhang:
    https://scholar.google.com/citations?hl=en&user=n4QjVfoAAAAJ&view_op=list_works&sortby=pubdate
    """
    def __init__(self):
        self.data_name = 'yelp_review_full'
        
        self.data_folder = "{}/{}".format(DATA_FOLDER, self.data_name)
        self.n_classes = 5

    def _generator(self, file_name):
        DataPath=os.path.join(self.data_folder, file_name)
        with open(DataPath, mode='r', encoding='utf-8') as f:
            reader = csv.DictReader(f, fieldnames=['label', 'title', 'description'], quotechar='"')
            for i,line in enumerate(reader):
                #if(i%10==0):
                    sentence = "{} {}".format(line['title'], line['description'])
                    label = int(line['label']) - 1
                    yield sentence, label

    def load_train_data(self):
        return self._generator("train.csv")

    def load_test_data(self):
        return self._generator("test.csv")

class YelpPolarity(object):
    """
    credit goes to Xiang Zhang:
    https://scholar.google.com/citations?hl=en&user=n4QjVfoAAAAJ&view_op=list_works&sortby=pubdate
    """
    def __init__(self):

        self.data_name ='yelp_review_polarity'
        self.data_folder = "{}/{}".format(DATA_FOLDER, self.data_name)
        self.n_classes = 2        

    def _generator(self, file_name):
        DataPath=os.path.join(self.data_folder, file_name)
        with open(DataPath, mode='r', encoding='utf-8') as f:
            reader = csv.DictReader(f, fieldnames=['label', 'title', 'description'], quotechar='"')
            for i,line in enumerate(reader):
                #if(i%10==0):     #if you want use part of datasets
                    sentence = "{} {}".format(line['title'], line['description'])
                    label = int(line['label']) - 1
                    yield sentence, label

    def load_train_data(self):
        return self._generator("train.csv")

    def load_test_data(self):
        return self._generator("test.csv")
    
def load_datasets(name="yelp_review_polarity"):
    dataset =None
    if name=='yelp_review_full':
        dataset=YelpReview()
    if name=='yelp_review_polarity':
        dataset=YelpPolarity()
    return dataset
