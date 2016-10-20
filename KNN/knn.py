import argparse
from collections import Counter, defaultdict

import random
import pickle
import numpy as np
from numpy import median
from numpy import array
from sklearn.neighbors import BallTree

class Numbers:
    """
    Class to store MNIST data
    """

    def __init__(self, location):
        # You shouldn't have to modify this class, but you can if
        # you'd like.

        import cPickle, gzip

        # Load the dataset
        f = gzip.open(location, 'rb')
        train_set, valid_set, test_set = cPickle.load(f)

        self.train_x, self.train_y = train_set
        self.test_x, self.test_y = valid_set
        f.close()


class Knearest:
    """
    kNN classifier
    """

    def __init__(self, x, y, k):
        """
        Creates a kNN instance
        :param x: Training data input
        :param y: Training data output
        :param k: The number of nearest points to consider in classification
        """

        # You can modify the constructor, but you shouldn't need to.
        # Do not use another data structure from anywhere else to
        # complete the assignment.
        
        self._kdtree = BallTree(x)
        self._y = y
        self._k = k

    def majority(self, item_indices):
        """
        Given the indices of training examples, return the majority label.  If
        there's a tie, return the median of the majority labels (as implemented 
        in numpy).
        :param item_indices: The indices of the k nearest neighbors
        """
        flag=0
        for j in range(0,self._k):
            if(self._y[item_indices[j]]==-1):
                flag=1
                break;
        if(flag==1):
            labels=2
            l=list()
            arr= np.zeros((labels,), dtype=int)
            assert len(item_indices) == self._k, "Did not get k inputs"

        
            for i in range(0,self._k):
                if self._y[item_indices[i]]==1:
                    arr[0]+=1;
                else:
                    arr[1]+=1
 
                
            val=max(arr)
            if(arr[0]==arr[1]):
                r=0;
            elif(arr[0]==val):
                r=1;
            else:
                r=-1;
            
        else:
            labels=10
            l=list()
            arr= np.zeros((labels,), dtype=int)
            assert len(item_indices) == self._k, "Did not get k inputs"

            for i in range(0,self._k):
                arr[self._y[item_indices[i]]]= arr[self._y[item_indices[i]]]+1
            
            val=max(arr)
            for j in range(0,labels):
                if(arr[j]==val):
                    l.append(j)
            
            r=median(l)
        

        return r

    def classify(self, example):
        """
        Given an example, classify the example.
        :param example: A representation of an example in the same
        format as training data
        """
        
        # Finish this function to find the k closest points, query the
        # majority function, and return the predicted label.
        # Again, the current return value is a placeholder 
        # and definitely needs to be changed. 

        dist,ind = self._kdtree.query(example,self._k)
        
        return self.majority(ind.flatten().tolist())


    def confusion_matrix(self, test_x, test_y):
        """
        Given a matrix of test examples and labels, compute the confusion
        matrix for the current classifier.  Should return a dictionary of
        dictionaries where d[ii][jj] is the number of times an example
        with true label ii was labeled as jj.
        :param test_x: Test data representation
        :param test_y: Test data answers
        """

        # Finish this function to build a dictionary with the
        # mislabeled examples.  You'll need to call the classify
        # function for each example.
      
        data_index = 0
        d={0:{0:0,1:0,2:0,3:0,4:0,5:0,6:0,7:0,8:0,9:0},1:{0:0,1:0,2:0,3:0,4:0,5:0,6:0,7:0,8:0,9:0},2:{0:0,1:0,2:0,3:0,4:0,5:0,6:0,7:0,8:0,9:0},3:{0:0,1:0,2:0,3:0,4:0,5:0,6:0,7:0,8:0,9:0},4:{0:0,1:0,2:0,3:0,4:0,5:0,6:0,7:0,8:0,9:0},5:{0:0,1:0,2:0,3:0,4:0,5:0,6:0,7:0,8:0,9:0},6:{0:0,1:0,2:0,3:0,4:0,5:0,6:0,7:0,8:0,9:0},7:{0:0,1:0,2:0,3:0,4:0,5:0,6:0,7:0,8:0,9:0},8:{0:0,1:0,2:0,3:0,4:0,5:0,6:0,7:0,8:0,9:0},9:{0:0,1:0,2:0,3:0,4:0,5:0,6:0,7:0,8:0,9:0}}
        
        for ii in range(0,len(test_x)):
            ret_label=self.classify(test_x[ii])
            d[test_y[ii]][int(ret_label)]=d[test_y[ii]][int(ret_label)]+1
            
        
        for xx, yy in zip(test_x, test_y):
            data_index += 1
            if data_index % 100 == 0:
                print("%i/%i for confusion matrix" % (data_index, len(test_x)))
        return d

    @staticmethod
    def accuracy(confusion_matrix):
        """
        Given a confusion matrix, compute the accuracy of the underlying classifier.
        """

        # You do not need to modify this function

        total = 0
        correct = 0
        for ii in confusion_matrix:
            total += sum(confusion_matrix[ii].values())
            correct += confusion_matrix[ii].get(ii, 0)

        if total:
            return float(correct) / float(total)
        else:
            return 0.0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='KNN classifier options')
    parser.add_argument('--k', type=int, default=3,
                        help="Number of nearest points to use")
    parser.add_argument('--limit', type=int, default=-1,
                        help="Restrict training to this many examples")
    args = parser.parse_args()

    data = Numbers("../data/mnist.pkl.gz")

    # You should not have to modify any of this code

    if args.limit > 0:
        print("Data limit: %i" % args.limit)
        knn = Knearest(data.train_x[:args.limit], data.train_y[:args.limit],
                       args.k)
    else:
        knn = Knearest(data.train_x, data.train_y, args.k)
    print("Done loading data")

    confusion = knn.confusion_matrix(data.test_x, data.test_y)
    print("\t" + "\t".join(str(x) for x in xrange(10)))
    print("".join(["-"] * 90))
    for ii in xrange(10):
        print("%i:\t" % ii + "\t".join(str(confusion[ii].get(x, 0))
                                       for x in xrange(10)))
    print("Accuracy: %f" % knn.accuracy(confusion))
