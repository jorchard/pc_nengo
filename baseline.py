import numpy as np


class BaselineClassifier:
    """
    Implements a suite of simple classifiers that can be used as a baseline measuring classification power of other models.
    
    Classifiers to use:
        - Random Classifier
        - Zero Rate Classifier
        - Uniform Classifier
    
    Assumes one-hot encoding is used for multi-class classification.
    """

    def __init__(self):
        self.binary = None #boolean, true if the problem is binary classification
        self.most_common_label = None #the most common label 
        self.num_labels = None #the number of labels when doing multi-class classification
        self.dist = None #the empirical distribution of labels in the training set
        self.labels = None #all the possible labels


    def fit(self, X, y):
        #if y is a vector this is a binary classification problem
        self.binary = len(y.shape) == 1

        if not self.binary: #how many labels are there?
            self.num_labels = y.shape[1] #works with one-hot encoding
        else:
            self.num_labels = 2

        #extract relavent information
        if self.binary: #binary classification
            labs, counts = np.unique(y, return_counts=True)

            #find most common label
            self.most_common_label = labs[np.argmax(counts)]

            #find label distribution
            self.dist = counts/np.sum(counts)

            #save the labels
            self.labels = labs
        else: #multi-class classification
            labs, counts = np.unique(y, return_counts=True, axis=0)
            
            #find most common label
            self.most_common_label = labs[np.argmax(counts),:]
            
            #find label distribution
            self.dist = counts/np.sum(counts)

            #save the labels
            self.labels = labs       
        

    def random_classifier_predict(self, x_test):
        """
        Randomly predicts the classes with probabilities determined by the relative frequencies
        of the labels in the training set.
        """
        guess_indices = np.random.choice(np.arange(self.num_labels), x_test.shape[0], p=self.dist)

        if self.binary:
            return self.labels[guess_indices]
        else:
            return self.labels[guess_indices, :]

    def zero_rate_classifier_predict(self, x_test):
        """
        Always predicts the most frequent labels in the training set.
        """
        return np.array([self.most_common_label for i in range(x_test.shape[0])])

    def uniform_classifier_predict(self, x_test):
        """
        Randomly predicts the classes with uniform probability.
        """
        guess_indices = np.random.randint(0, self.num_labels, x_test.shape[0])

        if self.binary:
            return self.labels[guess_indices]
        else:
            return self.labels[guess_indices, :]
    

    def empirical_accuracy(self, x_test, y_test, N=10):
        """
        Estimates the empirical accuracy of each of the classification methods.
        To do this, we run the classifier on the x_test set N times to get predictions.
        We compare these predictions to y_test.
        """
        samples = x_test.shape[0] #the number of test samples

        random_acc = 0 #accuracy of the random classifier
        uniform_acc = 0 #accuracy of the uniform classifier
        zero_acc = 0 #accuracy of the zero rate classifier

        for n in range(N): #epochs
            #random classifier
            random_pred = self.random_classifier_predict(x_test)

            if self.binary: #binary classification
                random_acc += np.sum(np.equal(random_pred, y_test))/samples
            else: #multi-class classification
                comparison = np.equal(random_pred, y_test)
                random_acc += np.sum(np.all(comparison, axis=1))/samples

            #zero rate classifier
            zero_pred = self.zero_rate_classifier_predict(x_test)

            if self.binary: #binary classification
                zero_acc += np.sum(np.equal(zero_pred, y_test))/samples
            else: #multi-class classification
                comparison = np.equal(zero_pred, y_test)
                zero_acc += np.sum(np.all(comparison, axis=1))/samples

            #uniform classifier
            uniform_pred = self.uniform_classifier_predict(x_test)

            if self.binary: #binary classification
                uniform_acc += np.sum(np.equal(uniform_pred, y_test))/samples
            else: #multi-class classification
                comparison = np.equal(uniform_pred, y_test)
                uniform_acc += np.sum(np.all(comparison, axis=1))/samples

        
        return random_acc/N, uniform_acc/N, zero_acc/N





if __name__ == "__main__":
    #run tests


    print("50/50 Dataset")

    X = np.arange(100)
    y = np.append(np.ones(50), np.zeros(50))

    base = BaselineClassifier()
    base.fit(X, y)

    random_acc, uniform_acc, zero_acc = base.empirical_accuracy(X, y, N=100)

    print(f"Random accuracy = {random_acc}. Should be ~0.5")
    print(f"Uniform accuracy = {uniform_acc}. Should be ~0.5")
    print(f"Zero rate accuracy = {zero_acc}. Should be ~0.5")


    print("\n25/25/25/25 Dataset")
    X = np.arange(100)
    y = []
    for i in range(25):
        y.append([1, 0, 0, 0])
        y.append([0, 1, 0, 0])
        y.append([0, 0, 1, 0])
        y.append([0, 0, 0, 1])
    y = np.array(y)

    base = BaselineClassifier()
    base.fit(X, y)

    random_acc, uniform_acc, zero_acc = base.empirical_accuracy(X, y, N=100)

    print(f"Random accuracy = {random_acc}. Should be ~0.25")
    print(f"Uniform accuracy = {uniform_acc}. Should be ~0.25")
    print(f"Zero rate accuracy = {zero_acc}. Should be ~0.25")

    print("\n90/10 Dataset")
    X = np.arange(100)
    y = []
    for i in range(90):
        y.append(1)
    for i in range(10):
        y.append(0)
    y = np.array(y)

    base = BaselineClassifier()
    base.fit(X, y)

    random_acc, uniform_acc, zero_acc = base.empirical_accuracy(X, y, N=100)

    print(f"Random accuracy = {random_acc}. Should be ~0.82")
    print(f"Uniform accuracy = {uniform_acc}. Should be ~0.5")
    print(f"Zero rate accuracy = {zero_acc}. Should be ~0.90")

    