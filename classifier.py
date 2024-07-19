import random

from sklearn.svm import SVC
from sklearn.externals import joblib

from utils import load_images


class BinaryImageClassificationData(object):

    class Data:
        def __init__(self, data, labels):
            self.data = data
            self.labels = labels

    def __init__(self, class1, class2, resize_dim=(20, 20), test_ratio=0.3, seed=42):
        """Load the data. Classes are tuples on the form: (path, name), e.g. ('/some/dir/of/cats', 'cat')"""
        
        random.seed(seed) # Ensures the same result every time

        # Load images
        class1_images = load_images(class1[0], resize_dim=resize_dim)
        class2_images = load_images(class2[0], resize_dim=resize_dim)

        # Create labels; class1 = 0, class2 = 1
        labels = [0] * len(class1_images) + [1] * len(class2_images)

        # Create samples by combining the two classes (in the same order as the labels)
        samples = class1_images + class2_images
        
        # Shffle samples and labels
        tmp = list(zip(samples, labels))
        random.shuffle(tmp)
        samples, labels = zip(*tmp)
        
        # Number of samples for testing
        num_test = int(len(samples) * test_ratio)

        # Create the test data
        self.test = self.Data(samples[:num_test], labels[:num_test])
        # Create the train data
        self.train = self.Data(samples[num_test:], labels[num_test:])
        # Store the names of the different classes such that names[0] = name of class1, names[1] = name of class2
        self.names = [class1[1], class2[1]]


class SVM(object):

    def __init__(self, samples, load_from_file=False, seed=42):
        assert type(samples) is BinaryImageClassificationData, 'Samples must be of type `BinaryImageClassificationData`'

        self.samples = samples

        if load_from_file:
            self.load(load_from_file)
        else:
            self.svm = SVC(probability=True, random_state=seed)
    
    def predict(self, image):
        return {self.samples.names[i]: p for i, p in enumerate(self.svm.predict_proba([image])[0])}

    def predict_multiple(self, images):
        return [{self.samples.names[i]: p for i, p in enumerate(prediction)} for prediction in self.svm.predict_proba(images)]
    
    def train(self):
        self.svm.fit(self.samples.train.data, self.samples.train.labels)
    
    def evaluate(self):
        return self.samples.test.labels, self.svm.predict(self.samples.test.data)
    
    def save(self, name='model'):
        joblib.dump(self.svm, '{}.pkl'.format(name))
    
    def load(self, name='model'):
        self.svm = joblib.load('{}.pkl'.format(name))
