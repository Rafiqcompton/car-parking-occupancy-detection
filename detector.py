from skimage.color import rgb2gray
from skimage.transform import resize

from roi import roi
from classifier import BinaryImageClassificationData, SVM

class SheepDetector(object):

    def __init__(self, neg_dir, pos_dir):
        samples = BinaryImageClassificationData((neg_dir, 'background'), (pos_dir, 'sheep'))
        self.svmClassifier = SVM(samples)
        self.svmClassifier.train()
    
    def evaluate_classifier(self):
        return self.svmClassifier.evaluate()
    
    def detect(self, image, probability_threshold=0.9, threshold=200, padding=3, min_size=10, max_size=2000):
        detections = []

        rois = roi(image, threshold=threshold, padding=padding, min_size=min_size, max_size=max_size)

        for ((x, y), (w, h)) in rois:
            crop_img = image[y: y + h, x: x + w]
            resized_img = resize(crop_img, (20, 20))
            gray_image = rgb2gray(resized_img)
            img_array = gray_image.flatten()

            probability_of_sheep = self.svmClassifier.predict(img_array)['sheep']

            if probability_of_sheep > probability_threshold:
                detections.append(((x, y), (w, h), probability_of_sheep))
            
        return detections
