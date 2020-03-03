#encoding:utf-8
class ThreeMatrix:
    def __init__(self,coding_matrix,feature_matrix,base_learner_matrix, f1score, distance, segment_accuracies,segment_confusion_matrixs):
        self.coding_matrix = coding_matrix
        self.feature_matrix = feature_matrix
        self.base_learner_matrix = base_learner_matrix
        self.f1score = f1score
        self.distance = distance
        self.segment_accuracies = segment_accuracies
        self.segment_confusion_matrixs = segment_confusion_matrixs
