#encoding:utf-8
class Talents:
    def __init__(self, code_segment, feature_segment, estimator, c_accuracy, c_accuracies):
        self.code_segment = code_segment
        self.feature_segment = feature_segment
        self.estimator = estimator
        self.segment_accuracy = c_accuracy#混淆矩阵准确率
        self.c_accuracies = c_accuracies#混淆矩阵每类准确率