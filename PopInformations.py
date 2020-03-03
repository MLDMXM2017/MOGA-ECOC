# coding: utf-8
class PopInfors:
    def __init__(self, pop_size, class_size, feature_size, num_classifier, maximum_iteration, train_x, train_y, test_x, test_y,validate_x,validate_y,base_learners):
        self.pop_size = pop_size
        self.class_size = class_size
        self.feature_size = feature_size
        self.num_classifier = num_classifier
        self.maximum_iteration = maximum_iteration
        self.train_x = train_x
        self.train_y = train_y
        self.test_x = test_x
        self.test_y = test_y
        self.validate_x = validate_x
        self.validate_y = validate_y
        self.base_learners = base_learners
        
