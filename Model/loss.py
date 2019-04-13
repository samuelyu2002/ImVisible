import torch.nn as nn

def my_loss(classifier, regression, points, mode):
    #classifier is the predicted class
    #regression is an array of predicted coordinates
    #points is an array of ground truth coordinates
    #mode is the ground truth class
    alpha = 0.5
    MSE = nn.MSELoss()
    MSEl = MSE(regression, points)
    cross_entropy = nn.CrossEntropyLoss()
    ce = cross_entropy(classifier, mode)
    
    loss = ce*alpha + MSEl*(1-alpha)
    return loss, MSEl, ce
