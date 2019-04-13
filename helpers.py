import math
import matplotlib.pyplot as plt

def angle_difference(pred, label):
    #pred is the array of coordinate predictions
    #label is the array of ground truth coordinates
    
    #get all 4 coordinates from pred array
    pred_x1 = pred[0][0]
    pred_x2 = pred[0][2]
    pred_y1 = pred[0][1]
    pred_y2 = pred[0][3]
    
    pred_x_distance = pred_x1 - pred_x2 #distance between 2 x-coordinates
    pred_y_distance = pred_y2 - pred_y1 #distance between 2 y-coordinates

    pred_angle = math.atan2(pred_y_distance, pred_x_distance) #predicted angle between direction vector and x-axis
    
    #get all 4 coordinates from label array
    act_x1 = label[0][0]
    act_x2 = label[0][2]
    act_y1 = label[0][1]
    act_y2 = label[0][3]
    
    act_x_distance = act_x1 - act_x2
    act_y_distance = act_y2 - act_y1
    
    actual_angle = math.atan2(act_y_distance, act_x_distance)
    
    return (pred_angle - actual_angle)*180/math.pi #returns difference between predicted and ground truth angles
    

def startpoint_difference(pred, label):
    #the startpoint will always be the second set of coordinates in the pred and label array
    
    x_distance = pred[0][2] - label[0][2]
    y_distance = pred[0][3] - label[0][3]
    
    #distance between predicted and ground truth startpoints
    distance = math.sqrt(x_distance*x_distance + y_distance*y_distance)
    
    return distance

def endpoint_difference(pred, label):
    #the endpoint will always be the first set of coordinates in the pred and label array
    
    x_distance = pred[0][0] - label[0][0]
    y_distance = pred[0][1] - label[0][1]
    
    #distance between predicted and ground truth endpoints
    distance = math.sqrt(x_distance*x_distance + y_distance*y_distance)
    return distance

def direction_performance(pred, label):
    pred = pred.cpu().detach().numpy()
    label = label.cpu().detach().numpy()
    pred = pred.tolist()
    label = label.tolist()
    
    #gets the absolute error
    angle = math.fabs(angle_difference(pred,label))
    start = math.fabs(startpoint_difference(pred,label))
    end = math.fabs(endpoint_difference(pred,label))
    
    return angle, start, end

def display_image(image, title, points_pred, points_gt, factor):
    #factor is used to convert the coordinates from between [0,1] to desired image coordinates
    
    plt.imshow(image)
    plt.title(title)
    
    #plots predicted coordinates
    plt.scatter([points_pred[0]*factor*4,points_pred[2]*factor*4],[points_pred[1]*factor*3,points_pred[3]*factor*3], c = 'r')
    #plots ground truth coordinates
    plt.scatter([points_gt[0]*factor*4,points_gt[2]*factor*4],[points_gt[1]*factor*3,points_gt[3]*factor*3], c = 'b')
    plt.show()