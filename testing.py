import torch
from torch.utils.data import DataLoader
from network import LYTNet
from loss import my_loss
from helpers import direction_performance, display_image
from dataset import TrafficLightDataset
import matplotlib.pyplot as plt
import numpy as np

cuda_available = torch.cuda.is_available()

test_file_loc = 'testing_file.csv'
test_image_directory = 'PTL_Dataset_768x576'
MODEL_PATH = ''

dataset = TrafficLightDataset(csv_file = test_file_loc, root_dir = test_image_directory)
dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2)

net = LYTNet()
checkpoint = torch.load(MODEL_PATH)
net.load_state_dict(checkpoint)
net.eval()

if cuda_available:
    net = net.cuda()

loss_fn = my_loss

#storing data
running_loss = 0
running_test_angle = 0
running_test_start = 0
running_test_end = 0

#errors when zebra crossing is blocked
running_angle_block = 0
running_start_block = 0
running_end_block = 0
block_count = 0

#errors when zebra crossing is unblocked
running_angle_unblock = 0
running_start_unblock = 0
running_end_unblock = 0
unblock_count = 0

total = 0
correct = 0

tp = {'0':0, '1':0, '2':0, '3':0, '4':0}
fp = {'0':0, '1':0, '2':0, '3':0, '4':0}
fn = {'0':0, '1':0, '2':0, '3':0, '4':0}
classes = {'0':'red', '1':'green', '2':'none', '3':'countdown_blank', '4':'countdown_green'}
precisions = []
recalls = []

with torch.no_grad():
    
    for i, data in enumerate(dataloader):
        
        images = data['image'].type(torch.FloatTensor)
        mode = data['mode']
        points = data['points']
        blocked = data['block'] #tag for blocked zebra crossing
        
        if cuda_available:
            images = images.cuda()
            mode = mode.cuda()
            points = points.cuda()
 
        pred_classes, pred_direc = net(images)
        _, predicted = torch.max(pred_classes, 1)
        
        #correct prediction
        if (predicted == mode).sum().item() == 1:
            correct += 1
            tp[str(predicted.cpu().numpy()[0])] += 1
        
        #incorrect prediction
        if (predicted == mode).sum().item() == 0:
            predicted_idx = str(predicted.cpu().numpy()[0])
            gt_idx = str(mode.cpu().numpy()[0])
            fp[predicted_idx] += 1
            fn[gt_idx] += 1
            
            #display image when incorrect
            image = images.cpu().numpy()[0]
            image = np.transpose(image, (1,2,0))
            image = image.astype(int)
            
            title = 'predicted: ' + classes[predicted_idx] + ' ground_truth: ' + classes[gt_idx] + ' ' + str(i+1)            
            ax = plt.subplot()
            ax.axis('on')
            pred_points = pred_direc.cpu().detach().numpy()[0].tolist()
            gt_points = points.cpu().detach().numpy()[0]
            
            display_image(image,title,pred_points, gt_points, 192) #factor is 192 because 4*192 = 768

        loss, MSE, cross_entropy =  loss_fn(pred_classes, pred_direc, points, mode)
        running_loss += loss
        angle, start, end = direction_performance(pred_direc, points)
        
        if(blocked[0] == "blocked"):
            running_angle_block += angle
            running_start_block += start
            running_end_block += end
            block_count += 1
            
        else:
            running_angle_unblock += angle
            running_start_unblock += start
            running_end_unblock += end
            unblock_count += 1
               
        running_test_angle += angle
        running_test_start += start
        running_test_end += end
        total += 1
        

try:red_precision = tp['0']/(tp['0'] + fp['0'])
except: red_precision = 0
precisions.append(red_precision)
try: red_recall = tp['0']/(tp['0'] + fn['0'])
except: red_recall = 0
recalls.append(red_recall)
            
try: green_precision = tp['1']/(tp['1'] + fp['1'])
except: green_precision = 0
precisions.append(green_precision)
try: green_recall = tp['1']/(tp['1'] + fn['1'])
except: green_recall = 0
recalls.append(green_recall)
            
try: countdown_green_precision = tp['2']/(tp['2'] + fp['2'])
except: countdown_green_precision = 0
precisions.append(countdown_green_precision)
try: countdown_green_recall = tp['2']/(tp['2'] + fn['2'])
except: countdown_green_recall = 0
recalls.append(countdown_green_recall)
            
try: countdown_blank_precision = tp['3']/(tp['3'] + fp['3'])
except: countdown_blank_precision = 0
precisions.append(countdown_blank_precision)
try: countdown_blank_recall = tp['3']/(tp['3'] + fn['3'])
except: countdown_blank_recall = 0
recalls.append(countdown_blank_recall)
            
try: blank_precision = tp['4']/(tp['4'] + fp['4']) 
except: blank_precision = 0
precisions.append(blank_precision)
try: blank_recall = tp['4']/(tp['4'] + fn['4'])
except: blank_recall = 0
recalls.append(blank_recall)
            
print("Average loss: " + str(running_loss/total))
print("Average angle error: " + str(running_test_angle/total))
print("Average startpoint error: " + str(running_test_start/total))
print("Average endpoint error: " + str(running_test_end/total))
print("Blocked angle error: " + str(running_angle_block/block_count))
print("Blocked startpoint error: " + str(running_start_block/block_count))
print("Blocked endpoint error: " + str(running_end_block/block_count))
print("Unblocked angle error: " + str(running_angle_unblock/unblock_count))
print("Unblocked startpoint error: " + str(running_start_unblock/unblock_count))
print("Unblocked endpoint error: " + str(running_end_unblock/unblock_count))
print("Accuracy: " + str(correct/total*100))

print("Precisions: " + precisions)
print("Recalls: " + recalls)