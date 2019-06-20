import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from network_no_blank import LYTNet
from dataset import TrafficLightDataset
from loss import my_loss
from helpers import direction_performance

cuda_available = torch.cuda.is_available()

BATCH_SIZE = 32
MAX_EPOCHS = 800
INIT_LR = 0.001
WEIGHT_DECAY = 0.00005
LR_DROP_MILESTONES = [150,400,650]

train_file_dir = 'training_file.csv'
valid_file_dir = 'validation_file.csv'
train_img_dir = 'PTL_Dataset_876x657'
valid_img_dir = 'PTL_Dataset_768x576'
save_path = ''

train_dataset = TrafficLightDataset(csv_file = train_file_dir, img_dir = train_img_dir)
valid_dataset = TrafficLightDataset(csv_file = valid_file_dir, img_dir = valid_img_dir)

train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=8)
valid_dataloader = DataLoader(valid_dataset, batch_size=1, shuffle=False, num_workers=2)

net = LYTNet()

if cuda_available:
    net = net.cuda()

loss_fn = my_loss

optimizer = torch.optim.Adam(net.parameters(), lr = INIT_LR, weight_decay = 0.000005 )
lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, LR_DROP_MILESTONES)

#storing all data during training
train_losses = [] #stores the overall training loss at each epoch
train_MSE = [] #stores the MSE loss during training at each epoch
train_CE = [] #stores the cross entropy loss during training at each epoch
valid_losses = [] #stores the overall validation loss at each epoch
valid_MSE = [] #stores the MSE loss during validation at each epoch
valid_CE = [] #stores the cross entropy loss during validation at each epoch
train_accuracies = [] #stores the training accuracy of the network at each epoch
valid_accuracies = [] #stores the validation accuracy of the network at each epoch
val_angles = [] #stores the average angle error of the network during validation at each epoch
val_start = [] #stores the average startpoint error of the network during validation at each epoch
val_end = [] #stores the average endpoint error of the network during validation at each epoch


for epoch in range(MAX_EPOCHS):
    
    ##########
    #TRAINING#
    ########## 
    
    net.train()
    
    running_loss = 0.0 #stores the total loss for the epoch
    running_loss_MSE = 0.0 #stores the total MSE loss for the epoch
    running_loss_cross_entropy = 0.0 #store the total cross entropy loss for the epoch
    angle_error = 0.0 #stores average angle error for the epoch
    startpoint_error = 0.0 #stores average startpoint error for the epoch
    endpoint_error = 0.0 #stores average endpoint error for the epoch
    train_correct = 0 #stores total number of correctly predicted images during training for the epcoh
    train_total = 0 #stores total number of batches processed at each epoch
    
    for j, data in enumerate(train_dataloader, 0): 
        
        optimizer.zero_grad()
        train_total += 1
        
        images = data['image'].type(torch.FloatTensor)
        mode = data['mode'] #index of traffic light mode
        points = data['points'] #array of midline coordinates
        
        if cuda_available:
            images = images.cuda()
            mode = mode.cuda()
            points = points.cuda()
        
        pred_classes, pred_direc = net(images)
        _, predicted = torch.max(pred_classes, 1) #finds index of largest probability
        train_correct += (predicted == mode).sum().item() #increments train_correct if predicted index is correct
        loss, MSE, cross_entropy = loss_fn(pred_classes, pred_direc, points, mode)
        angle, start, end = direction_performance(pred_direc, points)
        angle_error += angle
        endpoint_error += end
        startpoint_error += start
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss
        running_loss_MSE += MSE
        running_loss_cross_entropy += cross_entropy

    print('Epoch: ' + str(epoch+1))
    print('Average training loss: ' + str(running_loss/(j+1)))
    print('Average training MSE loss: ' + str(running_loss_MSE/(j+1)))
    print('Average training cross entropy loss: ' + str(running_loss_cross_entropy/(j+1)))
    print('Training accuracy: ' + str(train_correct/train_total/BATCH_SIZE))

    train_MSE.append(running_loss_MSE/train_total)
    train_CE.append(running_loss_cross_entropy/train_total)
    train_losses.append(running_loss/train_total) 
    train_accuracies.append(train_correct/train_total/32*100) 
            
    lr_scheduler.step(epoch + 1) #decrease learning rate if at desired epoch   
    
    ############
    #VALIDATION#
    ############ 
    
    net.eval()
    
    tp = {'0':0, '1':0, '2':0, '3':0, '4':0} #stores number of true positives for each class
    fp = {'0':0, '1':0, '2':0, '3':0, '4':0} #stores number of false positives for each class
    fn = {'0':0, '1':0, '2':0, '3':0, '4':0} #stores number of false negatives for each class
    
    precisions = [] #stores the precision for each class
    recalls = [] #stores the recall for each class
    
    #stores losses and errors for network during validation
    val_running_loss = 0
    val_mse_loss = 0
    val_ce_loss = 0
    val_angle_error = 0
    val_start_error = 0
    val_end_error = 0
    val_correct = 0
    val_total = 0
    
    with torch.no_grad():
        
        for k, data in enumerate(valid_dataloader, 0):
            
            images = data['image'].type(torch.FloatTensor)
            mode = data['mode']
            points = data['points']
            
            if cuda_available:
                images = images.cuda()
                mode = mode.cuda()
                points = points.cuda()
            
            pred_classes, pred_direc = net(images)
            _, predicted = torch.max(pred_classes, 1)
            val_correct += (predicted == mode).sum().item()
            val_total += 1
            
            #incorrect prediction
            if (predicted == mode).sum().item() == 0:
                fp[str(predicted.cpu().numpy()[0])] += 1 #increments predicted class's false positive count by one
                fn[str(mode.cpu().numpy()[0])] += 1 #increments correct class's false negative count by one
            
            #correct prediction
            if (predicted == mode).sum().item() == 1: 
                tp[str(predicted.cpu().numpy()[0])] += 1 #increments correct class's true positive count by one
            
            loss, MSE, cross_entropy = loss_fn(pred_classes, pred_direc, points, mode)
            val_running_loss += loss
            val_mse_loss += MSE
            val_ce_loss += cross_entropy
            
            angle, start, end = direction_performance(pred_direc, points)
            val_angle_error += angle
            val_start_error += start
            val_end_error += end
            
        #calculates precision and recalls for each class given fp, tp, fn
        #try excepts are used to prevent division by zero errors
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
        
        try: none_precision = tp['4']/(tp['4'] + fp['4']) 
        except: none_precision = 0
        precisions.append(none_precision)
        try: none_recall = tp['4']/(tp['4'] + fn['4'])
        except: none_recall = 0
        recalls.append(none_recall)
        
        print("Average validation loss: " + str(val_running_loss/val_total))
        print("Average validation MSE loss: " + str(val_mse_loss/val_total))
        print("Average validation cross entropy loss: " + str(val_ce_loss/val_total))
        print("Validation accuracy: " + str(100*val_correct/val_total))
        
        valid_accuracies.append(100*val_correct/val_total)
        valid_losses.append(val_running_loss/val_total)
        valid_MSE.append(val_mse_loss/val_total)
        valid_CE.append(val_ce_loss/val_total)
        
        print("Precisions: " + str(precisions))
        print("Recalls: " + str(recalls))
        print("Angle Error: " + str(val_angle_error/val_total))
        print("Startpoint Error: " + str(val_start_error/val_total))
        print("Endpoint Error: " + str(val_end_error/val_total))
        
        val_angles.append(val_angle_error/val_total)
        val_start.append(val_start_error/val_total)
        val_end.append(val_end_error/val_total)
        
        #graphs average losses every epoch_num of epochs
        epoch_num = 100
        if epoch % epoch_num == (epoch_num - 1):
            plt.title('Train and Validation losses')
            plt.plot(train_losses)
            plt.plot(valid_losses)
            plt.show()
        
        #stores the network and optimizer weights at the 50th epoch
        if epoch == 50:
            states = {
                    'epoch': epoch+1,
                    'state_dict': net.state_dict(),
                    'optimizer': optimizer.state_dict()
                    }
            torch.save(states, save_path + 'epoch_50')
            
        #stores the network and optimizer weights every 200th epoch
        if epoch%200 == 199:
            states = {
                    'epoch': epoch+1,
                    'state_dict': net.state_dict(),
                    'optimizer': optimizer.state_dict()
                    }
            torch.save(states, save_path + '_epoch_' + str(epoch+1))
            
            
################
#AFTER TRAINING#
################

#plots training and validation loss
plt.title('train and validation loss')
plt.plot(valid_losses)
plt.plot(train_losses)
plt.savefig(save_path + '_losses')
plt.show()

#plots training and validation cross entropy loss
plt.title('train and valid cross entropy')
plt.plot(train_CE)
plt.plot(valid_CE)
plt.savefig(save_path + 'train_valid_ce')
plt.show()

#plots training and validation MSE loss
plt.title('train and valid MSE')
plt.plot(train_MSE)
plt.plot(valid_MSE)
plt.savefig(save_path + 'train_valid_MSE')
plt.show()

#plots training and validation accuracies
plt.title('train and validation accuracies')
plt.plot(valid_accuracies)
plt.plot(train_accuracies)
plt.savefig(save_path + '_accuracies')
plt.show()

#save final network weights
torch.save(net.state_dict(), save_path + '_final_weights')
