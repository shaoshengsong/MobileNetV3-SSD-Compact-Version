import time
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
from model import SSD300, MultiBoxLoss
from datasets import PascalVOCDataset
from utils import *
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Data parameters
data_folder = './'  # folder with data files
keep_difficult = True  # use objects considered difficult to detect?

# Model parameters
# Not too many here since the SSD300 has a very specific structure
n_classes = len(label_map)  # number of different types of objects
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Learning parameters
checkpoint =None #  path to model checkpoint, None if none
batch_size = 32  # batch size 
# iterations = 120000  # number of iterations to train  120000
workers = 8  # number of workers for loading data in the DataLoader 4
print_freq = 200  # print training status every __ batches
lr =1e-3  # learning rate
#decay_lr_to = 0.1  # decay learning rate to this fraction of the existing learning rate
momentum = 0.9  # momentum
weight_decay = 5e-4  # weight decay
grad_clip = None  # clip if gradients are exploding, which may happen at larger batch sizes (sometimes at 32) - you will recognize it by a sorting error in the MuliBox loss calculation

cudnn.benchmark = True


def main():
    """
    Training.
    """
    global start_epoch, label_map, epoch, checkpoint, decay_lr_at

    # Initialize model or load checkpoint
    if checkpoint is None:
        print("checkpoint none")
        start_epoch = 0
        model = SSD300(n_classes=n_classes)

        # Initialize the optimizer, with twice the default learning rate for biases, as in the original Caffe repo
        biases = list()
        not_biases = list()
        for param_name, param in model.named_parameters():
            if param.requires_grad:
                if param_name.endswith('.bias'):
                    biases.append(param)
                else:
                    not_biases.append(param)

        # differnet optimizer           
        # optimizer = torch.optim.SGD(params=[{'params': biases, 'lr': 2 * lr}, {'params': not_biases}],
        #                             lr=lr, momentum=momentum, weight_decay=weight_decay)
        optimizer = torch.optim.SGD(params=[{'params': biases, 'lr':  lr}, {'params': not_biases}],
                                    lr=lr, momentum=momentum, weight_decay=weight_decay)                            

        #optimizer = torch.optim.SGD(params=[{'params':model.parameters(), 'lr': 2 * lr}, {'params': model.parameters}],  lr=lr, momentum=momentum, weight_decay=weight_decay) 


    else:
        print("checkpoint load")
        checkpoint = torch.load(checkpoint)
        start_epoch = checkpoint['epoch'] + 1
        print('\nLoaded checkpoint from epoch %d.\n' % start_epoch)
        model = checkpoint['model']
        optimizer = checkpoint['optimizer']


    

    # Move to default device
    model = model.to(device)
    criterion = MultiBoxLoss(priors_cxcy=model.priors_cxcy).to(device)

    # Custom dataloaders
    train_dataset = PascalVOCDataset(data_folder,
                                     split='train',
                                     keep_difficult=keep_difficult)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                               collate_fn=train_dataset.collate_fn, num_workers=workers,
                                               pin_memory=True)  # note that we're passing the collate function here

    # Calculate total number of epochs to train and the epochs to decay learning rate at (i.e. convert iterations to epochs)
    # To convert iterations to epochs, divide iterations by the number of iterations per epoch
    # now it is mobilenet v3,VGG paper trains for 120,000 iterations with a batch size of 32, decays after 80,000 and 100,000 iterations,
    epochs = 600
    # decay_lr_at =[154, 193]
    # print("decay_lr_at:",decay_lr_at)
    print("epochs:",epochs)

    for param_group in optimizer.param_groups:
        optimizer.param_groups[1]['lr']=lr
    print("learning rate.  The new LR is %f\n" % (optimizer.param_groups[1]['lr'],))    
    # Epochs,I try to use different learning rate shcheduler
    #different scheduler six way you could try
    #scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max = (epochs // 7) + 1) 
    scheduler = ReduceLROnPlateau(optimizer,mode="min",factor=0.1,patience=15,verbose=True, threshold=0.00001, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08)

    for epoch in range(start_epoch, epochs):

        # Decay learning rate at particular epochs
        # if epoch in decay_lr_at:
        #     adjust_learning_rate_epoch(optimizer,epoch)
        

        # One epoch's training
        train(train_loader=train_loader,
              model=model,
              criterion=criterion,
              optimizer=optimizer,
              epoch=epoch)
        print("epoch loss:",train_loss)      
        scheduler.step(train_loss)      

        # Save checkpoint
        save_checkpoint(epoch, model, optimizer)


def train(train_loader, model, criterion, optimizer, epoch):

    model.train()  # training mode enables dropout

    batch_time = AverageMeter()  # forward prop. + back prop. time
    data_time = AverageMeter()  # data loading time
    losses = AverageMeter()  # loss

    start = time.time()
    global train_loss
    # Batches
    for i, (images, boxes, labels, _) in enumerate(train_loader):
        data_time.update(time.time() - start)

        # if(i%200==0):
        #     adjust_learning_rate_iter(optimizer,epoch)
        #     print("batch id:",i)#([8, 3, 300, 300])
        #N=8
        # Move to default device
        images = images.to(device)  # (batch_size (N), 3, 300, 300)
        
        boxes = [b.to(device) for b in boxes]
        labels = [l.to(device) for l in labels]

        # Forward prop.
        predicted_locs, predicted_scores = model(images)  # (N, anchor_boxes_size, 4), (N, anchor_boxes_size, n_classes)

        # Loss
        loss = criterion(predicted_locs, predicted_scores, boxes, labels)  # scalar
        train_loss=loss
        #print("training",train_loss)

        # Backward prop.
        optimizer.zero_grad()
        loss.backward()

        # Clip gradients, if necessary
        if grad_clip is not None:
            clip_gradient(optimizer, grad_clip)

        # Update model
        optimizer.step()

        losses.update(loss.item(), images.size(0))
        batch_time.update(time.time() - start)

        start = time.time()

        # Print status
        if i % print_freq == 0:
            print('Epoch: [{0}][{1}/{2}][{3}]\t'
                  'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data Time {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(epoch, i, len(train_loader),optimizer.param_groups[1]['lr'],
                                                                  batch_time=batch_time,
                                                                  data_time=data_time, loss=losses))

        #break #test
    del predicted_locs, predicted_scores, images, boxes, labels  # free some memory since their histories may be stored


def adjust_learning_rate_epoch(optimizer,cur_epoch):
    """
    Scale learning rate by a specified factor.

    :param optimizer: optimizer whose learning rate must be shrunk.
    :param scale: factor to multiply learning rate with.
    """
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * 0.1
    print("DECAYING learning rate. The new LR is %f\n" % (optimizer.param_groups[1]['lr'],))

#warmup ,how much learning rate.
def adjust_learning_rate_iter(optimizer,cur_epoch):

    if(cur_epoch==0 or cur_epoch==1 ):
        for param_group in optimizer.param_groups:
            param_group['lr'] =param_group['lr'] +  0.0001  
            print("DECAYING learning rate iter.  The new LR is %f\n" % (optimizer.param_groups[1]['lr'],))

      


if __name__ == '__main__':
    main()
