import sys
import os
from torch import optim
from torch.utils import data
from AutoEncoder.network import AutoEncoderNet
from AutoEncoder.loader import AutoEncoderDataset
from tools.utils import yaml_load, DictAsMember, load_from_file
from tools.hdr_image import HDRImageHandler

# from AutoEncoder.callback import AutoEncoderCallback

if __name__ == '__main__':
    #
    #   Load configurations
    #
    try:
        config_path = sys.argv[1]
    except IndexError:
        config_path = "AutoEncoder/train.yml"
    opt = DictAsMember(yaml_load(config_path))

    #
    #   Instantiate models
    #
    # model = load_from_file(opt.network, 'AutoEncoderNet')
    model = AutoEncoderNet()
    #
    #   Instantiate loaders
    #
    hdr_image_handler = HDRImageHandler(opt.hdr_mean_std, perform_scale_perturbation=True)
    train_dataset = AutoEncoderDataset(os.path.join(opt.data_path, "train"),
                                       transform=hdr_image_handler.normalization_ops)

    valid_dataset = AutoEncoderDataset(os.path.join(opt.data_path, "valid"),
                                       transform=hdr_image_handler.normalization_ops)

    train_loader = data.DataLoader(train_dataset,
                                   batch_size=opt.batch_size,
                                   num_workers=opt.workers,
                                   pin_memory=opt.use_shared_memory,
                                   drop_last=True)

    valid_loader = data.DataLoader(valid_dataset,
                                   batch_size=opt.batch_size,
                                   num_workers=opt.workers,
                                   pin_memory=opt.use_shared_memory)
    
    if opt.optimizer == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=opt.learning_rate, weight_decay=1e-5)
    else:
        optimizer = optim.SGD(model.parameters(), lr=opt.learning_rate, momentum=0.9)
    
    running_loss = 0
    dataiter = iter(train_loader)

    epoch = 100
    for e in range(epoch):
        running_loss = 0
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            
            optimizer.zero_grad()

            outputs = model(inputs)
            loss = model.loss(outputs, labels)

            loss.backgward()

            optimizer.step()

            running_loss += loss.item()

            if i % 2000 == 1999:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                    (e + 1, i + 1, running_loss / 2000))
                running_loss = 0.0




    

