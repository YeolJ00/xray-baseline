import argparse
import glob
import os

import torch
from torchvision import transforms

from .dataloader import ElbowxrayDataset
from .model import EfficientNet
from .utils import *

def main():
    parser = argparse.ArgumentParser(description='Transfer images to styles.')
    parser.add_argument('--dataset-dir', type=str, dest='data_dir',
                        help='path to content images directory',
                        )
    parser.add_argument('--chk-path', type=str, dest='weight_path',
                        help='path to model weights',
                        default='none')
    parser.add_argument('--device', type=str,
                        help='type of inference device',
                        default='cuda',
                        )
    parser.add_argument('--batch-size', type=int, dest='batch_size',
                        default=4)
    parser.add_argument('--epoch', type=int, dest='num_epoch',
                        default=10)


    args = parser.parse_args()
    num_epoch = args.num_epoch

    source_images_path = glob.glob(os.path.join(args.data_dir, '*')) # glob.glob("arg.data_dir/*")

    # elbow_xray_dataset = ElbowxrayDataset(csv_file='data/elbow_aligned.xlsx', root_dir='data/imgs_tiff/', 
    #                                         transform=transforms.Compose([transforms.ToTensor(), partial(transforms.functional.resize(size=(1024,512)))])
    #                                        )
    elbow_xray_dataset = ElbowxrayDataset(csv_file='data/elbow_aligned.xlsx', root_dir='data/imgs_tiff/', 
                                            transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
                                           )

    dataloader_train = torch.utils.data.DataLoader(elbow_xray_dataset, batch_size=args.batch_size, shuffle=True)

    if args.weight_path == 'none':
        model = EfficientNet.from_pretrained('efficientnet-b6', num_classes=2)
        model.train()
    else:
        state = torch.load(args.weight_path)
        model.load_state_dict(state)
        model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    criterion = torch.nn.BCELoss() # subject due to change
    
    for epoch in num_epoch:

        running_loss = 0.0

        for i, data in enumerate(dataloader_train, 0):
            images, labels = data['image'], data['label']

            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss
            if i % 200 == 199:
                print('[Epoch %d/%d, %5d] loss: %.3f' %(epoch + 1, num_epoch, i + 1, running_loss / 2000))
        if loss < running_loss:
            torch.save(model.state_dict(), 'ckpt/' + epoch + '.pkl')

