import argparse
import glob
import os

import torch
from torchvision import transforms

from dataloader import ElbowxrayDataset
from model import EfficientNet
from utils import *
from utils import _ns

def main():
    parser = argparse.ArgumentParser(description='Transfer images to styles.')
    parser.add_argument('--dataset-dir', type=str, dest='data_dir',
                        help='path to content images directory',
                        default='data/img_tiff'
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

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    args = parser.parse_args()
    num_epoch = args.num_epoch

    source_images_path = glob.glob(os.path.join(args.data_dir, '*')) # glob.glob("arg.data_dir/*")

    elbow_xray_dataset = ElbowxrayDataset(xlsx_file='data/elbow.xlsx', root_dir='data/img_tiff/', 
                                            transform=transforms.Compose([transforms.ToTensor(),transforms.Resize(size=(512,512))])
                                           )

    dataloader_train = torch.utils.data.DataLoader(elbow_xray_dataset, batch_size=args.batch_size, shuffle=True)

    if args.weight_path == 'none':
        model = EfficientNet.from_pretrained('efficientnet-b6', num_classes=2).to(device)
        model.train()
    else:
        state = torch.load(args.weight_path)
        model.load_state_dict(state)
        model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    criterion = torch.nn.CrossEntropyLoss() # subject due to change
    
    for epoch in range(num_epoch):


        for i, data in enumerate(dataloader_train, 0):
            images, labels = data['image'].to(device), data['label'].to(device)

            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            if i % 20 == 19:
                print('[Epoch %d/%d, %5d] loss: %.3f' %(epoch + 1, num_epoch, i + 1, loss / 20))

        torch.save(model.state_dict(), 'ckpt/' + str(epoch) + '.pkl')

if __name__=='__main__':
    main()