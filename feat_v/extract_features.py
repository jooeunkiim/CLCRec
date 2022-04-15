from tqdm import tqdm
import numpy as np
from torchvision import transforms, models
from torch.utils.data import DataLoader

from dataloader import *

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("current device:", device)

    transform = transforms.Compose([
        ToTensor(), 
        Normalize(),
        ])

    dataset = CustomDataset(transform = transform)
    dataloader = DataLoader(dataset, batch_size=128, collate_fn=dataset.custom_collate_fn)

    resnet = models.resnet50(pretrained=True).to(device)
    # model = resnet
    model = torch.nn.Sequential(*(list(resnet.children())[:-1]))

    model.eval()

    movielens_ids = []
    initialize = True

    with torch.no_grad():
        progress_bar = tqdm(enumerate(dataloader), total=len(dataloader))
        for _, data in progress_bar:
        # for _, data in enumerate(dataloader):
            input = data['input'].to(device)
            output = model(input)
            output = output.cpu().detach().numpy()
            output = np.squeeze(output)
            if initialize:
                features = output
                initialize = False
            else:
                features = np.append(features, output, axis=0)
            progress_bar.set_description(str(features.shape))

    print(features.shape)
    np.save("/data/private/CLCRec/Data/movie/feat_v.npy", features)
    

