import os
import sys
curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder

from AEanoWgan.test_anomaly_detection import test_anomaly_detection


def main(opt):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    pipeline = [transforms.Resize([opt.img_size]*2),
                transforms.RandomHorizontalFlip()]
    if opt.channels == 1:
        pipeline.append(transforms.Grayscale())
    pipeline.extend([transforms.ToTensor(),
                     transforms.Normalize([0.5]*opt.channels, [0.5]*opt.channels)])

    transform = transforms.Compose(pipeline)
    dataset = ImageFolder(opt.test_root, transform=transform)
    test_dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
    from mvtec_ad.model import Generator, Discriminator, Encoder

    generator = Generator(opt)
    discriminator = Discriminator(opt)
    encoder = Encoder(opt)

    test_anomaly_detection(opt, generator, discriminator, encoder,
                           test_dataloader, device)





if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("test_root", type=str,
                        help="root name of your dataset in test mode")
    parser.add_argument("--force_download", "-f", action="store_true",
                        help="flag of force download")
    parser.add_argument("--latent_dim", type=int, default=100,
                        help="dimensionality of the latent space")
    parser.add_argument("--img_size", type=int, default=256,
                        help="size of each image dimension")
    parser.add_argument("--channels", type=int, default=3,
                        help="number of image channels (If set to 1, convert image to grayscale)")
    opt = parser.parse_args()

    main(opt)
