import os
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torch.utils.data.distributed import DistributedSampler
from torchvision.datasets.mnist import MNIST
from torchvision.datasets.cifar import CIFAR10
from torchvision.datasets.stl10 import STL10
from torchvision.datasets.coco import CocoCaptions
from dataset.dataset import HuggingFaceDataset
import datasets

def get_data(data, img_size, data_folder, bsize, num_workers, is_multi_gpus, seed, args):
    """ Class to load data """

    if data == "mnist":
        data_train = MNIST('./dataset_storage/mnist/', download=False,
                           transform=transforms.Compose([transforms.Resize(img_size),
                                                         transforms.ToTensor(),
                                                         ]))

    elif data == "cifar10":
        data_train = CIFAR10(data_folder, train=True, download=True,
                             transform=transforms.Compose([
                                 transforms.Resize(img_size),
                                 transforms.RandomHorizontalFlip(),
                                 transforms.ToTensor(),
                                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                             ]))

        data_test = CIFAR10(data_folder, train=False, download=False,
                            transform=transforms.Compose([
                                transforms.Resize(img_size),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                            ]))

    elif data == "stl10":
        data_train = STL10('./Dataset/stl10', split="train+unlabeled",
                           transform=transforms.Compose([
                               transforms.Resize(img_size),
                               transforms.RandomHorizontalFlip(),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                           ]))
        data_test = STL10('./Dataset/stl10', split="test",
                          transform=transforms.Compose([
                              transforms.Resize(img_size),
                              transforms.ToTensor(),
                              transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                          ]))

    elif data == "imagenet" or data == "imagenet256" or data == "timm/imagenet-1k-wds":
        t_train = transforms.Compose([transforms.Resize(img_size),
                                      transforms.CenterCrop((img_size, img_size)),
                                      # transforms.RandomHorizontalFlip(),
                                      transforms.ToTensor(),
                                      transforms.Normalize(
                                         mean=[.5, .5, .5],
                                         std=[.5, .5, .5])
                                      ])

        t_test = transforms.Compose([transforms.Resize(img_size),
                                     transforms.CenterCrop((img_size, img_size)),
                                     transforms.ToTensor(),
                                     transforms.Normalize(
                                         mean=[.5, .5, .5],
                                         std=[.5, .5, .5])
                                     ])

       
        data_train = HuggingFaceDataset('timm/imagenet-1k-wds', split="train", img_size=img_size, transform=t_train, data_cache=data_folder)
        data_test = HuggingFaceDataset('timm/imagenet-1k-wds', split="validation", img_size=img_size, transform=t_test, data_cache=data_folder)
    
    elif data == "imagenet_feat":
        input_size = args.img_size // args.f_factor    
        def process(batch):
            batch["code"] = batch['code'][:, 1:]
            batch["code"] = batch["code"].reshape(-1, input_size, input_size)
            return batch
    
        data_train = datasets.load_from_disk(os.path.join(data_folder, "train")) \
            .rename_columns({"input_ids": "code", "label": "y"}) \
            .map(process, batched=True)

        data_test = datasets.load_from_disk(os.path.join(data_folder, "eval")) \
            .rename_columns({"input_ids": "code", "label": "y"}) \
            .map(process, batched=True)
        data_train.set_format("torch")
        data_test.set_format("torch")
        
        if args.debug:
            data_train = data_train.select(range(16)) 
            data_test = data_test.select(range(16)) 
                
    elif data == 'imagenet_feat_pgm':
        input_size = args.img_size // args.f_factor    
        if args.debug:
            data_train = datasets.load_from_disk(os.path.join(data_folder, "train")).select(range(16)) \
            .rename_columns({"input_ids": "code", "label": "y"}) 
        data_test = datasets.load_from_disk(os.path.join(data_folder, "eval")).select(range(16)) \
            .rename_columns({"input_ids": "code", "label": "y"}) 
            
        data_train = datasets.load_from_disk(os.path.join(data_folder, "train")) \
            .rename_columns({"input_ids": "code", "label": "y"}) 
        data_test = datasets.load_from_disk(os.path.join(data_folder, "eval")) \
            .rename_columns({"input_ids": "code", "label": "y"}) 
        data_train.set_format("torch")
        data_test.set_format("torch")
    elif data == "mscoco":
        data_test = CocoCaptions(root=os.path.join(data_folder, 'images/val2017/'),
                                 annFile=os.path.join(data_folder, 'annotations/captions_val2017.json'),
                                 transform=transforms.Compose([
                                     transforms.Resize(img_size),
                                     transforms.CenterCrop((img_size, img_size)),
                                     transforms.ToTensor(),
                                     transforms.Normalize(
                                         mean=[.5, .5, .5],
                                         std=[.5, .5, .5])
                                 ]),
                                 target_transform=lambda x: x[:5])
        test_sampler = DistributedSampler(data_test, shuffle=False, seed=seed) if is_multi_gpus else None
        test_loader = DataLoader(data_test, batch_size=bsize, shuffle=False,
                                 num_workers=num_workers, pin_memory=True,
                                 drop_last=False, sampler=test_sampler)

        return None, test_loader
    else:
        data_train = None
        data_test = None
    if data_train is not None:
        print(f"Number of training samples: {len(data_train)}")
        train_sampler = DistributedSampler(data_train, shuffle=True, seed=seed) if is_multi_gpus else None
        train_loader = DataLoader(data_train, batch_size=bsize,
                                shuffle=False if is_multi_gpus else True,
                                num_workers=num_workers, pin_memory=True,
                                drop_last=True, sampler=train_sampler)
    else:
        train_loader = None
    if data_test is not None:
        print(f"Number of testing samples: {len(data_test)}")
        test_sampler = DistributedSampler(data_test, shuffle=True, seed=seed) if is_multi_gpus else None
        test_loader = DataLoader(data_test, batch_size=bsize,
                             shuffle=False if is_multi_gpus else True,
                             num_workers=num_workers, pin_memory=True,
                             drop_last=True, sampler=test_sampler)
    else:
        test_loader = None
    return train_loader, test_loader
