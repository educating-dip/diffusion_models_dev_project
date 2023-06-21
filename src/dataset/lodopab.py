"""
Adapted from Dival
"""
import numpy as np 
import torch 
from torch import Tensor 
from torch.utils.data import DataLoader
from dival import get_standard_dataset
from torch.nn.functional import interpolate
from torch.utils.data import Dataset

class LoDoPabDatasetFromDival():
    def __init__(self, 
        impl: str = 'astra_cuda',
        im_size: int = 362,
        ) -> None:

        self.impl = impl
        dataset = get_standard_dataset('lodopab', impl=self.impl)
        self.ray_trafo = dataset.ray_trafo
        self.im_size = im_size
        self.use_transform = self.im_size != 362
        def transform(sample: Tensor) -> Tensor:
            x = sample[1]
            if self.use_transform:
                x = interpolate(
                        x.unsqueeze(0), 
                        size=(self.im_size, self.im_size), 
                        mode='bilinear'
                    ).squeeze().unsqueeze(0) 

            return x

        self.lodopab_train = dataset.create_torch_dataset(part='train',
            reshape=((1,) + dataset.space[0].shape,
            (1,) + dataset.space[1].shape), transform=transform)
        self.lodopab_val = dataset.create_torch_dataset(part='validation',
            reshape=((1,) + dataset.space[0].shape,
            (1,) + dataset.space[1].shape), transform=transform)
        self.lodopab_test = dataset.create_torch_dataset(part='test',
            reshape=((1,) + dataset.space[0].shape,
            (1,) + dataset.space[1].shape), transform=transform)

    def get_trainloader(self,
            batch_size: int,
            num_data_loader_workers: int = 0
            ) -> DataLoader:
        return DataLoader(self.lodopab_train, 
            batch_size=batch_size, num_workers=num_data_loader_workers, pin_memory=True, shuffle=True)

    def get_valloader(self,
            batch_size: int, 
            num_data_loader_workers: int = 0
            ) -> DataLoader:
        return DataLoader(self.lodopab_val, 
            batch_size=batch_size, num_workers=num_data_loader_workers, pin_memory=True, shuffle=False)

    def get_testloader(self,
            batch_size: int, 
            num_data_loader_workers: int = 0
            ) -> DataLoader:
        return DataLoader(self.lodopab_test, 
            batch_size=batch_size, num_workers=num_data_loader_workers, pin_memory=True, shuffle=False)


class LoDoPabChallenge():
    def __init__(self, 
        impl: str = 'astra_cuda',
        im_size: int = 362,
        ) -> None:

        self.impl = impl
        dataset = get_standard_dataset('lodopab', impl=self.impl)
        self.ray_trafo = dataset.ray_trafo
        self.im_size = im_size
        
        self.lodopab_train = dataset.create_torch_dataset(part='train',
            reshape=((1,) + dataset.space[0].shape,
            (1,) + dataset.space[1].shape))
        self.lodopab_val = dataset.create_torch_dataset(part='validation',
            reshape=((1,) + dataset.space[0].shape,
            (1,) + dataset.space[1].shape))
        self.lodopab_test = dataset.create_torch_dataset(part='test',
            reshape=((1,) + dataset.space[0].shape,
            (1,) + dataset.space[1].shape))

    def get_trainloader(self,
            batch_size: int,
            num_data_loader_workers: int = 0
            ) -> DataLoader:
        return DataLoader(self.lodopab_train, 
            batch_size=batch_size, num_workers=num_data_loader_workers, pin_memory=True, shuffle=True)

    def get_valloader(self,
            batch_size: int, 
            num_data_loader_workers: int = 0
            ) -> DataLoader:
        return DataLoader(self.lodopab_val, 
            batch_size=batch_size, num_workers=num_data_loader_workers, pin_memory=True, shuffle=False)

    def get_testloader(self,
            batch_size: int, 
            num_data_loader_workers: int = 0
            ) -> DataLoader:
        return DataLoader(self.lodopab_test, 
            batch_size=batch_size, num_workers=num_data_loader_workers, pin_memory=True, shuffle=False)



class SubsetLoDoPab(Dataset):
    def __init__(self, 
        part: str, # val, test 
        impl: str = 'astra_cuda',
        im_size: int = 362,
        ) -> None:

        assert part in ['val', 'test']

        self.impl = impl
        dataset = get_standard_dataset('lodopab', impl=self.impl)
        self.ray_trafo = dataset.ray_trafo
        self.im_size = im_size
        self.use_transform = self.im_size != 362
        def transform(sample: Tensor) -> Tensor:
            x = sample[1]
            if self.use_transform:
                x = interpolate(
                        x.unsqueeze(0), 
                        size=(self.im_size, self.im_size), 
                        mode='bilinear'
                    ).squeeze().unsqueeze(0) 

            return x

        lodopab_val = dataset.create_torch_dataset(part='validation',
            reshape=((1,) + dataset.space[0].shape,
            (1,) + dataset.space[1].shape), transform=transform)
        
        rng = np.random.default_rng(seed=0)
        indices = rng.choice(len(lodopab_val), 110, replace=False)

        indices_val = indices[:10]
        indices_test = indices[10:]
        if part == "val":
            self.lodopab_subset = torch.utils.data.Subset(lodopab_val, indices_val)
        elif part == "test":
            self.lodopab_subset = torch.utils.data.Subset(lodopab_val, indices_test)

    def __len__(self):
        return len(self.lodopab_subset)

    def __getitem__(self, idx: int):
        
        x = self.lodopab_subset[idx]

        return x


if __name__ == "__main__":
    import matplotlib.pyplot as plt 

    dataset = SubsetLoDoPab(part="val", im_size=256) 

    print(dataset.lodopab_subset[0].shape)
    fig, axes = plt.subplots(1,5)

    for idx, ax in enumerate(axes.ravel()):
        ax.imshow(dataset.lodopab_subset[idx][0,:,:])
    plt.show()