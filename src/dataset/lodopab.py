from torch.utils.data import DataLoader
from dival import get_standard_dataset
from torch.nn.functional import interpolate

class LoDoPabDatasetFromDival():
    def __init__(self, 
            impl: str = "astra_cuda",
            im_size=362
        ):

        self.impl = impl
        dataset = get_standard_dataset('lodopab', impl=self.impl)
        self.im_size = im_size

        def transform(sample):
            x = sample[1]

            x = interpolate(x.unsqueeze(0), size=(self.im_size, self.im_size), mode="bilinear").squeeze()
            return x.unsqueeze(0) 

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
            ):
        return DataLoader(self.lodopab_train, 
            batch_size=batch_size, num_workers=num_data_loader_workers, pin_memory=True, shuffle=True)

    def get_valloader(self,
            batch_size: int, 
            num_data_loader_workers: int = 0
        ):
        return DataLoader(self.lodopab_val, 
            batch_size=batch_size, num_workers=num_data_loader_workers, pin_memory=True, shuffle=False)

    def get_testloader(self,
            batch_size: int, 
            num_data_loader_workers: int = 0
        ):
        return DataLoader(self.lodopab_test, 
            batch_size=batch_size, num_workers=num_data_loader_workers, pin_memory=True, shuffle=False)


if __name__ == "__main__":

    dataset = LoDoPabDatasetFromDival(im_size=501)

    x= dataset.lodopab_train[0]

    print(x.shape)

    import matplotlib.pyplot as plt 

    plt.figure()
    plt.imshow(x[0,:,:])

    plt.show()