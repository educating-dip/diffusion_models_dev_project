from torch.utils.data import DataLoader
from dival import get_standard_dataset

class EllipseDatasetFromDival():
    def __init__(self, 
        impl: str = 'astra_cuda'
        ) -> None:

        self.impl = impl
        dataset = get_standard_dataset('ellipses', impl=self.impl)
        self.ellipses_train = dataset.create_torch_dataset(
                part='train',
                reshape=( (1, ) + dataset.space[0].shape, (1, ) + dataset.space[1].shape)
                )
        self.ellipses_val = dataset.create_torch_dataset(
            part='validation',
            reshape=( (1, ) + dataset.space[0].shape, (1, ) + dataset.space[1].shape)
            )
                    
    def get_trainloader(self,
            batch_size: int,
            num_data_loader_workers: int = 0
            ) -> DataLoader:
        return DataLoader(self.ellipses_train, 
            batch_size=batch_size, num_workers=num_data_loader_workers, pin_memory=True)

    def get_valloader(self,
        batch_size: int, 
        num_data_loader_workers: int = 0
        ) -> DataLoader:
        return DataLoader(self.ellipses_val, 
            batch_size=batch_size, num_workers=num_data_loader_workers, pin_memory=True)