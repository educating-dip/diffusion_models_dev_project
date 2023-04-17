
import os 
from pydicom.filereader import dcmread
import numpy as np 

import torch
from torch.utils.data import Dataset
from torch.nn.functional import interpolate

class MayoDataset(Dataset):
    def __init__(self, part, base_path="/localdata/jleuschn/data/LDCT-and-Projection-data"):
        assert part in ["N", "L", "C"]

        self.part = part
        self.base_path = base_path

        self.full_subjects = {"N": ["N001", "N003", "N005", "N012", "N017", "N021", "N024", "N025", "N029", "N030"],
                    "L": ["L004", "L006", "L012", "L014", "L019", "L024", "L027", "L030", "L033", "L035"], 
                    "C": ["C001", "C002", "C004", "C009", "C012", "C016", "C019", "C021", "C023", "C027"]}

        self.subjetcs = self.full_subjects[self.part]

    def __len__(self):
        return len(self.subjetcs)

    def __getitem__(self, idx):
        subject = self.subjetcs[idx]

        dirs = os.listdir(os.path.join(self.base_path, subject))
        full_dose_image_dirs = [d for d in os.listdir(os.path.join(self.base_path, subject, dirs[0])) if "Full Dose" in d]
        assert len(full_dose_image_dirs) == 1

        path = os.path.join(self.base_path, subject, dirs[0], full_dose_image_dirs[0])

        dcm_files = os.listdir(path)
        dcm_files.sort(key=lambda f: float(dcmread(os.path.join(path, f), specific_tags=['SliceLocation'])['SliceLocation'].value))

        sample_slice = (len(dcm_files) - 1) // 2
        dcm_dataset = dcmread(os.path.join(path, dcm_files[sample_slice]))

        rng = np.random.default_rng(1)

        array = dcm_dataset.pixel_array[75:-75,75:-75].astype(np.float32).T
        array *= dcm_dataset.RescaleSlope
        array += dcm_dataset.RescaleIntercept
        array += rng.uniform(0., 1., size=array.shape)

        # convert values
        MU_WATER = 20
        MU_AIR = 0.02
        MU_MAX = 3071 * (MU_WATER - MU_AIR) / 1000 + MU_WATER
        array *= (MU_WATER - MU_AIR) / 1000
        array += MU_WATER
        array /= MU_MAX
        np.clip(array, 0., 1., out=array)

        array = torch.from_numpy(array).float().unsqueeze(0).unsqueeze(0)

        x = interpolate(array, size=(501, 501), mode="bilinear").squeeze()

        return x.unsqueeze(0)


if __name__ == "__main__":

    dataset = MayoDataset(part="C")

    x = dataset[0]

    print(x.shape)

    import matplotlib.pyplot as plt 

    plt.figure()
    plt.imshow(x[0,:,:])
    plt.show()