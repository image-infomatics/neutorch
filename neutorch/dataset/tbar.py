import json
import os
from typing import Union
from time import time

import numpy as np
import h5py

import torch
import torchvision
import torchio as tio
import toml


def image_reader(path: str):
    with h5py.File(path, 'r') as file:
        img = np.asarray(file['main'])
    # the last one is affine transformation matrix in torchio image type
    return img, None

class Dataset(torch.utils.data.Dataset):
    def __init__(self, config_file: str, training_split_ratio: float = 0.9,
            patch_size: Union[int, tuple]=64, queue_length: int=8,
            num_workers: int=4, sampling_distance: int = 22):
        """
        Parameters:
            config_file (str): file_path to provide metadata of all the ground truth data.
            training_split_ratio (float): split the datasets to training and validation sets.
            patch_size (int or tuple): the patch size we are going to provide.
        """
        super().__init__()
        assert training_split_ratio > 0.
        assert training_split_ratio < 1.
        config_file = os.path.expanduser(config_file)
        assert config_file.endswith('.toml'), "we use toml file as configuration format."

        with open(config_file, 'r') as file:
            meta = toml.load(file)

        config_dir = os.path.dirname(config_file)
        
        # load all the datasets
        subjects = []
        # subject_weights = []
        # walkthrough the directory and find all the groundtruth files automatically
        for gt in meta.values():
            image_path = gt['image']
            synapse_path = gt['ground_truth']
            assert image_path.endswith('.h5')
            assert synapse_path.endswith('.json')
            image_path = os.path.join(config_dir, image_path)
            synapse_path = os.path.join(config_dir, synapse_path)

            with h5py.File(image_path, 'r') as file:
                img = np.asarray(file['main'])
                voxel_offset = np.asarray(file['voxel_offset'], dtype=np.uint32)
            # use the voxel number as the sampling weights
            # subject_weights.append(len(img))
            with open(synapse_path, 'r') as file:
                synapses = json.load(file)
            
            # use the number of T-bars as subject sampling weights
            # subject_weights.append(len(synapses['presynapses']))

            bin_presyn, sampling_map = self._annotation_to_volumes(
                img, voxel_offset, synapses,
                sampling_distance = sampling_distance
            )

            img = np.expand_dims(img, axis=0)
            img = torch.from_numpy(img)
            bin_presyn = np.expand_dims(bin_presyn, axis=0)
            bin_presyn = torch.from_numpy(bin_presyn)
            sampling_map = np.expand_dims(sampling_map, axis=0)
            sampling_map = torch.from_numpy(sampling_map)
            subject = tio.Subject(
                image = tio.ScalarImage(tensor=img),
                tbar = tio.LabelMap(tensor=bin_presyn),
                sampling_map = tio.Image(tensor=sampling_map, type=tio.SAMPLING_MAP)
            )
            subjects.append(subject)
            # self.datasets.append((img, synapses))
            # use the image voxel number as sampling weight
            # self.dataset_weights.append(len(img))
        subjects_dataset = tio.SubjectsDataset(subjects, transform=self.transform)
        print('number of volumes in dataset: ', len(subjects_dataset))

        patch_sampler = tio.data.WeightedSampler(patch_size, 'sampling_map')
        self.patches_queue = tio.Queue(
            subjects_dataset,
            queue_length,
            1,
            patch_sampler,
            num_workers=num_workers,
        )
        # only sample one subject, so replacement option could be ignored
        # self.subject_sampler = torch.utils.data.WeightedRandomSampler(subject_weights, 1)

    @property
    def random_patches(self):
        return self.patches_queue
                
    def _annotation_to_volumes(self, img: np.ndarray, voxel_offset: np.ndarray, synapses: dict,
            sampling_distance: int = 22) -> tuple:
        """transform point annotation to volumes

        Args:
            img (np.ndarray): image volume
            voxel_offset (np.ndarray): offset of image volume
            synapses (dict): the annotated synapses
            sampling_distance (int, optional): the maximum distance from the annotated point to 
                the center of sampling patch. Defaults to 22.

        Returns:
            bin_presyn: binary label of annotated position.
            sampling_probability_map: the probability map of sampling
        """
        assert synapses['order'] == ["x", "y", "z"]
        # assert synapses['resolution'] == [8, 8, 8]
        bin_presyn = np.zeros_like(img, dtype=np.float32)
        sampling_map = np.zeros_like(img, dtype=np.uint8)
        for coordinate in synapses['presynapses'].values():
            # transform coordinate from xyz order to zyx
            coordinate = coordinate[::-1]
            coordinate = np.asarray(coordinate, dtype=np.uint32)
            coordinate -= voxel_offset
            bin_presyn[
                coordinate[0]-1 : coordinate[0]+1,
                coordinate[1]-1 : coordinate[1]+1,
                coordinate[2]-1 : coordinate[2]+1,
            ] = 1.
            sampling_map[
                coordinate[0]-sampling_distance : coordinate[0]+sampling_distance,
                coordinate[1]-sampling_distance : coordinate[1]+sampling_distance,
                coordinate[2]-sampling_distance : coordinate[2]+sampling_distance,
            ] += 1
        return bin_presyn, sampling_map
    
    @property
    def transform(self):
        return tio.Compose([
            # tio.RandomMotion(p=0.2),
            # tio.RandomBiasField(p=0.3),
            tio.RandomNoise(p=0.5),
            tio.RandomFlip(),
            # tio.OneOf({
            #     tio.RandomAffine(): 0.3,
            #     tio.RandomElasticDeformation(): 0.7
            # }),
            tio.RandomGamma(p=0.1),
            # tio.RandomGhosting(p=0.1),
            # tio.RandomAnisotropy(p=0.2),
            # tio.RandomSpike(p=0.1),
        ])

if __name__ == '__main__':
    dataset = Dataset(
        "~/Dropbox (Simons Foundation)/40_gt/tbar.toml",
        num_workers=1,
        sampling_distance=4,
    )
    
    training_batch_size = 5
    patches_loader = torch.utils.data.DataLoader(
        dataset.random_patches,
        batch_size=training_batch_size
    )
    
    model = torch.nn.Identity()
    n = 0
    print('start generating random patches...')
    ping = time()
    for patches_batch in patches_loader:
        print(f'generating a patch takes {int(time()-ping)} seconds.')
        # print(patch)
        image = patches_batch['image'][tio.DATA]
        assert image.shape[0] == training_batch_size
        image = image[:, :, 32, :, :]
        tbar = patches_batch['tbar'][tio.DATA]
        logits = model(image)
        tbar, _ = torch.max(tbar, dim=2, keepdim=False)
        # breakpoint()
        slices = torch.cat((image, tbar))
        image_path = os.path.expanduser('~/Downloads/patches.png')
        print('save a batch of patches to ', image_path)
        torchvision.utils.save_image(
            slices,
            image_path,
            nrow=training_batch_size,
            normalize=True,
            scale_each=True,
        )

        ping = time()
        n += 1
        if n>0:
            break
