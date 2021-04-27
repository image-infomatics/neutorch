import json
import os

import numpy as np
import h5py

import torch
import torchio as tio
import toml


def image_reader(path: str):
    with h5py.File(path, 'r') as file:
        img = np.asarray(file['main'])
    # the last one is affine transformation matrix in torchio image type
    return img, None

class Dataset(torch.utils.data.Dataset):
    def __init__(self, config_file: str, training_split_ratio: float = 0.9):
        """
        Parameters:
            config_file (str): file_path to provide metadata of all the ground truth data.
            training_split_ratio (float): split the datasets to training and validation sets.
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
        # self.datasets = []
        # self.dataset_weights = []
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
            
            with open(synapse_path, 'r') as file:
                synapses = json.load(file)
            
            bin_presyn, sampling_probability_map = self._annotation_to_volumes(
                img, voxel_offset, synapses
            )

            img = np.expand_dims(img, axis=0)
            img = torch.from_numpy(img)
            bin_presyn = np.expand_dims(bin_presyn, axis=0)
            bin_presyn = torch.from_numpy(bin_presyn)
            sampling_probability_map = np.expand_dims(sampling_probability_map, axis=0)
            sampling_probability_map = torch.from_numpy(sampling_probability_map)
            subject = tio.Subject(
                img = tio.ScalarImage(tensor=img),
                tbar = tio.LabelMap(tensor=bin_presyn),
                sampling_probability_map = tio.Image(tensor=sampling_probability_map)
            )
            subjects.append(subject)
            # self.datasets.append((img, synapses))
            # use the image voxel number as sampling weight
            # self.dataset_weights.append(len(img))
        dataset = tio.SubjectsDataset(subjects, transform=self.transform)
        print('number of volumes in dataset: ', len(dataset))
        breakpoint()
                
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
        sampling_probability_map = np.zeros_like(img, dtype=np.uint8)
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
            sampling_probability_map[
                coordinate[0]-sampling_distance : coordinate[0]+sampling_distance,
                coordinate[1]-sampling_distance : coordinate[1]+sampling_distance,
                coordinate[2]-sampling_distance : coordinate[2]+sampling_distance,
            ] += 1
        return bin_presyn, sampling_probability_map
    
    @property
    def transform(self):
        return tio.Compose([
            tio.RandomMotion(p=0.2),
            tio.RandomBiasField(p=0.3),
            tio.RandomNoise(p=0.5),
            tio.RandomFlip(),
            tio.OneOf({
                tio.RandomAffine(): 0.3,
                tio.RandomElasticDeformation(): 0.7
            }),
            tio.RandomGamma(p=0.1),
            tio.RandomGhosting(p=0.1),
            tio.RandomAnisotropy(p=0.2),
            tio.RandomSpike(p=0.1),
        ])

if __name__ == '__main__':
    dataset = Dataset("~/Dropbox (Simons Foundation)/40_gt/tbar.toml")
