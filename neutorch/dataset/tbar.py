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
    def __init__(self, path: str, training_split_ratio: float = 0.9):
        """
        Parameters:
            path (str): directory containing all the datasets
            training_split_ratio (float): split the datasets to training and validation sets.
        """
        super().__init__()
        assert training_split_ratio > 0.
        assert training_split_ratio < 1.
        path = os.path.expanduser(path)

        # load all the datasets
        subjects = []
        # self.datasets = []
        # self.dataset_weights = []
        # walkthrough the directory and find all the groundtruth files automatically
        for dirpath, _, filenames in os.walk(path):
            if "meta.toml" in filenames and "synapse" in os.path.basename(dirpath):
                with open(os.path.join(dirpath, "meta.toml"), 'r') as file:
                    meta = toml.load(file)
                image_file_name = meta['synapse']['image']
                with h5py.File(os.path.join(dirpath, image_file_name), 'r') as file:
                    img = np.asarray(file['main'])
                    voxel_offset = np.asarray(file['voxel_offset'], dtype=np.uint32)
                synapses_file_name = meta['synapse']['annotation']
                with open(synapses_file_name, 'r') as file:
                    synapses = json.load(synapses_file_name)
                bin_presyn, sampling_probability_map = self._annotation_to_volumes(
                    img, voxel_offset, synapses
                )
                subject = tio.Subject(
                    img = tio.ScalarImage(img),
                    tbar = tio.LabelMap(bin_presyn),
                    sampling_probability_map = tio.Image(sampling_probability_map)
                )
                subjects.append(subject)
                # self.datasets.append((img, synapses))
                # use the image voxel number as sampling weight
                # self.dataset_weights.append(len(img))
        dataset = tio.SubjectsDataset(subjects, transform=self.transform)
        print('number of volumes in dataset: ', len(dataset))
                
    def _annotation_to_volumes(img: np.ndarray, voxel_offset: np.ndarray, synapses: dict,
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
    dataset = Dataset('~/Dropbox\ \(Simons\ Foundation\)/40_gt/sample1')
