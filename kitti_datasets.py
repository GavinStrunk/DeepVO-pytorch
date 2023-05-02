import math
import numpy as np
from PIL import Image
import pykitti
import torch
from torch.utils.data import Dataset
import torchvision



# @todo add support for different cameras
# @todo add lidar dataset
def matrix_to_pose(matrix, degrees=False):
    # Extract the rotation matrix from the homogeneous matrix
    rotation_matrix = matrix[:3, :3]

    # Extract the translation vector from the homogeneous matrix
    translation = matrix[:3, 3]

    # Calculate the Euler angles from the rotation matrix
    sy = math.sqrt(rotation_matrix[0, 0] * rotation_matrix[0, 0] + rotation_matrix[1, 0] * rotation_matrix[1, 0])
    singular = sy < 1e-6

    if not singular:
        x = math.atan2(rotation_matrix[2, 1], rotation_matrix[2, 2])
        y = math.atan2(-rotation_matrix[2, 0], sy)
        z = math.atan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
    else:
        x = math.atan2(-rotation_matrix[1, 2], rotation_matrix[1, 1])
        y = math.atan2(-rotation_matrix[2, 0], sy)
        z = 0

    # Convert the Euler angles to degrees
    if degrees:
        x = math.degrees(x)
        y = math.degrees(y)
        z = math.degrees(z)

    # Create a 6-dimensional array with the translation vector and Euler angles
    output = np.array([translation[0], translation[1], translation[2], x, y, z])

    return output


class ImageSequenceDataset(Dataset):
    KITTI_MEANS = (0.19007764876619865, 0.15170388157131237, 0.10659445665650864)
    KITTI_STDS = (0.2610784009469139, 0.25729316928935814, 0.25163823815039915)
    def __init__(self, dataset_dir, sequences, resize=None, num_stack=1):

        assert num_stack >= 1

        self.dataset_dir = dataset_dir
        self.sequences = sequences
        self.resize = resize
        self.num_stack = num_stack

        # Initialize Data Variables
        self.poses = []
        self.images = []

        # Load the sequences
        for seq in self.sequences:
            self._load_sequence(seq)

        assert len(self.poses) == len(self.images)

        # Set up image transformations
        if resize is not None:
            self.tranforms = torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean=self.KITTI_MEANS, std=self.KITTI_STDS),
                torchvision.transforms.Resize(self.resize, antialias=True)
            ])
        else:
            self.tranforms = torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((self.KITTI_MEANS,), (self.KITTI_STDS,)),
            ])

    def __len__(self):
        return len(self.poses)

    def __getitem__(self, index):
        # Convert Pose to Tensor
        label = torch.Tensor(matrix_to_pose(self.poses[index]))

        # Load Images and convert to Tensor
        img_files = self.images[index]
        image = []
        for ifile in img_files:
            img = Image.open(ifile)
            img = self.tranforms(img)
            image.append(img)

        images = torch.cat(image)

        return images, label

    def _load_sequence(self, sequence):
        dataset = pykitti.odometry(self.dataset_dir, sequence)

        # Append the Poses to the list
        if self.num_stack > 1:
            # Subtract the number of measurements to stack
            poses = dataset.poses[:-(self.num_stack - 1)]
        else:
            poses = dataset.poses

        self.poses += poses

        # Append Image file names list
        images = []
        if self.num_stack > 1:
            imgs = []
            window_size = len(dataset.cam3_files) - self.num_stack + 1
            for i in range(self.num_stack):
                img = dataset.cam3_files[i:i + window_size]
                imgs.append(img)

            images += np.array(imgs).T.tolist()
        else:
            images = np.array(dataset.cam3_files)

        self.images += images


if __name__ == '__main__':
    basedir = '/media/Data/Datasets/KITTI/odometry/dataset'
    sequence = ['00', '01']

    # dataset = pykitti.odometry(basedir, sequence)
    dataset = ImageSequenceDataset(dataset_dir=basedir, sequences=sequence, resize=(1280, 384), num_stack=2)

    imgs, pose = dataset[0]
    print('Image Size: {}'.format(imgs.shape))
    print('Pose: {}'.format(pose))

    transform = torchvision.transforms.ToPILImage()
    im = transform(imgs[0:3])
    im.show()
