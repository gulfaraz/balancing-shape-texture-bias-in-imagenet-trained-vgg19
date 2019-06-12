from collections import defaultdict
from utils import *
from tqdm import tqdm
from PIL import Image, ImageFilter
import torch
from torch.utils.data import Dataset

DEBUG = False
VAL_FILES_1 = ['ILSVRC2012_val_00000709.JPEG', 'ILSVRC2012_val_00001496.JPEG', 'ILSVRC2012_val_00002072.JPEG', 'ILSVRC2012_val_00004612.JPEG', 'ILSVRC2012_val_00004719.JPEG', 'ILSVRC2012_val_00005207.JPEG', 'ILSVRC2012_val_00005362.JPEG', 'ILSVRC2012_val_00007023.JPEG', 'ILSVRC2012_val_00008778.JPEG', 'ILSVRC2012_val_00009683.JPEG', 'ILSVRC2012_val_00010417.JPEG', 'ILSVRC2012_val_00010598.JPEG', 'ILSVRC2012_val_00011952.JPEG', 'ILSVRC2012_val_00013410.JPEG', 'ILSVRC2012_val_00013918.JPEG', 'ILSVRC2012_val_00014432.JPEG', 'ILSVRC2012_val_00014773.JPEG', 'ILSVRC2012_val_00016918.JPEG', 'ILSVRC2012_val_00019119.JPEG', 'ILSVRC2012_val_00021675.JPEG', 'ILSVRC2012_val_00022337.JPEG', 'ILSVRC2012_val_00023393.JPEG', 'ILSVRC2012_val_00023696.JPEG', 'ILSVRC2012_val_00023786.JPEG', 'ILSVRC2012_val_00023871.JPEG', 'ILSVRC2012_val_00023906.JPEG', 'ILSVRC2012_val_00024000.JPEG', 'ILSVRC2012_val_00025168.JPEG', 'ILSVRC2012_val_00026403.JPEG', 'ILSVRC2012_val_00026412.JPEG', 'ILSVRC2012_val_00027189.JPEG', 'ILSVRC2012_val_00027448.JPEG', 'ILSVRC2012_val_00028628.JPEG', 'ILSVRC2012_val_00028970.JPEG', 'ILSVRC2012_val_00029864.JPEG', 'ILSVRC2012_val_00031185.JPEG', 'ILSVRC2012_val_00032637.JPEG', 'ILSVRC2012_val_00033582.JPEG', 'ILSVRC2012_val_00037997.JPEG', 'ILSVRC2012_val_00038683.JPEG', 'ILSVRC2012_val_00038976.JPEG', 'ILSVRC2012_val_00040192.JPEG', 'ILSVRC2012_val_00040600.JPEG', 'ILSVRC2012_val_00041427.JPEG', 'ILSVRC2012_val_00041579.JPEG', 'ILSVRC2012_val_00042232.JPEG', 'ILSVRC2012_val_00044374.JPEG', 'ILSVRC2012_val_00046770.JPEG', 'ILSVRC2012_val_00046964.JPEG', 'ILSVRC2012_val_00049735.JPEG']
VAL_FILES_2 = ['ILSVRC2012_val_00000473.JPEG', 'ILSVRC2012_val_00002112.JPEG', 'ILSVRC2012_val_00002556.JPEG', 'ILSVRC2012_val_00003992.JPEG', 'ILSVRC2012_val_00004416.JPEG', 'ILSVRC2012_val_00004853.JPEG', 'ILSVRC2012_val_00007618.JPEG', 'ILSVRC2012_val_00007645.JPEG', 'ILSVRC2012_val_00009568.JPEG', 'ILSVRC2012_val_00010131.JPEG', 'ILSVRC2012_val_00010440.JPEG', 'ILSVRC2012_val_00010458.JPEG', 'ILSVRC2012_val_00012789.JPEG', 'ILSVRC2012_val_00015989.JPEG', 'ILSVRC2012_val_00016525.JPEG', 'ILSVRC2012_val_00017567.JPEG', 'ILSVRC2012_val_00018520.JPEG', 'ILSVRC2012_val_00021834.JPEG', 'ILSVRC2012_val_00023896.JPEG', 'ILSVRC2012_val_00023931.JPEG', 'ILSVRC2012_val_00023936.JPEG', 'ILSVRC2012_val_00026659.JPEG', 'ILSVRC2012_val_00028684.JPEG', 'ILSVRC2012_val_00029513.JPEG', 'ILSVRC2012_val_00031522.JPEG', 'ILSVRC2012_val_00032026.JPEG', 'ILSVRC2012_val_00032929.JPEG', 'ILSVRC2012_val_00034096.JPEG', 'ILSVRC2012_val_00034487.JPEG', 'ILSVRC2012_val_00034956.JPEG', 'ILSVRC2012_val_00036326.JPEG', 'ILSVRC2012_val_00038147.JPEG', 'ILSVRC2012_val_00038171.JPEG', 'ILSVRC2012_val_00038252.JPEG', 'ILSVRC2012_val_00038499.JPEG', 'ILSVRC2012_val_00039860.JPEG', 'ILSVRC2012_val_00040260.JPEG', 'ILSVRC2012_val_00040398.JPEG', 'ILSVRC2012_val_00041667.JPEG', 'ILSVRC2012_val_00043400.JPEG', 'ILSVRC2012_val_00043691.JPEG', 'ILSVRC2012_val_00043864.JPEG', 'ILSVRC2012_val_00044147.JPEG', 'ILSVRC2012_val_00044263.JPEG', 'ILSVRC2012_val_00044757.JPEG', 'ILSVRC2012_val_00044794.JPEG', 'ILSVRC2012_val_00047066.JPEG', 'ILSVRC2012_val_00047181.JPEG', 'ILSVRC2012_val_00047971.JPEG', 'ILSVRC2012_val_00049267.JPEG']
CLASS_NAME_1 = 'n02124075'
CLASS_NAME_2 = 'n04067472'

def selector(line, split):
    if DEBUG:
        if split == 'train':
            return (CLASS_NAME_1 in line) or (CLASS_NAME_2 in line)
        elif split == 'val':
            return (line.strip() in VAL_FILES_1) or (line.strip() in VAL_FILES_2)
    else:
        return True

class BaseDataset(Dataset):

    def __init__(self, directory, split='train', transforms=None):
        self.datapoints = defaultdict(list)
        self.split = split
        self.directory = pathJoin(directory, split)
        self.datapoints = self.loadDataset()
        self.transforms = transforms

    def __len__(self):
        return len(self.datapoints)

    def __getitem__(self, idx):
        datapoint = self.loadDatapoint(idx)
        return datapoint

    def loadDatapoint(self, idx):
        raise NotImplementedError('Function "loadDatapoint" is not implemented')

    def loadDataset(self, name):
        raise NotImplementedError('Function "loadDataset" is not implemented')


class ImageNet200Dataset(BaseDataset):

    def __init__(self, directory, split='train', transforms=None):
        super().__init__(directory, split, transforms)
        self.descriptions = self.loadDescriptions()
        self.classes = self.loadClasses()
        self.groundtruths = self.loadValidationGroundtruths() if split == 'val' else []
        self.INDEX_IMAGE = 1
        self.INDEX_TARGET = 2
        self.INDEX_LABEL = 3

    def loadDatapoint(self, idx):
        filepath = self.datapoints[idx]
        if not os.path.isfile(filepath):
            filepath = filepath.replace('.JPEG', '.png')
        image = Image.open(filepath).convert('RGB')
        if self.split == 'val':
            groundtruth = self.groundtruths[idx]
        elif self.split == 'train':
            groundtruth = self.classes.index(filepath.split('/').pop().split('_')[0])
        if self.transforms:
            image = self.transforms(image)
        return (filepath, image, groundtruth, self.descriptions[groundtruth])

    def loadDataset(self):
        datapoints = []

        dataset_file_list_filename = '{}.txt'.format(self.split)
        dataset_file_list_path = os.path.join(self.directory, dataset_file_list_filename)

        with open(dataset_file_list_path, 'r') as dataset_file_list_file:
            for line in tqdm(dataset_file_list_file, total=sum(1 for line in open(dataset_file_list_path))):
                file_path = pathJoin(self.directory, self.sanitizeFilename(line))
                if selector(line, self.split):
                    datapoints.append(file_path)
        
        return datapoints
    
    def sanitizeFilename(self, filename):
        return filename.replace('"', '').strip()

    def loadDescriptions(self):
        descriptions = []

        descriptions_filename = 'wnids_with_descriptions.txt'
        descriptions_path = pathJoin(self.directory, '..', descriptions_filename)

        with open(descriptions_path, 'r') as descriptions_file:
            for line in descriptions_file:
                description_breakdown = line.split(' ')
                description_breakdown.pop(0)
                description = ' '.join(description_breakdown).strip()
                descriptions.append(description)

        return descriptions

    def loadValidationGroundtruths(self):
        groundtruths = []

        groundtruths_filename = 'val_groundtruth.txt'
        groundtruths_path = pathJoin(self.directory, '..', groundtruths_filename)

        with open(groundtruths_path, 'r') as groundtruths_file:
            for line in groundtruths_file:
                groundtruth_breakdown = line.split(' ')
                filename = groundtruth_breakdown.pop(0)
                if selector(filename, 'val'):
                    groundtruth = ' '.join(groundtruth_breakdown).strip()
                    groundtruths.append(int(groundtruth))

        return groundtruths

    def loadClasses(self):
        classes = []

        classes_filename = 'wnids.txt'
        classes_path = pathJoin(self.directory, '..', classes_filename)

        with open(classes_path, 'r') as classes_file:
            for line in classes_file:
                classes.append(line.strip())

        return classes

    def idx2label(self, class_idx):
        return self.classes[class_idx]


class ImageNetDataset(BaseDataset):

    def __init__(self, directory, split='train', transforms=None):
        super().__init__(directory, split, transforms)
        self.descriptions = self.loadDescriptions()
        self.classes = self.loadClasses()
        self.groundtruths = self.loadValidationGroundtruths() if split == 'val' else []
        self.INDEX_IMAGE = 1
        self.INDEX_TARGET = 2
        self.INDEX_LABEL = 3

    def loadDatapoint(self, idx):
        filepath = self.datapoints[idx]
        image = Image.open(filepath).convert('RGB')
        if self.split == 'val':
            groundtruth = self.groundtruths[idx]
        elif self.split == 'train':
            groundtruth = self.classes.index(filepath.split('/').pop().split('_')[0])
        if self.transforms:
            image = self.transforms(image)
        return (filepath, image, groundtruth, self.descriptions[groundtruth])

    def loadDataset(self):
        datapoints = []

        dataset_file_list_filename = 'ilsvrc2012{}.txt'.format(self.split)
        dataset_file_list_path = os.path.join(self.directory, dataset_file_list_filename)

        with open(dataset_file_list_path, 'r') as dataset_file_list_file:
            for line in tqdm(dataset_file_list_file, total=sum(1 for line in open(dataset_file_list_path))):
                file_path = pathJoin(self.directory, self.sanitizeFilename(line))
                datapoints.append(file_path)
        
        return datapoints
    
    def sanitizeFilename(self, filename):
        return filename.replace('"', '').strip()

    def loadDescriptions(self):
        descriptions = []

        descriptions_filename = 'synsets_with_descriptions.txt'
        descriptions_path = pathJoin(self.directory, '..', descriptions_filename)

        with open(descriptions_path, 'r') as descriptions_file:
            for line in descriptions_file:
                description_breakdown = line.split(' ')
                description_breakdown.pop(0)
                description = ' '.join(description_breakdown).strip()
                descriptions.append(description)

        return descriptions

    def loadValidationGroundtruths(self):
        groundtruths = []

        groundtruths_filename = 'validation_ground_truth.txt'
        groundtruths_path = pathJoin(self.directory, '..', groundtruths_filename)

        with open(groundtruths_path, 'r') as groundtruths_file:
            for line in groundtruths_file:
                groundtruth_breakdown = line.split(' ')
                groundtruth_breakdown.pop(0)
                groundtruth = ' '.join(groundtruth_breakdown).strip()
                groundtruths.append(int(groundtruth))

        return groundtruths

    def loadClasses(self):
        classes = []

        classes_filename = 'synsets.txt'
        classes_path = pathJoin(self.directory, '..', classes_filename)

        with open(classes_path, 'r') as classes_file:
            for line in classes_file:
                classes.append(line.strip())

        return classes


class ImageNet200PairDataset(BaseDataset):

    def __init__(self, input_directory, target_directory, split='train', transforms=None, target_type=None, target_transforms=None):
        assert target_type in ['nonstylized', 'stylized', 'highpass', 'swap', 'mix'], 'Unknown target type ({}) for pair dataset'.format(target_type)
        self.target_directory = pathJoin(target_directory, split)
        super().__init__(input_directory, split, transforms)
        self.target_type = target_type
        self.descriptions = self.loadDescriptions()
        self.classes = self.loadClasses()
        self.groundtruths = self.loadValidationGroundtruths() if split == 'val' else []
        self.INDEX_IMAGE = 2
        self.INDEX_TARGET_IMAGE = 3
        self.INDEX_TARGET = 4
        self.INDEX_LABEL = 5
        self.target_transforms = target_transforms

    def loadImage(self, filepath):
        if not os.path.isfile(filepath):
            filepath = filepath.replace('.JPEG', '.png')
        return Image.open(filepath).convert('RGB')

    def loadDatapoint(self, idx):
        input_filepath = self.datapoints[idx][0]
        target_filepath = self.datapoints[idx][1]
        input_image = self.loadImage(input_filepath)
        if self.target_type == 'nonstylized':
            target_image = input_image
        elif self.target_type == 'stylized':
            target_image = self.loadImage(target_filepath)
        elif self.target_type == 'highpass':
            target_image = input_image.filter(ImageFilter.FIND_EDGES)
        elif self.target_type == 'swap':
            target_image = input_image
            input_image = self.loadImage(target_filepath)
        elif self.target_type == 'mix':
            target_image = self.loadImage(target_filepath)
            if torch.rand(1) > 0.5:
                input_image, target_image = target_image, input_image

        if self.split == 'val':
            groundtruth = self.groundtruths[idx]
        elif self.split == 'train':
            groundtruth = self.classes.index(input_filepath.split('/').pop().split('_')[0])
        if self.transforms:
            input_image = self.transforms(input_image)
            target_image = self.target_transforms(target_image)
        return (input_filepath, target_filepath, input_image, target_image, groundtruth, self.descriptions[groundtruth])

    def loadDataset(self):
        datapoints = []

        dataset_file_list_filename = '{}.txt'.format(self.split)
        dataset_file_list_path = os.path.join(self.directory, dataset_file_list_filename)

        with open(dataset_file_list_path, 'r') as dataset_file_list_file:
            for line in tqdm(dataset_file_list_file, total=sum(1 for line in open(dataset_file_list_path))):
                file_path = pathJoin(self.directory, self.sanitizeFilename(line))
                target_file_path = pathJoin(self.target_directory, self.sanitizeFilename(line))
                if selector(line, self.split):
                    datapoints.append([file_path, target_file_path])
        
        return datapoints
    
    def sanitizeFilename(self, filename):
        return filename.replace('"', '').strip()

    def loadDescriptions(self):
        descriptions = []

        descriptions_filename = 'wnids_with_descriptions.txt'
        descriptions_path = pathJoin(self.directory, '..', descriptions_filename)

        with open(descriptions_path, 'r') as descriptions_file:
            for line in descriptions_file:
                description_breakdown = line.split(' ')
                description_breakdown.pop(0)
                description = ' '.join(description_breakdown).strip()
                descriptions.append(description)

        return descriptions

    def loadValidationGroundtruths(self):
        groundtruths = []

        groundtruths_filename = 'val_groundtruth.txt'
        groundtruths_path = pathJoin(self.directory, '..', groundtruths_filename)

        with open(groundtruths_path, 'r') as groundtruths_file:
            for line in groundtruths_file:
                groundtruth_breakdown = line.split(' ')
                filename = groundtruth_breakdown.pop(0)
                if selector(filename, 'val'):
                    groundtruth = ' '.join(groundtruth_breakdown).strip()
                    groundtruths.append(int(groundtruth))

        return groundtruths

    def loadClasses(self):
        classes = []

        classes_filename = 'wnids.txt'
        classes_path = pathJoin(self.directory, '..', classes_filename)

        with open(classes_path, 'r') as classes_file:
            for line in classes_file:
                classes.append(line.strip())

        return classes

    def idx2label(self, class_idx):
        return self.classes[class_idx]

class DeNormalize(object):
    # Source: https://discuss.pytorch.org/t/simple-way-to-inverse-transform-normalization/4821/3
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        tensor = image.clone()
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
            # The normalize code -> t.sub_(m).div_(s)
        return tensor


class CelebADataset(BaseDataset):

    def __init__(self, root_directory, split='train', transforms=None):
        self.root_directory = root_directory
        super().__init__(root_directory, split, transforms)
        self.INDEX_IMAGE = 1
        self.INDEX_TARGET_IMAGE = 1

    def loadImage(self, filepath):
        return Image.open(filepath)

    def loadDatapoint(self, idx):
        input_filepath = self.datapoints[idx]
        input_image = self.loadImage(input_filepath)

        if self.transforms:
            input_image = self.transforms(input_image)

        return (input_filepath, input_image)

    def loadDataset(self):
        all_datapoints = [ pathJoin(self.root_directory, filename) for filename in os.listdir(self.root_directory) ]
        return all_datapoints[:1000] if DEBUG else all_datapoints


def find_normalization_values(dataset, image_index):
    loader = DataLoader(dataset, batch_size=len(dataset), shuffle=True)
    all_images = None
    for i in loader:
        all_images = i[image_index]
    mean_image = all_images.mean(0)
    std_image = all_images.std(0)
    mean = mean_image.view(mean_image.size(0), -1).mean(-1)
    std = std_image.view(std_image.size(0), -1).mean(-1)
    return {
        'mean': mean.numpy().tolist(),
        'std': std.numpy().tolist()
    }

