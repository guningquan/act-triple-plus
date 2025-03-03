import numpy as np
import torch
import os
import h5py
import pickle
import fnmatch
import cv2
from time import time
from torch.utils.data import TensorDataset, DataLoader
import torchvision.transforms as transforms
import random
import IPython
e = IPython.embed

def flatten_list(l):
    return [item for sublist in l for item in sublist]


def random_vertical_flip(img, p=0.5):
    """随机垂直翻转"""
    if random.random() < p:
        img = cv2.flip(img, 0)  # 0 表示垂直翻转
    return img


def random_rotation(img, degrees=(-5, 5)):
    """随机旋转，角度在指定区间内"""
    angle = random.uniform(degrees[0], degrees[1])
    h, w = img.shape[:2]
    center = (w // 2, h // 2)
    # scale=1.0 不缩放
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    # 保持尺寸不变
    rotated = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
    return rotated


def random_resized_crop(img, output_size=(480, 640), scale=(0.9, 1.0)):
    """
    随机裁剪并缩放到指定输出尺寸，scale 表示裁剪区域相对于原图的面积比例范围。
    这里是近似实现，可根据需求自行优化。
    """
    h, w = img.shape[:2]
    # 随机选一个缩放比例
    scale_factor = random.uniform(scale[0], scale[1])

    # 计算裁剪区域大小
    new_h = int(h * scale_factor)
    new_w = int(w * scale_factor)

    # 随机决定裁剪起点
    top = random.randint(0, h - new_h)
    left = random.randint(0, w - new_w)

    # 裁剪
    cropped = img[top: top + new_h, left: left + new_w]

    # 缩放回目标尺寸
    resized = cv2.resize(cropped, (output_size[1], output_size[0]), interpolation=cv2.INTER_LINEAR)
    return resized


def random_brightness_contrast(img, brightness=0.2, contrast=0.3):
    """
    随机调整亮度和对比度。
    brightness 和 contrast 这里先简单视为“最大加减量”和“最大乘除比例”，
    实际可根据需要改成更精细的实现。
    """
    # alpha: 对比度系数 (1 ± contrast_range)
    # beta:  亮度偏置   (± brightness_range * 255)
    alpha = 1.0 + random.uniform(-contrast, contrast)
    beta = random.uniform(-brightness, brightness) * 255

    # img 需要是浮点运算再裁剪回 [0,255]
    adj = img.astype(np.float32) * alpha + beta
    adj = np.clip(adj, 0, 255).astype(np.uint8)
    return adj


def random_affine(img, max_translate=(0.1, 0.1)):
    """
    随机仿射变换，这里只演示随机平移，角度固定为0。
    若要包含旋转或缩放，可自行修改。
    max_translate 表示最大平移系数，相对于图像宽高。
    """
    h, w = img.shape[:2]
    max_trans_x = max_translate[0] * w
    max_trans_y = max_translate[1] * h

    tx = random.uniform(-max_trans_x, max_trans_x)
    ty = random.uniform(-max_trans_y, max_trans_y)

    # 构造仿射矩阵: 仅平移
    M = np.float32([[1, 0, tx],
                    [0, 1, ty]])

    # 应用仿射变换
    shifted = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
    return shifted

# def random_affine(img, max_translate=(0.1, 0.1)):
#     """
#     Apply a random affine transformation with wrapping border fill.
#     This function performs random translation only (no rotation or scaling).
#
#     Args:
#         img: Input image (numpy array).
#         max_translate: Maximum translation factor as a fraction of image width and height
#                        (e.g., (0.1, 0.1) allows up to 10% translation of width and height).
#
#     Returns:
#         shifted: Image after applying the affine transformation with wrapping fill.
#     """
#     h, w = img.shape[:2]  # Get the height and width of the input image
#     max_trans_x = max_translate[0] * w  # Maximum horizontal translation
#     max_trans_y = max_translate[1] * h  # Maximum vertical translation
#
#     # Generate random translation values within the range [-max_trans_x, max_trans_x] and [-max_trans_y, max_trans_y]
#     tx = random.uniform(-max_trans_x, max_trans_x)  # Random horizontal shift
#     ty = random.uniform(-max_trans_y, max_trans_y)  # Random vertical shift
#
#     # Construct the affine transformation matrix for translation
#     M = np.float32([[1, 0, tx],  # Horizontal translation (tx)
#                     [0, 1, ty]])  # Vertical translation (ty)
#
#     # Apply the affine transformation using wrapping border fill
#     shifted = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR,
#                              borderMode=cv2.BORDER_WRAP)  # Use BORDER_WRAP for wrapping fill
#     return shifted

def random_augment_pipeline(img):
    """
    组合多个随机操作的示例，可根据需要增删。
    输出图像的尺寸仍是 (480, 640, 3)。
    """
    # 1. 随机垂直翻转
    # img = random_vertical_flip(img, p=0.5)
    # 2. 随机旋转
    # img = random_rotation(img, degrees=(-5, 5))
    # 3. 随机裁剪并缩放
    # img = random_resized_crop(img, output_size=(480, 640), scale=(0.9, 1.0))
    # 4. 随机亮度+对比度
    # img = random_brightness_contrast(img, brightness=0.2, contrast=0.3)
    # 5. 随机平移仿射
    img = random_affine(img, max_translate=(0.1, 0.1))

    return img

class EpisodicDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_path_list, camera_names, norm_stats, episode_ids, episode_len, chunk_size, policy_class):
        super(EpisodicDataset).__init__()
        self.episode_ids = episode_ids
        self.dataset_path_list = dataset_path_list
        self.camera_names = camera_names
        self.norm_stats = norm_stats
        self.episode_len = episode_len
        self.chunk_size = chunk_size
        self.cumulative_len = np.cumsum(self.episode_len)
        self.max_episode_len = max(episode_len)
        self.policy_class = policy_class
        if self.policy_class == 'Diffusion':
            self.augment_images = True
        else:
            self.augment_images = False
        self.transformations = None
        self.__getitem__(0) # initialize self.is_sim and self.transformations
        self.is_sim = False

    # def __len__(self):
    #     return sum(self.episode_len)

    def _locate_transition(self, index):
        assert index < self.cumulative_len[-1]
        episode_index = np.argmax(self.cumulative_len > index) # argmax returns first True index
        start_ts = index - (self.cumulative_len[episode_index] - self.episode_len[episode_index])
        episode_id = self.episode_ids[episode_index]
        return episode_id, start_ts

    def __getitem__(self, index):
        episode_id, start_ts = self._locate_transition(index)
        dataset_path = self.dataset_path_list[episode_id]
        try:
            # print(dataset_path)
            with h5py.File(dataset_path, 'r') as root:
                try: # some legacy data does not have this attribute
                    is_sim = root.attrs['sim']
                except:
                    is_sim = False
                compressed = root.attrs.get('compress', False)

                # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
                if '/base_action' in root:
                    base_action = root['/base_action'][()]
                    base_action = preprocess_base_action(base_action)
                    action = np.concatenate([root['/action'][()], base_action], axis=-1)
                else:
                    action = root['/action'][()]
                    dummy_base_action = np.zeros([action.shape[0], 2])
                    action = np.concatenate([action, dummy_base_action], axis=-1)

                # action = root['/action'][()] # gnq
                # dummy_base_action = np.zeros([action.shape[0], 2])
                # action = np.concatenate([action, dummy_base_action], axis=-1)
                # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

                original_action_shape = action.shape
                episode_len = original_action_shape[0]
                # get observation at start_ts only
                qpos = root['/observations/qpos'][start_ts]
                qvel = root['/observations/qvel'][start_ts]
                image_dict = dict()
                # for cam_name in self.camera_names:
                #     image_dict[cam_name] = root[f'/observations/images/{cam_name}'][start_ts]
                #
                # if compressed:
                #     for cam_name in image_dict.keys():
                #         decompressed_image = cv2.imdecode(image_dict[cam_name], 1)
                #         image_dict[cam_name] = np.array(decompressed_image)
                # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
                default_image_shape = None  # @gnq 用于存储解压后的图像形状
                for cam_name in self.camera_names:
                    image_path = f'/observations/images/{cam_name}'
                    if image_path in root:  # 检查键是否存在
                        image_dict[cam_name] = root[image_path][start_ts]
                    else:
                        # 缺失摄像头数据，赋值为默认值
                        # print("no image, using zero")
                        image_dict[cam_name] = np.zeros((1,), dtype=np.uint8)  # 临时占位
                        # print("no image")
                        # exit()
                if compressed:
                    for cam_name in image_dict.keys():
                        # 如果全为 0，则跳过解压缩，但调整尺寸
                        if np.all(image_dict[cam_name] == 0):  # 判断是否全为 0
                            if default_image_shape is not None:
                                # 调整尺寸为其他解压后的图像尺寸
                                image_dict[cam_name] = np.zeros(default_image_shape, dtype=np.uint8)
                            else:
                                # 如果没有参考尺寸，初始化为默认尺寸
                                image_dict[cam_name] = np.zeros((224, 224, 3), dtype=np.uint8)  # 默认尺寸
                            continue
                        # 解压正常的压缩图像
                        decompressed_image = cv2.imdecode(image_dict[cam_name], 1)
                        decompressed_image = np.array(decompressed_image)
                        # 记录第一个解压图像的形状，用作参考
                        if default_image_shape is None:
                            default_image_shape = decompressed_image.shape
                        # 更新解压后的图像
                        image_dict[cam_name] = decompressed_image
                # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

                # get all actions after and including start_ts
                if is_sim:
                    action = action[start_ts:]
                    action_len = episode_len - start_ts
                else:
                    action = action[max(0, start_ts - 1):] # hack, to make timesteps more aligned
                    action_len = episode_len - max(0, start_ts - 1) # hack, to make timesteps more aligned

            # self.is_sim = is_sim
            padded_action = np.zeros((self.max_episode_len, original_action_shape[1]), dtype=np.float32)
            padded_action[:action_len] = action
            is_pad = np.zeros(self.max_episode_len)
            is_pad[action_len:] = 1

            padded_action = padded_action[:self.chunk_size]
            is_pad = is_pad[:self.chunk_size]


            #
            # gel_transforms = [
            #     transforms.RandomVerticalFlip(p=0.5),  # 随机垂直翻转
            #     transforms.RandomRotation(degrees=[-5.0, 5.0], expand=False),
            #     transforms.RandomResizedCrop(size=[480,640], scale=(0.9, 1.0)),  # size_modify
            #     transforms.ColorJitter(brightness=0.2, contrast=0.3),  # , hue=0.08,brightness=0.3,
            #     transforms.RandomAffine(degrees=0, translate=(0.1, 0.1))
            # ]

            # new axis for different cameras
            all_cam_images = []

            for cam_name in self.camera_names:
                # all_cam_images.append(image_dict[cam_name])
                if 'gel' in cam_name:
                    gel_image_data = image_dict[cam_name]
                    # print("gel is gray image")
                    # gray_image = 0.299 * gel_image_data[0] + 0.587 * gel_image_data[1] + 0.114 * gel_image_data[2]  # @gnq gray
                    # gel_image_data = torch.stack([gray_image, gray_image, gray_image], dim=0)
                    # gel_image_data = random_augment_pipeline(gel_image_data)
                    all_cam_images.append(gel_image_data)
                    # cv2.imshow('Original', gel_image_data)
                    # cv2.imshow('Augmented', gel_image_data_aug)
                    # cv2.waitKey(0)
                    # cv2.destroyAllWindows()
                else:
                    all_cam_images.append(image_dict[cam_name])
                    # cv2.imshow('OK', image_dict[cam_name])
                    # cv2.waitKey(0)
                    # cv2.destroyAllWindows()

            all_cam_images = np.stack(all_cam_images, axis=0)

            # construct observations
            image_data = torch.from_numpy(all_cam_images)
            qpos_data = torch.from_numpy(qpos).float()
            action_data = torch.from_numpy(padded_action).float()
            is_pad = torch.from_numpy(is_pad).bool()

            # channel last
            image_data = torch.einsum('k h w c -> k c h w', image_data)

            # augmentation
            if self.transformations is None:
                # print("*"*100)
                print('Initializing transformations')
                original_size = image_data.shape[2:]
                ratio = 0.95
                self.transformations = [
                    transforms.RandomCrop(size=[int(original_size[0] * ratio), int(original_size[1] * ratio)]),
                    transforms.Resize(original_size, antialias=True),
                    transforms.RandomRotation(degrees=[-5.0, 5.0], expand=False),
                    transforms.ColorJitter(brightness=0.3, contrast=0.4, saturation=0.5) #, hue=0.08)
                ]
            # self.augment_images = True
            if self.augment_images:
                for transform in self.transformations:
                    image_data = transform(image_data)
                    print("augment_images!!!!!!!!!")
            # normalize image and change dtype to float
            image_data = image_data / 255.0

            if self.policy_class == 'Diffusion':
                # normalize to [-1, 1]
                action_data = ((action_data - self.norm_stats["action_min"]) / (self.norm_stats["action_max"] - self.norm_stats["action_min"])) * 2 - 1
            else:
                # normalize to mean 0 std 1
                action_data = (action_data - self.norm_stats["action_mean"]) / self.norm_stats["action_std"]

            qpos_data = (qpos_data - self.norm_stats["qpos_mean"]) / self.norm_stats["qpos_std"]

        except:
            print(f'Error loading {dataset_path} in __getitem__')
            quit()

        # print(image_data.dtype, qpos_data.dtype, action_data.dtype, is_pad.dtype)
        return image_data, qpos_data, action_data, is_pad


def get_norm_stats(dataset_path_list):
    all_qpos_data = []
    all_action_data = []
    all_episode_len = []

    for dataset_path in dataset_path_list:
        try:
            with h5py.File(dataset_path, 'r') as root:
                qpos = root['/observations/qpos'][()]
                qvel = root['/observations/qvel'][()]
                if '/base_action' in root:
                    base_action = root['/base_action'][()]
                    base_action = preprocess_base_action(base_action)
                    action = np.concatenate([root['/action'][()], base_action], axis=-1)
                else:
                    action = root['/action'][()]
                    dummy_base_action = np.zeros([action.shape[0], 2])
                    action = np.concatenate([action, dummy_base_action], axis=-1)
        except Exception as e:
            print(f'Error loading {dataset_path} in get_norm_stats')
            print(e)
            quit()
        all_qpos_data.append(torch.from_numpy(qpos))
        all_action_data.append(torch.from_numpy(action))
        all_episode_len.append(len(qpos))
    all_qpos_data = torch.cat(all_qpos_data, dim=0)
    all_action_data = torch.cat(all_action_data, dim=0)

    # normalize action data
    action_mean = all_action_data.mean(dim=[0]).float()
    action_std = all_action_data.std(dim=[0]).float()
    action_std = torch.clip(action_std, 1e-2, np.inf) # clipping

    # normalize qpos data
    qpos_mean = all_qpos_data.mean(dim=[0]).float()
    qpos_std = all_qpos_data.std(dim=[0]).float()
    qpos_std = torch.clip(qpos_std, 1e-2, np.inf) # clipping

    action_min = all_action_data.min(dim=0).values.float()
    action_max = all_action_data.max(dim=0).values.float()

    eps = 0.0001
    stats = {"action_mean": action_mean.numpy(), "action_std": action_std.numpy(),
             "action_min": action_min.numpy() - eps,"action_max": action_max.numpy() + eps,
             "qpos_mean": qpos_mean.numpy(), "qpos_std": qpos_std.numpy(),
             "example_qpos": qpos}

    return stats, all_episode_len

def find_all_hdf5(dataset_dir, skip_mirrored_data):
    hdf5_files = []
    for root, dirs, files in os.walk(dataset_dir):
        for filename in fnmatch.filter(files, '*.hdf5'):
            if 'features' in filename: continue
            if skip_mirrored_data and 'mirror' in filename:
                continue
            hdf5_files.append(os.path.join(root, filename))
    print(f'Found {len(hdf5_files)} hdf5 files')
    return hdf5_files

def BatchSampler(batch_size, episode_len_l, sample_weights):
    sample_probs = np.array(sample_weights) / np.sum(sample_weights) if sample_weights is not None else None
    sum_dataset_len_l = np.cumsum([0] + [np.sum(episode_len) for episode_len in episode_len_l])
    while True:
        batch = []
        for _ in range(batch_size):
            episode_idx = np.random.choice(len(episode_len_l), p=sample_probs)
            step_idx = np.random.randint(sum_dataset_len_l[episode_idx], sum_dataset_len_l[episode_idx + 1])
            batch.append(step_idx)
        yield batch

def load_data(dataset_dir_l, name_filter, camera_names, batch_size_train, batch_size_val, chunk_size, skip_mirrored_data=False, load_pretrain=False, policy_class=None, stats_dir_l=None, sample_weights=None, train_ratio=0.99):
    if type(dataset_dir_l) == str:
        dataset_dir_l = [dataset_dir_l]
    dataset_path_list_list = [find_all_hdf5(dataset_dir, skip_mirrored_data) for dataset_dir in dataset_dir_l]
    num_episodes_0 = len(dataset_path_list_list[0])
    dataset_path_list = flatten_list(dataset_path_list_list)
    dataset_path_list = [n for n in dataset_path_list if name_filter(n)]
    num_episodes_l = [len(dataset_path_list) for dataset_path_list in dataset_path_list_list]
    num_episodes_cumsum = np.cumsum(num_episodes_l)

    # obtain train test split on dataset_dir_l[0]
    shuffled_episode_ids_0 = np.random.permutation(num_episodes_0)
    train_episode_ids_0 = shuffled_episode_ids_0[:int(train_ratio * num_episodes_0)]
    val_episode_ids_0 = shuffled_episode_ids_0[int(train_ratio * num_episodes_0):]
    train_episode_ids_l = [train_episode_ids_0] + [np.arange(num_episodes) + num_episodes_cumsum[idx] for idx, num_episodes in enumerate(num_episodes_l[1:])]
    val_episode_ids_l = [val_episode_ids_0]
    train_episode_ids = np.concatenate(train_episode_ids_l)
    val_episode_ids = np.concatenate(val_episode_ids_l)
    print(f'\n\nData from: {dataset_dir_l}\n- Train on {[len(x) for x in train_episode_ids_l]} episodes\n- Test on {[len(x) for x in val_episode_ids_l]} episodes\n\n')


    _, all_episode_len = get_norm_stats(dataset_path_list)
    train_episode_len_l = [[all_episode_len[i] for i in train_episode_ids] for train_episode_ids in train_episode_ids_l]
    val_episode_len_l = [[all_episode_len[i] for i in val_episode_ids] for val_episode_ids in val_episode_ids_l]
    train_episode_len = flatten_list(train_episode_len_l)
    val_episode_len = flatten_list(val_episode_len_l)
    if stats_dir_l is None:
        stats_dir_l = dataset_dir_l
    elif type(stats_dir_l) == str:
        stats_dir_l = [stats_dir_l]
    norm_stats, _ = get_norm_stats(flatten_list([find_all_hdf5(stats_dir, skip_mirrored_data) for stats_dir in stats_dir_l]))
    print(f'Norm stats from: {stats_dir_l}')

    batch_sampler_train = BatchSampler(batch_size_train, train_episode_len_l, sample_weights)
    batch_sampler_val = BatchSampler(batch_size_val, val_episode_len_l, None)

    # print(f'train_episode_len: {train_episode_len}, val_episode_len: {val_episode_len}, train_episode_ids: {train_episode_ids}, val_episode_ids: {val_episode_ids}')

    # construct dataset and dataloader
    train_dataset = EpisodicDataset(dataset_path_list, camera_names, norm_stats, train_episode_ids, train_episode_len, chunk_size, policy_class)
    val_dataset = EpisodicDataset(dataset_path_list, camera_names, norm_stats, val_episode_ids, val_episode_len, chunk_size, policy_class)
    train_num_workers = (8 if os.getlogin() == 'guningquan' else 16) if train_dataset.augment_images else 2
    val_num_workers = 8 if train_dataset.augment_images else 2
    print(f'Augment images: {train_dataset.augment_images}, train_num_workers: {train_num_workers}, val_num_workers: {val_num_workers}')
    train_dataloader = DataLoader(train_dataset, batch_sampler=batch_sampler_train, pin_memory=True, num_workers=train_num_workers, prefetch_factor=2)
    val_dataloader = DataLoader(val_dataset, batch_sampler=batch_sampler_val, pin_memory=True, num_workers=val_num_workers, prefetch_factor=2)

    return train_dataloader, val_dataloader, norm_stats, train_dataset.is_sim

def calibrate_linear_vel(base_action, c=None):
    if c is None:
        c = 0.0 # 0.19
    v = base_action[..., 0]
    w = base_action[..., 1]
    base_action = base_action.copy()
    base_action[..., 0] = v - c * w
    return base_action

def smooth_base_action(base_action):
    return np.stack([
        np.convolve(base_action[:, i], np.ones(5)/5, mode='same') for i in range(base_action.shape[1])
    ], axis=-1).astype(np.float32)

def preprocess_base_action(base_action):
    # base_action = calibrate_linear_vel(base_action)
    base_action = smooth_base_action(base_action)

    return base_action

def postprocess_base_action(base_action):
    linear_vel, angular_vel = base_action
    linear_vel *= 1.0
    angular_vel *= 1.0
    # angular_vel = 0
    # if np.abs(linear_vel) < 0.05:
    #     linear_vel = 0
    return np.array([linear_vel, angular_vel])

### env utils

def sample_box_pose():
    x_range = [0.0, 0.2]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]

    ranges = np.vstack([x_range, y_range, z_range])
    cube_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    cube_quat = np.array([1, 0, 0, 0])
    return np.concatenate([cube_position, cube_quat])

def sample_insertion_pose():
    # Peg
    x_range = [0.1, 0.2]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]

    ranges = np.vstack([x_range, y_range, z_range])
    peg_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    peg_quat = np.array([1, 0, 0, 0])
    peg_pose = np.concatenate([peg_position, peg_quat])

    # Socket
    x_range = [-0.2, -0.1]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]

    ranges = np.vstack([x_range, y_range, z_range])
    socket_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    socket_quat = np.array([1, 0, 0, 0])
    socket_pose = np.concatenate([socket_position, socket_quat])

    return peg_pose, socket_pose

### helper functions

def compute_dict_mean(epoch_dicts):
    result = {k: None for k in epoch_dicts[0]}
    num_items = len(epoch_dicts)
    for k in result:
        value_sum = 0
        for epoch_dict in epoch_dicts:
            value_sum += epoch_dict[k]
        result[k] = value_sum / num_items
    return result

def detach_dict(d):
    new_d = dict()
    for k, v in d.items():
        new_d[k] = v.detach()
    return new_d

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)


# def calculate_weights_exponential(actions_for_curr_step, w_max=0.01):
#     k = -np.log(w_max) / (actions_for_curr_step - 1)  # 动态计算衰减系数 k
#     exp_weights = np.exp(-k * np.arange(actions_for_curr_step))  # 指数衰减权重
#     exp_weights = exp_weights / exp_weights.sum()  # 归一化
#     return torch.tensor(exp_weights, dtype=torch.float32)

def calculate_weights_exponential(num_steps, w_max=0.01):
    if num_steps == 1:  # 如果只有一个步骤，权重为 1
        return torch.tensor([1.0], dtype=torch.float32)
    k = -np.log(w_max) / (num_steps - 1)  # 动态计算衰减系数 k
    exp_weights = np.exp(-k * np.arange(num_steps))  # 指数衰减权重
    exp_weights = exp_weights / exp_weights.sum()  # 归一化
    return torch.tensor(exp_weights, dtype=torch.float32)


def calculate_weights_aloha(actions_for_curr_step, k=0.01):
    exp_weights = np.exp(-k * np.arange(actions_for_curr_step))  # ALOHA 衰减
    exp_weights = exp_weights / exp_weights.sum()  # 归一化
    return torch.tensor(exp_weights, dtype=torch.float32)


def calculate_weights_gaussian(actions_for_curr_step, sigma=10):
    distances = np.arange(actions_for_curr_step)  # 距离数组
    gaussian_weights = np.exp(-distances ** 2 / (2 * sigma ** 2))  # 高斯分布公式
    gaussian_weights = gaussian_weights / gaussian_weights.sum()  # 归一化
    return torch.tensor(gaussian_weights, dtype=torch.float32)


def apply_weights_to_all_l1(all_l1, weight_function, actions_for_curr_step, **kwargs):
    """
    对 all_l1 的每个 num_queries 应用权重。

    参数:
        all_l1 (torch.Tensor): 尺寸为 [batch_size, num_queries, 16] 的 L1 损失张量。
        weight_function (function): 权重计算函数。
        actions_for_curr_step (int): 当前步长的动作数 (num_queries)。
        **kwargs: 传递给权重计算函数的额外参数。

    返回:
        torch.Tensor: 应用权重后的 L1 损失张量，尺寸仍为 [batch_size, num_queries, 16]。
    """
    # 获取权重，形状为 [num_queries]
    weights = weight_function(actions_for_curr_step, **kwargs)  # 权重形状为 [num_queries]

    # 将权重扩展为 [1, num_queries, 1]，以便与 all_l1 广播匹配
    weights = weights.to(all_l1.device).view(1, -1, 1)

    # 对 all_l1 的每个位置按权重缩放
    weighted_l1 = all_l1 * weights  # 广播乘法

    return weighted_l1  # 返回加权后的 L1 张量


def calculate_weighted_action(actions_for_curr_step, weight_function, **kwargs):
    # if actions_for_curr_step.ndim == 1:
    #     actions_for_curr_step = actions_for_curr_step.unsqueeze(dim=1)  # 扩展为二维张量
    # if len(actions_for_curr_step) == 1:  # TODO
    #     return actions_for_curr_step
    # 计算权重
    exp_weights = weight_function(len(actions_for_curr_step), **kwargs)  # 权重计算函数
    # print("before exp_weights", exp_weights)
    exp_weights = exp_weights.flip(dims=[0])  # reverse the value
    # print("after exp_weights", exp_weights)
    exp_weights = exp_weights.to(actions_for_curr_step.device).unsqueeze(dim=1)  # 转为 Torch 张量并扩展维度
    # 计算加权后的动作
    # print("after exp_weights", exp_weights)
    # input("enter to continue...")
    raw_action = (actions_for_curr_step * exp_weights).sum(dim=0, keepdim=True)
    return raw_action
