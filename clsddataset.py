import os
from PIL import Image
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms

class CLBaseDataset(Dataset):
    def __init__(self,
                 data_root=None,  # 数据集路径
                 mode="train",  # 模式："train" 或 "test"
                 task_id=None,  # 当前任务 ID
                 num_classes_per_task=None,  # 每个任务的类别数
                 first_task_num_classes=None,  # 第一个任务的类别数，仅在任务划分不均匀时使用
                 generated_data_root=None,  # 生成数据的根路径，仅在 "train" 模式下使用
                 transform=None):
        """
        初始化基础数据集。

        Args:
            data_root (str): 数据集的根路径。
            mode (str): 模式，"train" 或 "test"。
            task_id (int): 当前任务的 ID，从1开始。
            num_classes_per_task (int): 每个任务的类别数。
            generated_data_root (str): 生成数据的根路径，仅在 "train" 模式下使用。
            transform (callable, optional): 图像转换函数。
        """
        assert mode in ["train", "test", "val"], "mode 参数必须是 'train' 或 'test' 或 'val'"
        self.data_root = data_root
        self.mode = mode
        self.task_id = task_id
        self.num_classes_per_task = num_classes_per_task
        self.first_task_num_classes = first_task_num_classes
        self.generated_data_root = generated_data_root
        self.transform = transform

        # 初始化存储
        self.current_image_paths = []
        self.current_labels = []
        self.current_class_to_idx = {}
        self.generated_image_paths = []
        self.generated_labels = []
        self.generated_class_to_idx = {}

        # 加载当前任务的真实数据
        self._load_current_task_real_data()

        # 加载所有之前任务的生成数据（仅在 train 模式下）
        if mode == "train" and generated_data_root is not None:
            self._load_previous_tasks_generated_data()

        # 合并真实数据和生成数据
        self.all_image_paths = self.current_image_paths + self.generated_image_paths
        self.all_labels = self.current_labels + self.generated_labels
        self.all_class_to_idx = {**self.current_class_to_idx, **self.generated_class_to_idx}

        # 更新数据集长度
        self._length = len(self.all_image_paths)

    def _load_current_task_real_data(self):
        """
        加载当前任务的真实数据。
        """
        current_task_dir = os.path.join(self.data_root, f"task_{self.task_id}")
        if self.task_id <= 1:
            task_offset = 0
        elif self.task_id == 2:
            task_offset = self.first_task_num_classes
        elif self.task_id > 2:
            task_offset = self.first_task_num_classes + (self.task_id - 2) * self.num_classes_per_task
        class_names = set()

        for root, dirs, files in os.walk(current_task_dir):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tif')):
                    file_path = os.path.join(root, file)
                    self.current_image_paths.append(file_path)

                    # 假设类别是文件路径的上一级目录名
                    class_label = os.path.basename(os.path.dirname(file_path))
                    class_names.add(class_label)

        # 对类别进行排序并生成映射
        sorted_class_names = sorted(class_names)
        self.current_class_to_idx = {cls_name: idx + task_offset for idx, cls_name in enumerate(sorted_class_names)}

        # 为每个图像分配标签
        for file_path in self.current_image_paths:
            class_label = os.path.basename(os.path.dirname(file_path))
            self.current_labels.append(self.current_class_to_idx[class_label])

    def _load_previous_tasks_generated_data(self):
        """
        加载所有之前任务的生成数据。
        """
        for prev_task_id in range(1, self.task_id):  # 遍历之前所有任务
            prev_task_dir = os.path.join(self.generated_data_root, f"samples_util_task_{self.task_id-1}/task_{prev_task_id}")
            if prev_task_id <= 1:
                task_offset = 0
            elif prev_task_id == 2:
                task_offset = self.first_task_num_classes
            elif prev_task_id > 2:
                task_offset = self.first_task_num_classes + (prev_task_id - 2) * self.num_classes_per_task
            class_names = set()

            for root, dirs, files in os.walk(prev_task_dir):
                for file in files:
                    if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tif')):
                        file_path = os.path.join(root, file)
                        self.generated_image_paths.append(file_path)

                        # 假设类别是文件路径的上一级目录名
                        class_label = os.path.basename(os.path.dirname(file_path))
                        class_names.add(class_label)

            # 对类别进行排序并生成映射
            sorted_class_names = sorted(class_names)
            task_class_to_idx = {cls_name: idx + task_offset for idx, cls_name in enumerate(sorted_class_names)}
            self.generated_class_to_idx.update(task_class_to_idx)

            # 为每个图像分配标签
        for file_path in self.generated_image_paths:
            class_label = os.path.basename(os.path.dirname(file_path))
            self.generated_labels.append(self.generated_class_to_idx[class_label])

    def __len__(self):
        return self._length

    def __getitem__(self, index):
        image_path = self.all_image_paths[index]
        label = self.all_labels[index]
        image = Image.open(image_path).convert("RGB")
        
        # 🔧 修正潜在的 numpy 类型兼容性问题
        image = Image.fromarray(np.array(image))
        if self.transform:
            image = self.transform(image)
        if image.shape != (3, 256, 256):
            print(f"Image {image_path} size: {image.size}")
        return image, label

class TrainDataset(CLBaseDataset):
    def __init__(self, **kwargs):
        super().__init__(mode="train", **kwargs)

class ValidationDataset(CLBaseDataset):
    def __init__(self, **kwargs):
        super().__init__(mode="val", generated_data_root=None, **kwargs)


# ucm_dataset = ValidationDataset(
#     data_root="/data3/zbh/CL/data/ucmerced/val",
#     task_id=1,
#     num_classes_per_task=7,
#     transform=None
# )
# ucm_train_dataset = TrainDataset(
#     data_root="/data3/zbh/CL/data/ucmerced/train",
#     task_id=1,
#     num_classes_per_task=7,
#     generated_data_root="/data3/zbh/CL/ucmerced_generated",
#     transform=None
# )