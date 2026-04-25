# 持续学习不同模型在不同任务测试集上进行测试
"""循环读取第i个任务的测试数据，然后加载对应的分类器模型进行测试，获取每个分类器在每个数据集上的分类精度、F1等指标。
1. 写一个单个任务在单个数据集上的测试函数；
2. 循环获取第i个任务训练好的分类器模型；
3. 获取前i个数据集的测试数据集（j<i）；
4. 循环使用第i个任务训练好的分类器模型在第j个测试数据集上进行测试，并获取指标。

评价指标的表格：
- acc:
  task     1      2      3      4       5
  model1     xxx
  model2     xxx    xxx
  model3     xxx    xxx    xxx
  model4     xxx    xxx    xxx    xxx
  model5     xxx    xxx    xxx    xxx     xxx

- confusion matrices for each class in each task
- precision、recall、F1 for each class in each task:方便观察每个训练好的模型在各个类别上的表现变化

precision/recall/F1:
                      task1(下面的class1, ..., classn是任务1中的类别)
            class1   class2  ...  classn
model1       xxx      xxx    ...   xxx
model2       xxx      xxx    ...   xxx
model3       xxx      xxx    ...   xxx
model4       xxx      xxx    ...   xxx

                      task2(下面的class1, ..., classn是任务2中的类别)
            class1   class2  ...  classn
model1        /        /     ...    /      
model2       xxx      xxx    ...   xxx
model3       xxx      xxx    ...   xxx
model4       xxx      xxx    ...   xxx

                      task3(下面的class1, ..., classn是任务3中的类别)
            class1   class2  ...  classn
model1        /        /     ...    /      
model2        /        /     ...    /   
model3       xxx      xxx    ...   xxx
model4       xxx      xxx    ...   xxx  
"""
import os
import pandas as pd
import torch
import torchvision
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageFolder
from torchvision import transforms
# from sklearn.metrics import f1_score
from classifier.model import AlexNet
from clsddataset import ValidationDataset

@torch.no_grad()
def expand_model_output(head_var: str, model, new_num_classes: int):
    """
    扩展分类器输出层，兼容 AlexNet / ResNet / DenseNet / ViT
    保持原 expand_model_output 的接口
    
    Args:
        head_var (str): 分类头属性名
        model (nn.Module): 分类网络
        new_num_classes (int): 新类别数
        freeze_partial (bool): 是否冻结继承权重
    
    Returns:
        nn.Module: 扩展后的模型
    """
    # ------------------------
    # 1. 获取分类头
    # ------------------------
    if head_var == "heads.head" and hasattr(model, "heads") and hasattr(model.heads, "head"):
        last_layer = model.heads.head
        set_layer_func = lambda layer: setattr(model.heads, "head", layer)
    else:
        last_layer = getattr(model, head_var)
        set_layer_func = lambda layer: setattr(model, head_var, layer)
    
    # ------------------------
    # 2. 判断分类头类型
    # ------------------------
    if isinstance(last_layer, nn.Sequential):
        fc_layer = last_layer[-1]
    elif isinstance(last_layer, nn.Linear):
        fc_layer = last_layer
    else:
        raise TypeError(f"Unsupported layer type {type(last_layer)} for expansion")

    in_features = fc_layer.in_features
    old_out_features = fc_layer.out_features

    # ------------------------
    # 3. 新分类头
    # ------------------------
    new_fc = nn.Linear(in_features, new_num_classes)

    # ------------------------
    # 5. 替换原分类头
    # ------------------------
    if isinstance(last_layer, nn.Sequential):
        last_layer[-1] = new_fc
        set_layer_func(last_layer)
    else:
        set_layer_func(new_fc)

    print(f"Expanded model output layer to {new_num_classes} classes for model type {type(model).__name__}")
    return model

# # 动态扩展主模型的输出层
# @torch.no_grad()
# def expand_model_output(head_var: str, model, new_num_classes: int) -> AlexNet:
#     """
#     expand_model_output 在持续学习学习新任务前扩展分类器的分类头

#     Args:
#         head_var (str): 分类头变量名, AlexNet 是 'classifier', ResNet 是 'fc'
#         model (AlexNet | nn.Module): 分类网络
#         new_num_classes (int): 新的类别

#     Returns:
#         AlexNet | nn.Module: 扩展分类头后的网络
#     """
#     assert type(head_var) == str
#     assert hasattr(model, head_var), "Given model does not have a variable called {}".format(head_var)
#     assert type(getattr(model, head_var)) in [nn.Sequential, nn.Linear], \
#         "Given model's head {} does is not an instance of nn.Sequential or nn.Linear".format(head_var)
        
#     last_layer = getattr(model, head_var)
#     if type(getattr(model, head_var)) == nn.Sequential:
#         out_size = last_layer[-1].in_features
#     elif type(last_layer) == nn.Linear:
#         out_size = last_layer.in_features

#     new_fc = nn.Linear(out_size, new_num_classes)

#     # fc.weight: [out_features, in_feature]
#     if type(last_layer) == nn.Sequential:
#         last_layer[-1] = new_fc
#         setattr(model, head_var, last_layer)
#     elif type(last_layer) == nn.Linear:
#         last_layer = new_fc
#         setattr(model, head_var, last_layer)
        

#     print(f"Expanded model output layer to {new_num_classes} classes.")
#     return model

@torch.no_grad()
# 写一个单个任务在单个数据集上的测试函数
def test_single_task_on_dataset(task_id, k, num_class_per_task, dataloader, model, first_task_classes, device):
    """
    在单个任务的单个数据集上进行测试。

    Args:
        task_id (int): 任务ID。
        dataset (Dataset): 测试数据集。
        model (nn.Module): 训练好的分类器模型。
        device (torch.device): 设备（CPU或GPU）。
    
    Returns:
        dict: 包含测试结果的字典，包括准确率、F1分数等指标。
    """
    model.eval()
    taw_correct = 0
    tag_correct = 0
    total = 0
    all_labels = []
    all_tag_preds = []
    all_taw_preds = []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, tag_predicted = torch.max(outputs.data, 1)
            if k <= 1:
                _, taw_predicted = torch.max(outputs.data[:, 0:first_task_classes], 1)
            elif k >= 2:
                _, taw_predicted = torch.max(outputs.data[:,first_task_classes + (k-2) * num_class_per_task : first_task_classes + (k-1) * num_class_per_task], 1)
                taw_predicted += first_task_classes + (k-2) * num_class_per_task
            total += labels.size(0)
            tag_correct += (tag_predicted == labels).sum().item()
            taw_correct += (taw_predicted == labels).sum().item()
            all_labels.append(labels.cpu().numpy())
            all_tag_preds.append(tag_predicted.cpu().numpy())
            all_taw_preds.append(taw_predicted.cpu().numpy())

    taw_accuracy = taw_correct / total
    tag_accuracy = tag_correct / total
    # f1_score = f1_score(all_labels, all_preds, average='weighted')
    
    print(f"Model {task_id} on dataset {k} - Taw Accuracy: {taw_accuracy:.4f} - Tag Accuracy: {tag_accuracy:.4f}.")


    return taw_accuracy, tag_accuracy

@torch.no_grad()
def test_all_tasks_on_all_datasets(name, model_name, path, device):
    """
    在所有任务的所有数据集上进行测试。

    Args:
        name (str): 数据集名称。
        num_tasks (int): 任务数量。
        device (torch.device): 设备（CPU或GPU）。
    
    Returns:
        list: 包含每个任务在每个数据集上的测试结果的列表。
    """
    if name == "ucmerced":
        num_tasks = 3
        classes_first_task = 7
        num_classes_per_task = 7
    elif name == "aid":
        num_tasks = 6
        classes_first_task = 5
        num_classes_per_task = 5
    elif name == "nwpu":
        num_tasks = 5
        classes_first_task = 9
        num_classes_per_task = 9
    elif name == "clrs":
        num_tasks = 5
        classes_first_task = 5
        num_classes_per_task = 5
    elif name == "whu-rs":
        num_tasks = 4
        classes_first_task = 4
        num_classes_per_task = 5
    
    # 初始化结果字典
    taw_results_dict = {}
    tag_results_dict = {}
        
    
    for task_id in range(1,(num_tasks+1)):
        # 获取当前任务的模型路径
        model_path = f"{path}/classifier_model_for_task_{task_id}/main_model_epoch_50.pth"

        if model_name == 'alexnet':
            model = AlexNet()
            if name == "whu-rs":
                current_num_classes = classes_first_task + (task_id - 1) * num_classes_per_task
                model = expand_model_output('classifier', model, current_num_classes)
            else:
                current_num_classes = num_classes_per_task * task_id
                model = expand_model_output('classifier', model, current_num_classes)
        elif model_name == 'resnet34':
            model = torchvision.models.resnet34(pretrained=False)
            if name == "whu-rs":
                current_num_classes = classes_first_task + (task_id - 1) * num_classes_per_task
                model = expand_model_output('fc', model, current_num_classes)
            else:
                current_num_classes = num_classes_per_task * task_id
                model = expand_model_output('fc', model, current_num_classes)
        elif model_name == 'resnet50':
            model = torchvision.models.resnet50(pretrained=False)
            if name == "whu-rs":
                current_num_classes = classes_first_task + (task_id - 1) * num_classes_per_task
                model = expand_model_output('fc', model, current_num_classes)
            else:
                current_num_classes = num_classes_per_task * task_id
                model = expand_model_output('fc', model, current_num_classes)
        elif model_name == 'densenet121':
            model = torchvision.models.densenet121(pretrained=False)
            if name == "whu-rs":
                current_num_classes = classes_first_task + (task_id - 1) * num_classes_per_task
                model = expand_model_output('classifier', model, current_num_classes)
            else:
                current_num_classes = num_classes_per_task * task_id
                model = expand_model_output('classifier', model, current_num_classes)
        elif model_name == 'vit':
            model = torchvision.models.vit_b_16(pretrained=False)
            if name == "whu-rs":
                current_num_classes = classes_first_task + (task_id - 1) * num_classes_per_task
                model = expand_model_output('heads.head', model, current_num_classes)
            else:
                current_num_classes = num_classes_per_task * task_id
                model = expand_model_output('heads.head', model, current_num_classes)
        else:
            raise ValueError("Unsupported model type")
        model_ckpt = torch.load(model_path, map_location="cpu")
        model.load_state_dict(model_ckpt['model_state_dict'])
        model.to(device)

        # 初始化当前模型的结果字典
        taw_results_dict[f"model{task_id}"] = {}
        tag_results_dict[f"model{task_id}"] = {}
    
        # 获取所有见过的测试数据集
        for j in range(1, task_id + 1):
            # 读取文件夹中的数据集
            # 第j个数据集的测试数据
            test_transform = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,))
            ])
            test_dataset = ValidationDataset(
            data_root=f"sd_data/{name}/test",
            task_id=j,
            num_classes_per_task=num_classes_per_task,
            first_task_num_classes=classes_first_task,
            transform=test_transform,
            )

            dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)

            taw_acc, tag_acc = test_single_task_on_dataset(task_id, j, num_classes_per_task, dataloader, model, classes_first_task, device)
            tag_results_dict[f"model{task_id}"][f"task{j}"] = tag_acc
            taw_results_dict[f"model{task_id}"][f"task{j}"] = taw_acc

    return taw_results_dict, tag_results_dict


def print_results_dict(results_dict):
    """
    打印测试结果字典，格式化为表格形式。

    Args:
        results_dict (dict): 测试结果的字典。
    """
    # 将结果字典转换为 DataFrame
    df = pd.DataFrame(results_dict).T  # 转置以便任务作为列，模型作为行
    print("\n测试结果:")
    print(df.to_string(float_format="{:.4f}".format))  # 格式化为 4 位小数


def save_results_to_csv(results_dict, filename):
    """
    将测试结果字典保存到 CSV 文件，并格式化输出。

    Args:
        results_dict (dict): 测试结果的字典。
        filename (str): 保存的 CSV 文件名。
    """
    # 将结果字典转换为 DataFrame
    df = pd.DataFrame(results_dict).T  # 转置以便任务作为列，模型作为行

    # 填充缺失值为 NaN
    df = df.fillna("")

    # 保存到 CSV 文件
    df.to_csv(filename, index=True, float_format="%.4f", sep="\t")
    print(f"\n结果已保存到 {filename}")

        
if __name__ == "__main__":
    # 示例运行
    device = torch.device("cuda:7" if torch.cuda.is_available() else "cpu")
    name = "whu-rs"  # 数据集名称
    model_name = "alexnet"  # 模型名称
    model_paths = f"{name}_output"  # 模型存储路径

    taw_results_table, tag_results_table = test_all_tasks_on_all_datasets(name, model_name, model_paths, device)
    print_results_dict(taw_results_table)
    print_results_dict(tag_results_table)
    # 将结果保存到csv文件

    tag_save_results_path = f"{model_paths}/tag_results.csv"
    taw_save_results_path = f"{model_paths}/taw_results.csv"
    save_results_to_csv(tag_results_table, tag_save_results_path)
    save_results_to_csv(taw_results_table, taw_save_results_path)
