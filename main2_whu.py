import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import copy
import argparse
from typing import Any
import json

# Third-Party Library
from PIL import Image
from omegaconf import OmegaConf
import torch
import torchvision
from torchvision.transforms import transforms
from torch.utils.data import DataLoader, Subset, ConcatDataset
from torch import nn, optim
from torch.optim.lr_scheduler import ReduceLROnPlateau


# My Library
from classifier.model import AlexNet, ResNet
from clsddataset import TrainDataset, ValidationDataset
from train_classifier import train_main_model
from train_sd import train_generator
from samples2 import sample_and_save_images_for_each_task


try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url



def parse_args():
    parser = argparse.ArgumentParser(description="CL parameters")
    parser.add_argument("--data_config", type=str, default=None, required=True, help="不同数据集进行持续学习的配置文件")

    return parser.parse_args()
    

    
    
# 定义主模型（分类器）
def MainModel(model_name, pretrained=False, progress=True, num_classes=1000) -> AlexNet:
    """
    MainModel 获取主模型, 其实就是一个分类网络

    Args:
        pretrained (bool, optional): 是否加载预训练参数
        progress (bool, optional): 下载参数时是否显示进度条
        num_classes (int, optional): 分类层最终输出的类别数

    Returns:
        AlexNet | nn.Module: 分类网络, 目前是 AlexNet
    """
    if model_name == "alexnet":
        model: AlexNet = AlexNet()
        if pretrained:
            state_dict = load_state_dict_from_url(model_urls["alexnet"], progress=progress)
            model.load_state_dict(state_dict)
        model.classifier[6] = nn.Linear(4096, num_classes)
    elif model_name == "resnet34":
        model = torchvision.models.resnet34(pretrained=pretrained)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
    elif model_name == "densenet121":
        model = torchvision.models.densenet121(pretrained=pretrained)
        in_features = model.classifier.in_features
        model.classifier = nn.Linear(in_features, num_classes)
    elif model_name == "vit_b_16":
        model = torchvision.models.vit_b_16(pretrained=pretrained)
        in_features = model.heads.head.in_features
        model.heads.head = nn.Linear(in_features, num_classes)
    else:
        raise ValueError(f"Unsupported model name: {model_name}")
    return model


# 动态扩展主模型的输出层
@torch.no_grad()
def expand_model_output(head_var: str, model, new_num_classes: int, freeze_partial: bool):
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
    # 继承旧权重
    new_fc.weight.data[:old_out_features, :] = fc_layer.weight.data
    new_fc.bias.data[:old_out_features] = fc_layer.bias.data

    # ------------------------
    # 4. 冻结继承权重
    # ------------------------
    if freeze_partial:
        def freeze_weight_gradients(grad):
            grad[:old_out_features, :] = 0
            return grad
        def freeze_bias_gradients(grad):
            grad[:old_out_features] = 0
            return grad

        new_fc.weight.register_hook(freeze_weight_gradients)
        new_fc.bias.register_hook(freeze_bias_gradients)

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
# def expand_model_output(head_var: str, model, new_num_classes: int, freeze_partial: bool) -> AlexNet:
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

#     freeze_dim = last_layer[-1].out_features if type(last_layer) == nn.Sequential else last_layer.out_features

#     # fc.weight: [out_features, in_feature]
#     if type(last_layer) == nn.Sequential:
#         new_fc.weight.data[: last_layer[-1].out_features,] = last_layer[-1].weight.data
#         new_fc.bias.data[: last_layer[-1].out_features] = last_layer[-1].bias.data
#         if freeze_partial:
#             # 冻结继承的参数
#             def freeze_weight_gradients(grad):
#                 grad[: freeze_dim, :] = 0  # 冻结前 `out_features` 行
#                 return grad

#             # 冻结偏置的部分梯度
#             def freeze_bias_gradients(grad):
#                 grad[: freeze_dim] = 0  # 冻结前 `out_features` 个元素
#                 return grad

#             # 注册钩子函数
#             new_fc.weight.register_hook(freeze_weight_gradients)
#             new_fc.bias.register_hook(freeze_bias_gradients)

#         last_layer[-1] = new_fc
#         setattr(model, head_var, last_layer)
#     elif type(last_layer) == nn.Linear:
#         new_fc.weight.data[: last_layer.out_features,] = last_layer.weight.data
#         new_fc.bias.data[: last_layer.out_features] = last_layer.bias.data
#         if freeze_partial:
#             # 冻结继承的参数
#             def freeze_weight_gradients(grad):
#                 grad[: freeze_dim, :] = 0  # 冻结前 `out_features` 行
#                 return grad

#             # 冻结偏置的部分梯度
#             def freeze_bias_gradients(grad):
#                 grad[: freeze_dim] = 0  # 冻结前 `out_features` 个元素
#                 return grad

#             # 注册钩子函数
#             new_fc.weight.register_hook(freeze_weight_gradients)
#             new_fc.bias.register_hook(freeze_bias_gradients)

#         last_layer = new_fc
#         setattr(model, head_var, last_layer)
        

#     print(f"Expanded model output layer to {new_num_classes} classes.")
#     return model



if __name__ == "__main__":
    # args = parse_args()
    all_args = OmegaConf.load("sd_configs/whu-rs_config.yaml")
    cl_args = all_args.cl
    sd_args = all_args.sd
    
    num_tasks = cl_args.cl_num_tasks
    num_classes = cl_args.cl_num_classes
    classes_first_task = cl_args.classes_first_task
    num_classes_per_task = (num_classes-classes_first_task) // (num_tasks -1)
    main_epochs = cl_args.classifier_main_epochs
    if cl_args.model_name == "alexnet":
        head_name = "classifier"
    elif cl_args.model_name == "resnet34":
        head_name = "fc"
    elif cl_args.model_name == "densenet121":
        head_name = "classifier"
    elif cl_args.model_name == "vit_b_16":
        head_name = "heads.head"
    
    device = torch.device(f"cuda:0" if torch.cuda.is_available() else "cpu")
    
    model_urls = {
        "alexnet": "https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth",
    }

    output_dir = cl_args.cl_output_dir  # 分类模型和生成模型的保存路径
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    # 将 all_args 转换为字典
    all_args_dict = OmegaConf.to_container(all_args, resolve=True)

    # 保存到 JSON 文件
    with open(f"{output_dir}/all_args.json", "w") as json_file:
        json.dump(all_args_dict, json_file, indent=4)
    print(f"all_args 已成功保存到{output_dir}/all_args.json文件中。")



        
    # 数据加载和划分
    train_transform = transforms.Compose(
        [transforms.Resize((256, 256)), transforms.ToTensor(), transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))]
    )

    test_transform = transforms.Compose(
        [transforms.Resize((256, 256)), transforms.ToTensor(), transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))]
    )

    # 初始化主模型和生成器
    init_main_model = MainModel(
        model_name=cl_args.model_name, pretrained=True, progress=True, num_classes=classes_first_task
    ).to(device)

    criterion = nn.CrossEntropyLoss()

    # 持续学习循环
    init_num_classes = classes_first_task  # 初始任务主模型的类别数
    
    # =========== 持续学习 ===========
    # for task_id in range(1, num_tasks + 1):
    for task_id in range(1, 3):
        print(f"Training on Task {task_id}/{num_tasks}")

        # =========== 准备学习新任务 ===========
        current_num_classes = init_num_classes
        if task_id >= 2:
            # 更新当前任务的类别数
            current_num_classes = classes_first_task + (task_id-1) * num_classes_per_task
            pre_main_model = expand_model_output(
                head_name, init_main_model, current_num_classes - num_classes_per_task, False
            )
            # pre_checkpoint = torch.load(
            #     os.path.join(
            #         output_dir,
            #         f"classifier_model_for_task_{task_id-1}/main_model_epoch_{main_epochs}.pth",
            #     ),
            #     map_location=device,
            # )
            # pre_main_model.load_state_dict(pre_checkpoint["model_state_dict"])
            current_main_model = expand_model_output(
                head_name, pre_main_model, current_num_classes, cl_args.freeze_partial
            ).to(device)
            train_dataset = TrainDataset(
                data_root=f"sd_data/{cl_args.name}/train",
                task_id=task_id,
                num_classes_per_task=num_classes_per_task,
                first_task_num_classes=classes_first_task,
                generated_data_root=cl_args.cl_generated_dir,
                transform=train_transform,
            )
        else:
            current_main_model = init_main_model.to(device)
            train_dataset = TrainDataset(
                data_root=f"sd_data/{cl_args.name}/train",
                task_id=task_id,
                num_classes_per_task=num_classes_per_task,
                first_task_num_classes=classes_first_task,
                generated_data_root=None,   
                transform=train_transform,
            )

        # val_dataset = ValidationDataset(
        #     data_root=f"sd_data/{cl_args.name}/val",
        #     task_id=task_id,
        #     num_classes_per_task=num_classes_per_task,
        #     transform=test_transform,
        # )
        val_dataset = None

        all_train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True,num_workers=4)
        if val_dataset is not None:
            all_val_dataloader = DataLoader(val_dataset, batch_size=4, shuffle=True, num_workers=4)
        else:
            all_val_dataloader = None
            
        # =========== 学习新任务 ===========

        main_optimizer = optim.Adam(filter(lambda p: p.requires_grad, current_main_model.parameters()), lr=0.00001)

        # 初始化学习率调度器
        scheduler = ReduceLROnPlateau(main_optimizer, mode="min", factor=0.7, patience=5)

        # 1. 训练分类模型
        train_main_model(
            current_main_model,
            main_optimizer,
            criterion,
            scheduler,
            all_train_dataloader,
            all_val_dataloader,
            main_epochs,
            task_id,
            output_dir,
        )

        # # 2. 训练生成模型
        # # 训练生成器，结合旧任务生成的数据
        # # 重新初始化生成器
        # if task_id < num_tasks:
        #     sd_args.resume_from_checkpoint = None
        #     sd_args.output_dir = os.path.join(output_dir, f"sd-{cl_args.name}-model-lora_for_task_{task_id}")

        #     if os.path.exists(os.path.join(sd_args.output_dir)):
        #         sd_args.resume_from_checkpoint = "latest"
                
        #     sd_args.train_data_dir = f"sd_data/{cl_args.name}/train/task_{task_id}"
        #     train_generator(sd_config=sd_args)

        #     # 采样并保存生成的数据
        #     # 每个类别生成 100 个样本
            
        #     for i in range(1, task_id+1):
        #         if not os.path.exists(os.path.join(cl_args.cl_generated_dir,f"samples_util_task_{task_id}/task_{i}")):
        #             print(f"开始为任务 {i}/{task_id} 生成样本...")
        #             # Get the most recent checkpoint
        #             sd_resume_checkpoint = 'latest'
        #             model_dirs = os.path.join(output_dir, f"sd-{cl_args.name}-model-lora_for_task_{i}")
        #             model_dirs = [d for d in os.listdir(model_dirs) if d.startswith("checkpoint")]
        #             model_dirs = sorted(model_dirs, key=lambda x: int(x.split("-")[1]))
        #             model_path = model_dirs[-1] if len(model_dirs) > 0 else None
                    
        #             # 如果没有找到模型路径，则报错，给出错误信息：没有找到模型路径
        #             if model_path is None:
        #                 raise ValueError(f"没有找到任务 {i} 的模型路径，请检查任务 {i} 是否已完成训练。")
        #             else:
        #                 print(f"任务 {i} 的sd lora模型路径为：{model_path}")

                    
        #             task_class_indices = {
        #                 class_name: idx
        #                 for class_name, idx in train_dataset.all_class_to_idx.items()
        #                 if class_name in os.listdir(f"sd_data/{cl_args.name}/train/task_{i}")
        #             }
                    
        #             sample_and_save_images_for_each_task(
        #                 t_id=i,
        #                 class_idx=task_class_indices,
        #                 sd_model_path=os.path.join(os.path.join(output_dir, f"sd-{cl_args.name}-model-lora_for_task_{i}"),model_path),
        #                 m_model = current_main_model,
        #                 main_model_path=os.path.join(
        #                                     cl_args.cl_output_dir,
        #                                     f"classifier_model_for_task_{task_id}/main_model_epoch_{main_epochs}.pth"),
        #                 sample_dir=os.path.join(cl_args.cl_generated_dir,f"samples_util_task_{task_id}/task_{i}"),
        #                 n_samples_per_class=cl_args.n_samples_per_class,
        #                 top_k=cl_args.top_k,
        #                 device=device,
        #                 filter_mode=cl_args.filter_mode
        #             )

