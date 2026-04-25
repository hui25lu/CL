from diffusers import StableDiffusionPipeline
from typing import Dict, List, Tuple
import os
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torchvision.models import alexnet
from torchvision.utils import save_image
from torchvision import transforms
import math
import torch.nn as nn
from torchvision.models import alexnet as alexnet_model


def extract_features(model: nn.Module, images: torch.Tensor, device: str) -> torch.Tensor:
    """
    自动适配 AlexNet / ResNet34 的特征提取函数。
    提取倒数第二层的全局特征，用于样本多样性筛选。

    Args:
        model: 分类器 (AlexNet 或 ResNet34)
        images: [B, 3, H, W] 输入图像张量
        device: 设备字符串 ('cuda' or 'cpu')

    Returns:
        features: [B, D] 张量，归一化的特征
    """
    model.eval()
    features = []

    # --- Case 1: AlexNet ---
    if isinstance(model, alexnet_model().__class__):
        def hook_fn(_, __, output):
            features.append(output.flatten(1).detach())
        handle = model.classifier[-2].register_forward_hook(hook_fn)
        with torch.no_grad():
            _ = model(images.to(device))
        handle.remove()

    # --- Case 2: ResNet (e.g., resnet18, resnet34, resnet50...) ---
    elif hasattr(model, "avgpool") and hasattr(model, "fc"):
        def hook_fn(_, __, output):
            features.append(torch.flatten(output, 1).detach())
        handle = model.avgpool.register_forward_hook(hook_fn)
        with torch.no_grad():
            _ = model(images.to(device))
        handle.remove()

    else:
        raise ValueError(
            f"Unsupported model type: {model.__class__.__name__}. "
            "Currently supports AlexNet and ResNet-like architectures."
        )

    feats = features[0]
    feats = torch.nn.functional.normalize(feats, dim=1)  # L2 归一化
    return feats

def filter_images_diverse(
    model: torch.nn.Module,
    samples_dict: dict,
    device: str,
    class_to_index: dict,
    output_dir: str,
    batch_size: int = 32,
    top_k: int = 90,
):
    """
    🎨 多样性筛选版本
    基于特征空间的覆盖性采样：
      - 提取每个类别样本的特征；
      - 选择在特征空间中最分散的一组样本。

    Args:
        model: 分类器
        samples_dict: {class_name: [tensor]} 格式的生成样本
        device: 'cuda' 或 'cpu'
        class_to_index: 类别名到索引
        output_dir: 输出目录
        batch_size: 批处理大小
        top_k: 每类最终保留的样本数
    """
    model.to(device)
    model.eval()
    os.makedirs(output_dir, exist_ok=True)

    for target_class, img_list in samples_dict.items():
        all_imgs = img_list[0]  # tensor list
        num_samples = all_imgs.shape[0]
        print(f"\n🎯 类别 '{target_class}' - {num_samples} 样本，提取特征中...")

        # 提取特征
        feats_all = []
        for i in tqdm(range(0, num_samples, batch_size)):
            batch = all_imgs[i:i+batch_size]
            feats = extract_features(model, batch, device)
            feats_all.append(feats.cpu())
        feats_all = torch.cat(feats_all, dim=0)
        feats_all = F.normalize(feats_all, dim=1)  # L2归一化

        # 多样性选择（贪心最大化最小距离）
        selected_indices = []
        remaining_indices = list(range(num_samples))

        # Step 1: 随机选一个初始样本
        first_idx = torch.randint(0, num_samples, (1,)).item()
        selected_indices.append(first_idx)
        remaining_indices.remove(first_idx)

        for _ in tqdm(range(1, min(top_k, num_samples)), desc=f"多样性采样 {target_class}"):
            selected_feats = feats_all[selected_indices]
            rest_feats = feats_all[remaining_indices]
            # 计算每个未选样本到已选样本的最近距离
            dists = 1 - torch.mm(rest_feats, selected_feats.T)  # cosine distance
            min_dists, _ = dists.min(dim=1)
            # 选取距离已选样本最远的那个
            next_idx = remaining_indices[min_dists.argmax().item()]
            selected_indices.append(next_idx)
            remaining_indices.remove(next_idx)

        # 保存结果
        selected_imgs = [all_imgs[i].cpu() for i in selected_indices]
        class_dir = os.path.join(output_dir, target_class)
        os.makedirs(class_dir, exist_ok=True)

        for i, img_tensor in enumerate(selected_imgs):
            image_path = os.path.join(class_dir, f"{target_class}_{i}.png")
            save_image((img_tensor * 0.5) + 0.5, image_path)

        print(f"✅ 类别 '{target_class}'：保留 {len(selected_imgs)} 张最具多样性的样本。")

    print("\n🎉 全部类别多样性筛选完成。")
    
    
def filter_images_twotail(
    model: torch.nn.Module,
    samples_dict: Dict[int, List[torch.Tensor]],
    device: str,
    class_to_index,
    output_dir,
    batch_size: int = 32,
    top_k: int = 90,
    low_tail_ratio: float = 0.5,
    high_tail_ratio: float = 0.5,
):
    """
    两头筛选策略：
    - 保留每个类别置信度分布两端的样本（低端和高端）
    - 在生成图整体质量较好时，有助于保持分布多样性与边界信息。

    Args:
        model: 分类模型。
        samples_dict: {class_name: [tensor]}，每个类别的生成样本。
        device: 'cuda' 或 'cpu'。
        class_to_index: 类别名到索引的映射。
        output_dir: 输出目录。
        batch_size: 批大小。
        top_k: 每类最终保留的样本数。
        low_tail_ratio: 低置信度样本比例（例如0.3表示低端占30%）。
        high_tail_ratio: 高置信度样本比例（例如0.7表示高端占70%）。
    """
    model.to(device)
    model.eval()
    filtered_results = {class_key: [] for class_key in samples_dict.keys()}

    print(f"开始两头筛选，总计 {sum(len(imgs) for imgs in samples_dict.values())} 类别...")

    for target_class, img_list in samples_dict.items():
        all_class_samples = []
        all_probs = []

        print(f"\n处理类别 '{target_class}' ...")
        for i in tqdm(range(0, len(img_list[0]), batch_size), desc=f"计算置信度 {target_class}"):
            batch = img_list[0][i:i+batch_size].to(device)
            with torch.no_grad():
                logits = model(batch)
                probs = F.softmax(logits, dim=1)
                class_probs = probs[:, class_to_index[target_class]]

            for j, prob in enumerate(class_probs):
                all_class_samples.append(batch[j].cpu())
                all_probs.append(prob.item())

        all_probs = torch.tensor(all_probs)
        num_samples = len(all_probs)
        num_select = min(top_k, num_samples)

        # 计算低端和高端各自保留数量
        half_low = int(num_select * low_tail_ratio)
        half_high = num_select - half_low

        # 获取置信度排序索引
        sorted_idx = torch.argsort(all_probs)

        # 低置信度部分
        low_indices = sorted_idx[:half_low]
        # 高置信度部分
        high_indices = sorted_idx[-half_high:]

        selected_indices = torch.cat([low_indices, high_indices], dim=0)

        # 按索引取出图像
        selected_imgs = [all_class_samples[idx] for idx in selected_indices]

        # 保存结果
        class_dir = os.path.join(output_dir, target_class)
        os.makedirs(class_dir, exist_ok=True)
        for i, img_tensor in enumerate(selected_imgs):       
            image_path = os.path.join(class_dir, f"{target_class}_{i}.png")
            save_image((img_tensor * 0.5) + 0.5, image_path)

        print(f"类别 '{target_class}'：保留 {len(selected_imgs)} 张样本 "
              f"(低端 {half_low}，高端 {half_high})")

    return filtered_results

def filter_images_topk(
    model: torch.nn.Module,
    samples_dict: Dict[int, List[torch.Tensor]],
    device: str,
    class_to_index, 
    output_dir,
    confidence_threshold: float = 0.9,
    batch_size: int = 32,
    top_k: int = 90
) -> Dict[int, List[torch.Tensor]]:
    """
    使用场景分类模型筛选生成图像，仅保留分类正确且概率最高的样本。
    如果符合条件的样本不足 top_k，则从原始样本中按照概率最大准则补充。
    """
    # 确保模型在评估模式并转移到指定设备
    model.to(device)
    model.eval()
    
    # 结果字典：保存符合要求的图像
    filtered_results = {class_key: [] for class_key in samples_dict.keys()}
    
    print(f"开始分类筛选{sum(len(imgs) for imgs in samples_dict.values())}张图像...")
    
    # 遍历每个目标类别
    for target_class, img_list in samples_dict.items():
        # 按批次处理当前类别的所有图像
        all_class_samples = []  # 用于存储当前类别的所有符合条件的样本
        all_probs = []  # 用于存储所有样本的概率和索引
        for i in tqdm(range(0, len(img_list[0]), batch_size), 
                     desc=f"筛选'{target_class}'图像"):
            batch = img_list[0][i:i+batch_size].to(device)
            
            # 模型推理
            with torch.no_grad():
                logits = model(batch)
                probs = F.softmax(logits, dim=1)
                
                class_probs = probs[:, class_to_index[target_class]]  # 获取目标类别的概率
                
            
            # 筛选符合条件的图像
            for j, prob in enumerate(class_probs):
                # 保存所有样本的概率和索引
                # all_probs.append((batch[j].cpu(), prob.item(), j))
                # 保存符合条件的图像及其置信度
                all_class_samples.append((batch[j].cpu(), prob.item()))
        
        # 对符合条件的样本按置信度排序，并保留前 top_k 个
        if top_k < len(all_class_samples):
            all_class_samples = sorted(all_class_samples, key=lambda x: x[1], reverse=True)[:top_k]
        
        class_name = target_class
        class_dir = os.path.join(output_dir, class_name)
        os.makedirs(class_dir, exist_ok=True)
        
        for i, img_tensor in enumerate(all_class_samples):
            image_path = os.path.join(class_dir, f"{class_name}_{i}.png")
            save_image((img_tensor[0]*0.5)+0.5, image_path)

def sample_and_save_images_for_each_task(t_id, class_idx, sd_model_path, m_model, main_model_path, sample_dir, n_samples_per_class, top_k, device, filter_mode):
    """为任务i生成样本并保存，保存的结构为
    task i
     ├── class_1
     │   ├── class_1_1.png
     │   ├── class_1_2.png
     │   └── ...
     ├── class_2
     │   ├── class_2_1.png
     │   ├── class_2_2.png
     │   └── ...
     
    Args:
        task_id (int): 任务ID，从1开始。
        class_idx (dict): 类别索引字典，键为类别名称，值为类别ID。
        sd_model_path (str): 当前任务i 微调后的lora参数保存的路径。
        main_model_path (str): 主分类模型的路径。
        sample_dir (str): 保存任务i生成样本的目录。
        n_samples_per_class (int, optional): 每个类别生成的样本数量。默认为100。
        top_k (int, optional): 生成时使用的top-k参数。默认为10。
    """
    pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", torch_dtype=torch.float16)
    pipe.unet.load_attn_procs(sd_model_path)
    pipe.safety_checker = None
    pipe.to("cuda")
    
    
    checkpoint = torch.load(main_model_path,map_location="cpu")
    m_model.load_state_dict(checkpoint["model_state_dict"])
    m_model.to(device)
    
    # 确保保存目录存在
    os.makedirs(sample_dir, exist_ok=True)
    
    # 给出任务中不同类别的prompt
    # 遍历每个类别并生成样本
    for class_name, class_id in class_idx.items():
        
        all_samples = dict([(class_name,[])])
        print(f"rendering {n_samples_per_class} examples of class '{class_name}' in 30 steps.")
        n_batchs = math.ceil(n_samples_per_class / 20)  # 使用 math.ceil 确保批次数为整数
        prompt = [f"A photo of {class_name}"]  # 根据类别生成描述性提示
        x_samples_ddim = torch.empty([n_samples_per_class, 3, 256, 256]).to(device)
        for j in range(int(n_batchs)):
            # 计算当前批次的样本数量
            batch_size = min(20, n_samples_per_class - j * 20)
            images = pipe(prompt*batch_size, num_inference_steps=30, guidance_scale=7.5, height=256, width=256).images
            # 将图像列表转换为张量
            to_tensor = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5)),  # 确保图像在[-1, 1]范围内
            ])
            images_tensor = torch.stack([to_tensor(image).to(device) for image in images])  # 转换为张量并移动到 GPU
            x_samples_ddim[j*20:j*20+batch_size,:,:,:] = images_tensor
        all_samples[class_name].append(x_samples_ddim)
        
        if filter_mode == "topk":
            filter_images_topk(m_model, all_samples, device, confidence_threshold=0.9, batch_size=32,top_k=top_k, class_to_index=class_idx, output_dir=sample_dir)
        elif filter_mode == "twotail":
            filter_images_twotail(m_model, all_samples, device, batch_size=32,top_k=top_k, low_tail_ratio=0.5, high_tail_ratio=0.5, class_to_index=class_idx, output_dir=sample_dir)
        elif filter_mode == "diversity":
            filter_images_diverse(m_model, all_samples, device, batch_size=32,top_k=top_k, class_to_index=class_idx, output_dir=sample_dir)

        print(f"Task {t_id} - Class '{class_name}' samples saved to {class_name}")
    
    
    