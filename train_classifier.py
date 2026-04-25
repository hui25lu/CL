import os
import torch
from classifier.model import AlexNet, ResNet
# 主模型训练
def train_main_model(
    model: AlexNet,
    optimizer,
    criterion,
    scheduler,
    train_dataloader,
    val_dataloader,
    n_epochs,
    task_id,
    outputs_dir,
):

    model.device = next(model.parameters()).device

    # 检查是否存在之前的模型文件夹
    start_epoch = 0
    model_path = os.path.join(outputs_dir, f"classifier_model_for_task_{task_id}")
    if os.path.exists(model_path):
        # 获取最后一个epoch的模型文件
        model_files = [
            f for f in os.listdir(model_path) if f.startswith("main_model_epoch_")
        ]
        if model_files:
            model_files.sort(key=lambda x: int(x.split("_")[-1].split(".")[0]))
            last_model_file = model_files[-1]
            checkpoint = torch.load(os.path.join(model_path, last_model_file), map_location=model.device)
            model.load_state_dict(checkpoint["model_state_dict"])
            start_epoch = checkpoint["epoch"] + 1
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            print(f"Resumed training from {last_model_file}")

    for epoch in range(start_epoch, n_epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_dataloader:
            images, labels = images.to(model.device), labels.to(model.device)

            # 前向传播
            outputs = model(images)
            train_loss = criterion(outputs, labels)

            # 反向传播和优化
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

            running_loss += train_loss.item()
        train_losses = running_loss / len(train_dataloader)
        print(
            f"Epoch [{epoch + 1}/{n_epochs}], Loss: {train_losses:.4f}"
        )
        
        # 调整学习率
        if scheduler is not None:
            if val_dataloader is None:
                scheduler.step(train_losses)
                
            else:
            # 使用验证集验证损失的下降
                model.eval()
                best_loss = 100.0
                val_losses = 0.0
                correct = 0
                total = 0
                with torch.no_grad():
                    for images, labels in val_dataloader:
                        images, labels = images.to(model.device), labels.to(model.device)
                        outputs = model(images)
                        _, predicted = torch.max(outputs, 1)
                        total += labels.size(0)
                        correct += (predicted == labels).sum().item()
                        val_loss = criterion(outputs, labels)

                        val_losses += val_loss.item()
                    val_losses /= len(val_dataloader)
                accuracy = 100 * correct / total
                print(f"Validation Loss: {val_losses:.4f}, Accuracy: {accuracy:.2f}%")
                
                scheduler.step(val_losses)


                if val_loss < best_loss:
                    best_loss = val_loss
                    print(f"Best model at epoch {epoch + 1} with val_loss {best_loss:.4f}")
                    best_model_path = os.path.join(
                        outputs_dir, f"classifier_best_models_for_task_{task_id}"
                    )
                    if not os.path.exists(best_model_path):
                        os.makedirs(best_model_path, exist_ok=True)
                    # 保存最优模型
                    torch.save(
                        {
                            "epoch": epoch,
                            "model_state_dict": model.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                            "scheduler_state_dict": scheduler.state_dict(),
                        },
                        os.path.join(best_model_path, f"best_model_{task_id}.pth"),
                    )

        save_model_path = os.path.join(
            outputs_dir, f"classifier_model_for_task_{task_id}"
        )
        if not os.path.exists(save_model_path):
            os.makedirs(save_model_path, exist_ok=True)
        # if (epoch + 1) % 50 == 0 or (epoch + 1) == n_epochs:
        if epoch + 1 == n_epochs:
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                },
                os.path.join(save_model_path, f"main_model_epoch_{epoch + 1}.pth"),
            )