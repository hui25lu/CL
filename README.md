# Class-Incremental Learning for Remote Sensing Scene Classification via Stable Diffusion Based Data Regeneration
## Environment setup

```bash
conda create -n sd-lora python=3.10
conda activate sd-lora
pip install -r requirements.txt
```

## Download Models
We use SD1.4 as the generative model, but you can also substitute it with SD1.5 or others.

## Dataset preparation

You need to organize your dataset in the following format:

```angular2html
├── data1
│  ├── train/             # Training-set images
|  │  ├── task_1/         
|  |  |  |—— class1/
|  │  |  ├── 、、、
|  │  |  ├── metadata.json  # Reorganized JSON file
|  │  ├── task_2/
|  │  ├── 、、、  
│  ├── test/           # the same structure with train folder
│  └── val/           # the same structure with train folder
```
The `metadata.json` file contains two fields:

| Field | Description |
|-------|-------------|
| `file_name` | The corresponding image filename or path |
| `text` | The textual description associated with the image |

## Model inference

```bash
  python test2.py
```

Some important arguments for configurations of inference are:
- `name`: the dataset name.
- `model_name`: the classifier model name, such as "alexnet".[alexnet|resnet34|resnet50|vit|densenet121]
- `model_paths`: the trained classifier model checkpoint path.


## Model training

```bash
  python main2_whu.py
```

Some important parameters are saved in the sd_config/whu-rs_config.yaml file.:
- `name`: Name of the dataset or experiment.
- `model_name`: Backbone architecture used for classification.
- `freeze_partial`: When enabled(default: true), only the weights and biases for newly added classes are updated, while gradients for previously learned classes are frozen.
- `cl_num_tasks`: Total number of tasks
- `cl_num_classes`: Total number of classes across all tasks.
- `classes_first_task`: Number of classes introduced in the first task.
- `cl_output_dir`: Directory where continual learning outputs (e.g., checkpoints, logs, results) are saved.
- `cl_generated_dir`:	Directory where generated samples are stored for replay or augmentation.
- `classifier_main_epochs`:	Number of training epochs for the classifier in each continual learning stage.
- `classifier_batch_size`:	Batch size used when training the classifier.
- `n_samples_per_class`:	Number of generated or replayed samples per class.
- `top_k`:	Number of top candidate generated samples considered during sample selection.
- `filter_mode`:	Strategy for filtering generated samples. [twotail|topk]

## Acknowledgments:
This repo is built upon [DDGR(https://github.com/xiaocangshengGR/DDGR.git)], [diffusers(https://github.com/huggingface/diffusers.git)]. Sincere thanks to their excellent work!
