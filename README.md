# FYP on Histology Image Segmentation with Deep Learning

This repository contains code and experiments for semantic segmentation of histology images using deep learning models on the [BreCaHAD dataset](https://figshare.com/articles/dataset/BreCaHAD_A_Dataset_for_Breast_Cancer_Histopathological_Annotation_and_Diagnosis/7379186?file=13646363).

## Project Structure

### Main Files

- `deeplab.py` / `deeplab_exp_runner.py`: Implementation and training script for DeepLabV3.
- `segformer.py` / `experiment_runner.py`: Implementation and training script for the SegFormer models using Hugging Face Transformers.
- `config_generator.py`: Utility to create experiment configurations for the DeepLabV3 and SegFormer models.
- `run_MedSAM.py` / `medsam_configs.py` / `medsam_exp_runner.py`: Implementation, configuration and training scripts for MedSAM.

### Folders

- `Results_files/`: Contains CSV files with segmentation results from various model configurations.
- `MedSAM/`: Cloned repository of [MedSAM](https://github.com/bowang-lab/MedSAM), a promptable medical image segmentation model.

## Dataset

All models were trained and evaluated on the [BreCaHAD Dataset](https://figshare.com/articles/dataset/BreCaHAD_A_Dataset_for_Breast_Cancer_Histopathological_Annotation_and_Diagnosis/7379186?file=13646363), which contains breast cancer histology images with expert annotations.

---

This project was built using PyTorch and Hugging Face libraries and includes ablation studies and architectural modifications to assess model performance.
