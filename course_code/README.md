# Hardware

Our code was designed to run on a Google Cloud VM with at least 4 vCPUs, 15 GB RAM, and 1x NVIDIA T4 GPU (or some equivalent with 16+ GB of VRAM).

# Pipeline Setup + Execution

The end-to-end process of installing dependencies and running the pipeline is found in the `colab_vm.ipynb` notebook. This is the file that gets run on the Google Cloud VM. There are denoted sections in this notebook for running the baseline vanilla + RAG models, and for running our own model (dubbed "RetRobust++" here).

The process of fine-tuning the notebook using Unsloth is covered in the `finetune.ipynb` notebook. You can use a regular Colab session with a GPU runtime for this, as it only takes 40 minutes or so. It can be run in one go, with no need for a persistent filesystem. It assumes you have the CSV files located in the correct paths in your Google Drive account, and the CSVs in question are provided in this folder under `data/`.

The only notable omissions from this notebook are the scripts used to synthesize and prepare fine-tuning data. The scripts are still included as separate Python files, namely `azure_gpt.py`, `cleaning.py`, and `prompt.py`. The exact fine-tuning data that these scripts produced is located under `data/`.

There may also be some unused Python scripts, such as `data_collator.py`, and logic borrowed from the `reasoning-on-cots` repo (https://github.com/oriyor/reasoning-on-cots) from the authors of the RetRobust paper. This is because we attempted to reproduce the RetRobust methods exactly at first, but this code ended up being largely unworkable for us.

# Baseline Comparisons 

Vanilla baseline 1B: `../output/data/vanilla_baseline/Llama-3.2-1B-Instruct`

RAG baseline 1B: `../output/data/rag_baseline/Llama-3.2-1B-Instruct`

Vanilla baseline 3B: `../output/data/rag_baseline/Llama-3.2-3B-Instruct-bnb-4bit`

RAG baseline 3B: `../output/data/rag_baseline/Llama-3.2-3B-Instruct-bnb-4bit`

Our RAG model with only our synthesized training data (1B): `../output/data/retrobust_plusplus/retrobust_plusplus_f16`

Our RAG model with only our synthesized training data (3B): `../output/data/retrobust_plusplus/retrobust_plusplus_3b_f16`

Our final RAG model, with our data + NQ + WikiHop + StrategyQA (3B): `../output/data/retrobust_plusplus/retrobust_plusplus_combined_3b_f16`

Comparison of custom models against baselines: 

![image](https://github.com/user-attachments/assets/dd4b8452-fbaa-4f77-9e29-55478660b179)
