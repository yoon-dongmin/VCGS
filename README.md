# VCGS

[MODEL] Constrained Generative Sampling of 6-DoF Grasps

- Original Paper: [https://arxiv.org/abs/2302.10745](https://arxiv.org/abs/2302.10745)
- Original Repository: [https://github.com/jsll/position-contrained-6dof-graspnet](https://github.com/jsll/position-contrained-6dof-graspnet)

---

## 0. Setting

```
Ubuntu 20.04
CUDA 11.2.2
Python 3.8.18
Pytorch 1.10.0
```
    
1. Make Conda virtual env `conda create -n vcgs python==3.8.18`
2. Clone the repository & Install packages using requirements.txt `conda activate vcgs && pip install -r requirements.txt`

3. Download the dataset
    - There are two datasets for each GENERATOR(SAMPLER) and EVALUATOR
    - For GENRERATOR(`dataset` Folder)
      - Full dataset is available in [Original Repository](https://huggingface.co/datasets/jens-lundell/cong/resolve/main/full_dataset.zip) or [/cmes_ai_team_share/avery_shared/VCGS/dataset/full_dataset](http://gofile.me/6VqgB/bq8yVAaU1)
      - Generated dataset can get from [CONG repository](https://github.com/CMES-AI/CONG) Running or is available in [/cmes_ai_team_share/avery_shared/VCGS/dataset/generated_dataset](http://gofile.me/6VqgB/yupA5edIX)
    - For EVALUATOR(`data` Folder)
      - `shapenetsem_full_obj` is available in [/cmes_ai_team_share/avery_shared/VCGS/data/shapenetsem_full_obj](http://gofile.me/6VqgB/2Qy47tAGe)

4. Download pretrained model
   - go to [https://drive.google.com/uc?export=download&id=1B0EeVlHbYBki__WszkbY8A3Za941K8QI](https://drive.google.com/uc?export=download&id=1B0EeVlHbYBki__WszkbY8A3Za941K8QI)
   - unzip `model.zip` file at `checkpoints/models`

5. **Pytorch version** is also downgraded along with CUDA version.
    - `pip install torch==1.10.0+cu111 torchvision==0.11.0+cu111 torchaudio==0.10.0 -f https://download.pytorch.org/whl/torch_stable.html`

6. install PointNet2/PointNet++ dependencies
   - Clone pointnet++: https://github.com/erikwijmans/Pointnet2_PyTorch
   - Run `cd Pointnet2_PyTorch && pip3 install -r requirements.txt`

<details>
<summary><b>🧐 Installation Backup Data</b></summary>

> This is the history of installation error and solution. It is already fixed in the current installation process.
- The original installation manual in [https://github.com/jsll/pytorch_6dof-graspnet?tab=readme-ov-file#installation](https://github.com/jsll/pytorch_6dof-graspnet?tab=readme-ov-file#installation) 
  - **CUDA version**: `11.8` [Download Link](https://developer.nvidia.com/cuda-11-8-0-download-archive?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=22.04&target_type=runfile_local) (Because of **PointNet2/PointNet++**)
  - Change the *symbolink of CUDA*
    ```
    # Installation with runfile
    $sudo sh cuda-xx.x.run --toolkit --toolkitpath=/usr/local/cuda-xx.x --silent --toolkit 
        
    # Change Symbolink
    $ cd /usr/local 
    $ sudo rm cuda 
    $ sudo ln -s cuda-xx.x cuda 
        
    # Setting in .bashrc file
    export PATH=/usr/local/cuda/bin${PATH:+:${PATH}}
    export LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
    
    source ~/.bashrc
    
    ```
  
  - **Numpy version** should be `1.23.1` (written in `requirements.txt`)
    - Because of the `h5py` and rendering module dependency
  - **Wandb version** should be `0.15.11` (written in `requirements.txt`)

</details>

---

## 1. Dataset

- Refer to **[0. Setting - 3. Download the dataset]**

```
[full_dataset]
Total dataset size:  7896
------------------------------
Training set: 6317 samples
Validation set: 1184 samples
Test set: 395 samples

[genreated_dataset]
Total dataset size:  2309
------------------------------
Training set: 1847 samples
Validation set: 346 samples
Test set: 116 samples
```

## 1.1 For Generator

- Place the `full_dataset` folder under `dataset` folder
    ```
    root
    └── dataset
        ├── download_dataset.sh
        └── full_dataset
            ├── constrained_..._011099225885734912.pickle
            ├── constrained_..._011012345657343452.pickle
            ├── ...
    ```
  
- Place the `generated_dataset` folder under `dataset` folder
    ```
    root
    └── dataset
        ├── download_dataset.sh
        └── generated_dataset
            └── grasp
                 ├── constrained_..._005613932903906825.pickle
                 ├── constrained_..._0037487236921236685.pickle
                 ├── ...
    ```
  
> Check the dataset
- `python vis_full_dataset.py`
- `python vis_generated_dataset.py`
  
## 1.2 For Evaluator

- Place the `shapenetsem_full_obj` folder under `data` folder
    ```
    root
    └── data
        ├── __init__.py
        └── shapenetsem_full_obj
            ├── 1a0c94a2e3e67e4a2e4877b52b3fca7.obj
            ├── 1a2a5a06ce083786581bb5a25b17bed6.obj
            ├── ...
    ```
  

## 2. Running Code 
-  Before Running the codes, you should set Wandb User Name in `train.py` file
    ```python     
    if __name__ == '__main__':
        
        WANDB = True
        if WANDB:
            PROJECT_NAME = "VCGS"
            USER_NAME = input() ### 👈 Set your wandb user name
            
        main()
        
        if WANDB:
            wandb.finish()
    ```

### Train
- Sampler
  - `python train.py --arch sampler --gpu_ids 0,1`
  - Default training dataset is `generated_dataset`. If you want to use `full_dataset`, use `-d ./dataset/full_dataset`
- Evaluator
  - `python train.py --arch evaluator --gpu_ids 0`
  - Do not use multiple GPUs for evaluator
- `--save_epoch_freq` : save model more frequently

### Test
- Sampler
  - `python test.py --arch sampler`
- Evaluator
  - `python test.py --arch evaluator`
  - Do not use multiple GPUs for evaluator

### Debugging

- When debugging the code, set number of threads(`num_workers` in `data/__init__.py`) to `0`.

    ```python
    def create_dataloader(self, dataset, shuffle_batches):
        self.dataloader = torch.utils.data.DataLoader(dataset,
                                                      batch_size=self.opt.num_objects_per_batch,
                                                      shuffle=shuffle_batches,
                                                      num_workers= int(self.opt.num_threads), ### 👈 SET 0
                                                      collate_fn=collate_fn)
    ```
	
### Training Log in WandB

- https://wandb.ai/curieuxjy/VCGS?nw=nwusercurieuxjy