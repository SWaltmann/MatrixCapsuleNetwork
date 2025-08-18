# Matrix Capsule Networks with EM Routing  

Welcome to this repository! 🎉  
This project is a modular implementation of Hinton’s **Matrix Capsule Networks with EM Routing**, designed for clarity and easy experimentation.

---

##  Requirements  

The repository uses **Python** with common ML dependencies (TensorFlow, NumPy, etc.). It has been tested on Python 3.12 with TensorFlow 2.16 and TensorFlow 2.18.
A complete `requirements.txt` will be added soon.  

---

## Running Training  

The main entry point is `run_training.py`.  
- Define the experiment you want to run, and execute the script. Training of the experiments in `experiments/old_setup/` cannot be resumed. Those models can only be tested.  
- Behavior depends on the experiment folder:  
  - If only a `config.json` exists → a new model will be trained. Make sure to set `use_pretrained` to false when copying the config of an old run.  
  - If saved checkpoints are present → training will resume.  

### Starting a new experiment  
1. Create a new subdirectory in `experiments/`, e.g.:  
   ```
   experiments/EXAMPLE/
   ```  
2. Copy an existing `config.json` into it (file **must** be named `config.json`).  
3. Adjust the config to your needs (see `config_guide.md`).  
4. Update `run_training.py` to point `MatrixCapsuleNetwork` to your new experiment path (`experiments/EXAMPLE`).  
5. Run the script:  
   ```bash
   python run_training.py
   ```  

Tip: Keep the number of epochs low during single runs, since only the last model is stored. To train continuously, wrap training in a loop. On my RTX4060, the full model takes 3 minutes/epoch to train. So I tend to set the number of epochs to 20 so it is saved every hour.

---

## Repository Structure  

### 1. `experiments/`  
This directory contains the configuration and results for each experiment (or run).  
- Each experiment lives in its own subdirectory (e.g., `original_model/`).  
- At minimum, an experiment must include a `config.json` file, which defines the training setup (model architecture, loss function, number of iterations, validation set, etc.).  
- A detailed guide for available config options is provided in [`config_guide.md`](experiments/config_guide.md).  
- During training, the following files are saved automatically inside the experiment folder:  
  - **`best_model`** – the best-performing checkpoint  
  - **`last_model`** – the most recent checkpoint  
  - **`training_history`** – logs of the training progress  

---

### 2. `models/`  
This directory manages the model logic.  
- `model.py` (or equivalent) handles the overhead: loading/saving checkpoints, selecting the correct model based on the config, and resuming training if applicable.  
- `matrix_capsule_networks.py` defines the actual network architecture (layer order and dimensions).  
- In short: the config file is translated into a working model here.  

---

### 3. `utils/`  
This directory provides building blocks and helpers.  
- **`layers_em_hinton.py`**: Each capsule layer is defined as its own class for readability and modularity.  
  - Required layer order:  
    1. `ReLUConv`  
    2. `PrimaryCaps`  
    3. `ConvCaps + EMRouting` (can be repeated until the spatial size becomes too small)  
    4. `ClassCaps + EMRouting`  
- **`dataset.py`**: Handles dataset loading, preprocessing, and augmentation, while also reading parameters from `config.json`.  
- **`loss_functions.py`**: Contains the `SpreadLoss` implementation.  
  - Note: `SpreadLoss` requires access to the optimizer. To support training restarts, the optimizer is injected back into this class by the model wrapper.  


  
