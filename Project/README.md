## Reproducing CIFAR-10 Results

1. Training the Retain Models: run the script `scripts/train_retain.ipynb` with different
   seeds  
2. Obtaining the logits: run the script `scripts/create_logits.ipynb`, change
   paths if needed.  
   To obtain the logits of the retained models, change MODE to "retain", else to
   "unlearn".  
   To obtain the logits for a specific approach, change the line with
   `unlearn.two_stage_simple`
   to the approach, these can be found in `machine_unlearning/unlearn.py`  
3. Evaluate the Approaches: run the script `scripts/attack.py` with the following
   usage: 
          python3 attack.py <data_path> <approach_name>
   
   The data should be organized in the following way:
   
   data/  
   +-- Approach1/  
&nbsp;&nbsp;&nbsp;&nbsp;+-- logits\_unlearn.npy  
&nbsp;&nbsp;&nbsp;&nbsp;+-- metrics.csv  
   +-- Approach2/  
&nbsp;&nbsp;&nbsp;&nbsp;+-- logits\_unlearn.npy  
&nbsp;&nbsp;&nbsp;&nbsp;+-- metrics.csv  
   .  
   .  
   +-- logits\_retain.npy  
   +-- metrics.csv  

## Reproducing AgeDB Results
1. run `get_results_AgeDB.py` to get results for table 2

## Requirements:
1. [AgeDB.zip](https://polybox.ethz.ch/index.php/s/1h8Y1z9vbLeldzB) and [checkpoints.zip](https://polybox.ethz.ch/index.php/s/1h8Y1z9vbLeldzB) mus be downloaded and extracted in base directory. These files contain the Dataset and the checkpoint for models that have been finetuned on retain set.  
2. [data.zip](https://polybox.ethz.ch/index.php/s/Je6DHi0M1l9S7zk) must be downloaded and extracted in base directory