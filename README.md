# Machine Unlearning: Assessing Unlearning Approaches

This repository contains the code and resources for the project **"Forget with Precision: Assessing Machine Unlearning Approaches"**. The project explores various machine unlearning techniques, including finetuning, label poisoning, pruning, and re-initialization, to induce a model to forget a portion of the dataset it was trained on. The methods are evaluated on the **ResNet-18** architecture trained on **CIFAR-10** and **AgeDB** datasets.

---

## Key Contributions
1. **Machine Unlearning Framework**: A comprehensive framework for understanding and evaluating machine unlearning in deep learning models.
2. **Novel Unlearning Techniques**: Introduction of two new methods—**Pruning Complex** and **Activation Reset**—for efficient data removal.
3. **Experimental Analysis**: Evaluation of unlearning methods on CIFAR-10 and AgeDB datasets, focusing on computational efficiency, forget quality, and model accuracy.
4. **Custom Metric (\(\hat{\epsilon}\))**: A novel metric inspired by the Likelihood-Ratio attack to assess the effectiveness of unlearning.

---

## Results
- **Pruning Complex** demonstrated the best forget quality under the \(\hat{\epsilon}\) metric.
- All methods maintained high accuracy on the retain set, with some variations in forget set performance.
- The results highlight that low accuracy on the forget set does not necessarily imply effective unlearning.

---

## Reproducing Results

### CIFAR-10 Results
1. **Training the Retain Models**:
   - Run the script `scripts/train_retain.ipynb` with different seeds.

2. **Obtaining the Logits**:
   - Run the script `scripts/create_logits.ipynb`.
   - Change paths if needed.
   - To obtain the logits of the retained models, set `MODE` to `"retain"`. For unlearned models, set `MODE` to `"unlearn"`.
   - To obtain logits for a specific approach, modify the line with `unlearn.two_stage_simple` to the desired approach (found in `machine_unlearning/unlearn.py`).

3. **Evaluate the Approaches**:
   - Run the script `scripts/attack.py` with the following usage:
     ```bash
     python3 attack.py <data_path> <approach_name>
     ```
   - The data should be organized as follows:
     ```
     data/
     +-- Approach1/
     |   +-- logits_unlearn.npy
     |   +-- metrics.csv
     +-- Approach2/
     |   +-- logits_unlearn.npy
     |   +-- metrics.csv
     .
     .
     +-- logits_retain.npy
     +-- metrics.csv
     ```

### AgeDB Results
1. Run the script `get_results_AgeDB.py` to generate results for Table 2.

---

## Requirements
1. **Datasets and Checkpoints**:
   - Download and extract the following files in the base directory:
     - [AgeDB.zip](https://polybox.ethz.ch/index.php/s/1h8Y1z9vbLeldzB): Contains the AgeDB dataset.
     - [checkpoints.zip](https://polybox.ethz.ch/index.php/s/1h8Y1z9vbLeldzB): Contains checkpoints for models finetuned on the retain set.
   - Download and extract [data.zip](https://polybox.ethz.ch/index.php/s/Je6DHi0M1l9S7zk) in the base directory.

2. **Python Environment**:
   - Ensure you have the required Python packages installed. Use the following command to install dependencies:
     ```bash
     pip install -r requirements.txt
     ```

---

## References
- **Carlini et al. (2022)**: Membership inference attacks from first principles.
- **Eleni Triantafillou et al. (2023)**: Evaluation for the NeurIPS Machine Unlearning Competition.
- **He et al. (2016)**: Deep residual learning for image recognition.
- **Shokri et al. (2017)**: Membership inference attacks against machine learning models.

---

## Team Members
1. Max Krahenmann
2. Leo Neubecker
3. Virgilio Strozzi
4. Igor Martinelli

---
