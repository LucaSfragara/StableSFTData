# Project Setup and Execution Guide

## How to Run

```bash
# 1. SSH into the cluster
ssh <username>@<cluster_address>

# 2. Request GPU resources (adjust command for your scheduler)
srun --gres=gpu:<num_gpus> --pty bash

# 3. Load the Miniforge module
module load miniforge/24.3.0-0

# 4. Activate your environment
conda activate <env_name>


# 5. Extract answer from GSM8K and save result
python extract_answers.py