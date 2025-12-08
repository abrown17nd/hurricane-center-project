#!/bin/bash
#$ -q gpu
#$ -l gpu_card=1
#$ -N yolo_singleobj_job
#$ -cwd
#$ -M abrown17@nd.edu
#$ -m abe            # Send mail when job begins, ends and aborts


# Load required system modules
module load cuda/12.1     # GPU libraries



# Change to the working directory
cd /scratch365/abrown17/comp_vision_hurricane

# Run the Python training script and log output
python3 train_yolo_singleobj.py > training_output_singleobj.log 2>&1
