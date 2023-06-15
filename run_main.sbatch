#!/bin/bash -l

#####################
# job-array example #
#####################

#SBATCH --job-name=cgan_features
#SBATCH --qos=short
#SBATCH --time=04:00:00

# 500MB memory per core
# this is a hard limit
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=jobspernode:four:1
#SBATCH --mem-per-cpu=15000MB
#SBATCH --gres=gpu:turing:1

# you may not place bash commands before the last SBATCH directive
module use /opt/insy/modulefiles
module load cuda/11.0 cudnn/11.0-8.0.3.33
module list

cd /home/nfs/gperin/AISY_Framework
source env/bin/activate
cd /home/nfs/gperin/paper_9_gan/cgan_sca


# default values
dataset_reference=ascad-variable
dataset_target=ASCAD
target_byte_reference=2
target_byte_target=2
feature_selection_type=OPOI
leakage_model=ID
gaussian_noise=0
result_folder=06_06_2023_19_07_15_7266770
task=train

while [[ $# -gt 0 ]]
do
    key="$1"
    case $key in
        --dataset_reference)
        dataset_reference="$2"
        shift
        ;;
        --dataset_target)
        dataset_target="$2"
        shift
        ;;
        --target_byte_reference)
        target_byte_reference="$2"
        shift
        ;;
        --target_byte_target)
        target_byte_target="$2"
        shift
        ;;
        --feature_selection_type)
        feature_selection_type="$2"
        shift
        ;;
        --leakage_model)
        leakage_model="$2"
        shift
        ;;
         --result_folder)
        result_folder="$2"
        shift
        ;;
        --gaussian_noise)
        gaussian_noise="$2"
        shift
        ;;
         --task)
        task="$2"
        shift
        ;;
        *)
        # unknown option
        ;;
    esac
    shift
done

# traces
echo "dataset_reference is set to: $dataset_reference"
echo "dataset_target is set to: $dataset_target"
echo "target_byte_reference is set to: $target_byte_reference"
echo "target_byte_target is set to: $target_byte_target"
echo "feature_selection_type is set to: $feature_selection_type"
echo "leakage_model is set to: $leakage_model"
echo "gaussian_noise is set to: $gaussian_noise"
echo "result_folder is set to: $result_folder"
echo "task is set to: $task"

srun python main.py --dataset_reference $dataset_reference --dataset_target $dataset_target --target_byte_reference $target_byte_reference --target_byte_target $target_byte_target --features 100 --unprotected 0 --feature_selection_type $feature_selection_type --leakage_model $leakage_model --gaussian_noise $gaussian_noise --result_folder $result_folder --task $task
exit 0


