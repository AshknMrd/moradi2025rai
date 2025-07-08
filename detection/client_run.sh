#!/bin/bash

# Set all variables
export WORKDIR_PATH="./workdir"
export TASK_NAME="Task1001_detection_FL"
export NNUNET_WRAPPER_PATH="./train_pre_process_wrapper.py"
export INITIAL_MODEL_DIR="./initial_model.model"
export INITIAL_PLAN_DIR="./initial_plan.pkl"
export INITIAL_PLAN_IDENTIFIER="nnUNetData_plans_v2.1"
export FOLDS="0"
export CHECK_POINT="model_final_checkpoint.model"
export TRAINER="nnUNetTrainerV2_Loss_CE_FL"
export PLANS="nnUNetPlans_pretrained_nnUNetData_plans_v2.1"

echo "Setting environment variables:"
echo "---------------------------------------"
echo "  Running Python script with settings:"
echo "  WORKDIR_PATH         = $WORKDIR_PATH"
echo "  TASK_NAME            = $TASK_NAME"
echo "  CHECK_POINT          = $CHECK_POINT"
echo "  TRAINER              = $TRAINER"
echo "  PLANS                = $PLANS"
echo "---------------------------------------"
echo ""

# Run the Python script
python detection_client.py
