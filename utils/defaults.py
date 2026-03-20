import torch


# MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
MODEL_NAME = "kmseong/Llama3.2-3B-gsm8k-fullft-atfter-ssft"

BATCH_SIZE = 4
CONTEXT_LENGTH = 128
ITERATIONS = 100000
SAVE_CHECK_POINT_ITERATION = 100
VAL_LOSS_INTERVAL = 25
CHECKPOINT_FOLDER = "./checkpoints"

if torch.cuda.is_available():
    print("device set to CUDA")
    DEVICE = "cuda"
elif torch.mps.is_available():
    print("device set to MPS")
    DEVICE = "mps"
else:
    print("no gpu available, defaulting to CPU")
    DEVICE = "cpu"

DATASET_GSM_TRAINING = "dataset/gsm8k_processed_train.json"
DATASET_GSM_TESTING = "dataset/gsm8k_processed_test.json"