    
num_classes = 264
batch_size = 12
PRECISION = 16    
seed = 2023
model = "resnet50"
pretrained = False
use_mixup = False
mixup_alpha = 0.2   
DEVICE = 'mps' 

train_path = "./data/train.csv"
test_path = './data/test_audio.ogg'

#checkpoints
encoder_path = "checkpoints/birdclef_supconencoder.ckpt"
model_path = "checkpoints/birdclef_supconmodel.ckpt"

SR = 32000
DURATION = 5
LR = 5e-4