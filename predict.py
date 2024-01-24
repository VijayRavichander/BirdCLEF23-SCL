import torch
from dataset import BirdDataset
from model import SupConCE
import config 
import pandas as pd
from pathlib import Path
import numpy as np

def predict(data_loader, model):
        
    model.to('cpu')
    model.eval()    
    predictions = []
    for en in range(len(ds_test)):
        #print(en)
        images = torch.from_numpy(ds_test[en])
        #print(images.shape)
        with torch.no_grad():
            outputs = model(images).sigmoid().detach().cpu().numpy()
        predictions.append(outputs)
            
    
    return predictions

df_train = pd.read_csv(config.train_path)

class_23 = sorted(df_train.primary_label.unique())
label_to_class = {k+1:v for (k, v) in enumerate(class_23)}

test_path = Path(config.test_path)

df_test = pd.DataFrame([
        (path.stem, *path.stem.split("_"), path) for path in test_path.parent.glob("*.ogg")
    ], columns=["filename", "name", "id", "path"])

ds_test = BirdDataset(df_test, sr = config.SR, duration = config.DURATION)

audio_model = SupConCE()

model_path = config.model_path

model = SupConCE.load_from_checkpoint(model_path, train_dataloader=None,validation_dataloader=None) 

print("Running Inference..")

preds = predict(ds_test, model)   

all_pred_class = [label_to_class[item] for item in np.ravel(np.argmax(preds, -1))]

major_pred_class = label_to_class[np.argmax(np.bincount(np.ravel(np.argmax(preds, axis = -1))))]

print(major_pred_class)

print(all_pred_class)

