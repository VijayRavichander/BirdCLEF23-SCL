import streamlit as st
import torch 
import pandas as pd
from dataset import BirdDataset
from model import SupConCE
import config 
import numpy as np
from pathlib import Path
from utils import label_class_converter, class_common_name_converter

def predict(data_loader, model):
        
    model.to('cpu')
    model.eval()    
    predictions = []
    for en in range(len(data_loader)):
        #print(en)
        images = torch.from_numpy(data_loader[en])
        #print(images.shape)
        with torch.no_grad():
            outputs = model(images).sigmoid().detach().cpu().numpy()
        predictions.append(outputs)
            
    
    return predictions

# Streamlit app
st.title('Bird Audio Classification')

uploaded_file = st.file_uploader("Upload Bird Audio", type=["ogg"])

if uploaded_file is not None:
    # Load and display the uploaded audio file
    audio_bytes = uploaded_file.read()

    st.audio(audio_bytes, format='audio/ogg')

    # Perform classification when user clicks the button
    if st.button('Classify'):

        label_to_class = label_class_converter()

        class_to_common_name= class_common_name_converter()

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
        
        # Display prediction
        st.write('### Prediction:', class_to_common_name[major_pred_class])