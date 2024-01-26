# Project Name: BirdCLEF 2023 Competition Model

## Overview
This repository contains the implementation of a model for the BirdCLEF 2023 competition. 
The model is built using a combination of ResNet and supervised contrastive learning techniques to 
achieve optimal results in bird sound classification.


## Demo of the Web App
https://github.com/VijayRavichander/BirdCLEF23-SCL/assets/58650933/c8300341-a1b6-436b-8988-91617bab5aa9

## Project Structure
- **checkpoints:** This folder contains the saved model checkpoints that can be used for prediction or further training.

- **data:** The data folder contains the necessary datasets required for training and testing the model.

- **dataset.py:** This script handles the loading and preprocessing of the dataset for training and evaluation.

- **model.py:** The model script contains the architecture and implementation of the reset and supervised contrastive learning model.

- **predict.py:** Use this script to make predictions using the trained model. It takes input data and outputs the predicted bird species.

- **requirements.txt:** Lists all the dependencies required to run the project. You can install them using `pip install -r requirements.txt`.

## Usage
1. Install the required dependencies using the command:
   ```
   pip install -r requirements.txt
   ```

2. Ensure that the necessary dataset files are placed in the `data` folder.

3. You can use the `app.py` script to use the app:
   
   ```
   streamlit run app.py
   ```

## Additional Resources
For more detailed information on training and inference, refer to my Kaggle account where you can find comprehensive notebooks:
[Vijay Ravichander](https://www.kaggle.com/vijayravichander)

Feel free to explore and use this repository for your bird sound classification projects. If you have any questions or issues, please reach out through the GitHub repository's issue tracker. Happy bird sound classification!
