import config
import pandas as pd

def label_class_converter():
    df_train = pd.read_csv(config.train_path)

    class_23 = sorted(df_train.primary_label.unique())
    label_to_class = {k+1:v for (k, v) in enumerate(class_23)}

    return label_to_class

def class_common_name_converter():
    df_train = pd.read_csv(config.train_path)

    commonnames_dict = df_train.set_index('primary_label')['common_name'].to_dict()

    return commonnames_dict

