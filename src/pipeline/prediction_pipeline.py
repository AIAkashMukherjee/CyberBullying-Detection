# Create prediction pipeline class -> completed
# create function for load a object -> completed
# Create custome class basd upon our dataset -> completed
# Create function to convert data into Dataframe with the help of Dict

import os,sys
from src.logger.custom_logging import logger
from src.exceptions.expection import CustomException
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from dataclasses import dataclass
from src.components.data_ingestion import DataIngestion
from src.utlis.utlis import load_obj
import pandas as pd


class PredicitonPipeline:
    def __init__(self) -> None:
        pass

    def predict(self,features):
        preprocessor_obj_path=os.path.join("artifacts/data_transformation", "preprocessor.pkl")

        model_path=os.path.join("artifacts/model_trainer", "model.pkl")
        
        processor=load_obj(preprocessor_obj_path)
        model=load_obj(model_path)

        scaled=processor.transform(features)
        pred=model.predict(scaled)

        return pred
    

class CustomClass:
    def __init__(self,comment,	Race,	Religion,	Gender,	Sexual_Orientation,	Miscellaneous):
    
        self.comment =comment
        self.Race=Race
        self.Religion=Religion
        self.Gender=Gender
        self.Sexual_Orientation=Sexual_Orientation
        self.Misc=Miscellaneous
        


    def get_data_DataFrame(self):
        try:
            custom_input= {
                'comment':[self.comment],
                'Race':[self.Race],
                'Religion':[self.Religion],
                'Gender':[self.Gender],
                'Sexual_Orien':[self.Sexual_Orientation],
                'Misc':[self.Misc]
            }
            data=pd.DataFrame(custom_input)
            return data
        except Exception as e:
            raise CustomException(e,sys)    


