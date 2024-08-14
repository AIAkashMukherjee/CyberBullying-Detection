import sys
import pandas as pd
import numpy as np
from src.logger.custom_logging import logger
from src.exceptions.expection import CustomException
from sklearn.preprocessing import LabelEncoder
from dataclasses import dataclass
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from src.entity.config_entity import DataTransformationConfig
import emoji
import re
from src.utlis.utlis import save_obj
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

class DataTransformation:
    def __init__(self) -> None:
        self.data_transformation_config = DataTransformationConfig()

    def preprocess(self, text):
        """
        Preprocesses the given text by tokenizing, converting to lowercase, removing punctuation,
        hashtags, mentions, emojis, and stopwords.
        """
        tokens = word_tokenize(text)
        tokens = [token.lower() for token in tokens]
        tokens = [re.sub(r'[^\w\s]', '', token) for token in tokens]
        tokens = [re.sub(r'#\S+', '', token) for token in tokens]
        tokens = [re.sub(r'@\S+', '', token) for token in tokens]
        tokens = [token for token in tokens if not emoji.is_emoji(token)]
        stop_words = set(stopwords.words('english'))
        tokens = [token for token in tokens if token not in stop_words]
        return ' '.join(tokens)

    def combine_features(self, df):
        """
        Combine the 'Race', 'Religion', 'Gender', 'Sexual Orientation', and 'Miscellaneous' columns
        into a single 'Features' column and then combine it with the 'comment' column.
        """
    
        logger.info('enterd into combine feature function')

        df['Miscellaneous'] = df['Miscellaneous'].fillna('None')
        df['Features'] = (df['Race'] + ' ' + df['Religion'] + ' ' +
                          df['Gender'] + ' ' + df['Sexual Orientation'] + ' ' +
                          df['Miscellaneous'])
        df['text'] = df['comment'] + ' ' + df['Features']
        return df
    
    def apply_preprocessing(self, df):
        """
        Apply text preprocessing to the 'text' column.
        """

        logger.info('Applying preprocessing function')

        df['text'] = df['text'].apply(self.preprocess)
        return df

    def get_text_pipeline(self):
        """
        Define the data transformation pipeline for text features.
        """
        try:
            # Define the pipeline for text feature transformation
            text_pipeline = Pipeline([
                ('tfidf', TfidfVectorizer())
            ])
            return text_pipeline

        except Exception as e:
            raise CustomException(e, sys)
        
    def initiate_data_transformation(self, train_path, test_path):
        """
        Perform data transformation on the train and test datasets.
        """
        try:
            logger.info('Entered into Data Transformation')
            # Load data
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logger.info("Read train and test data completed")

            # Combine features and apply preprocessing
            train_df = self.combine_features(train_df)
            train_df = self.apply_preprocessing(train_df)
            test_df = self.combine_features(test_df)
            test_df = self.apply_preprocessing(test_df)

            logger.info('Applied Preprocess function')

            # Get text transformation pipeline
            text_pipeline = self.get_text_pipeline()

            # Transform text data
            X_train = text_pipeline.fit_transform(train_df['text'])
            X_test = text_pipeline.transform(test_df['text'])

            logger.info('Transformed X_train and X test')

            # Encode labels
            label_encoder = LabelEncoder()
            train_labels = label_encoder.fit_transform(train_df['label'])
            test_labels = label_encoder.transform(test_df['label'])

            logger.info('Transformed train label and test label')

            # Combine features and labels
            train_array = np.c_[X_train.toarray(), train_labels]
            test_array = np.c_[X_test.toarray(), test_labels]

            # Save the text transformation pipeline
            save_obj(file_path=self.data_transformation_config.preprocessor_obj_file_path, obj=text_pipeline)

            logger.info(f"Preprocessor object saved at {self.data_transformation_config.preprocessor_obj_file_path}")

            logger.info('Exited from Data Transformation')

            return train_array, test_array, self.data_transformation_config.preprocessor_obj_file_path

        except Exception as e:
            raise CustomException(e, sys)
