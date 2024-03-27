import requests
import logging
import numpy as np
import pandas as pd
from pycaret.time_series import *

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DataProcessor:
    pass
    
    
class ModelTrainer:
    pass

class DataLoader:
    def __init__(self, fh) -> None:
        """Initialize DataLoader instance and load dataset.
        """        
        self.destination = "daily-min-temperatures.csv"
        #self._load_dataset()
        self.df = self.read_dataset()
        self.best_model = None
        self.fh = fh
    def _load_dataset(self):
        """
        Load dataset from URL and save locally.
        """        
        url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/daily-min-temperatures.csv"
       
        try:
            response=requests.get(url)
            response.raise_for_status()
            with open(self.destination,'wb') as f:
                f.write(response.content)
            logger.info("File download succesfully.")
        except requests.exceptions.RequestException as e :
            logger.error(f"Failed to download the file.{e}")
        except IOError as e:
            logger.error(f"Error writing to file: {e}")
            
    def read_dataset(self):
        try:
            df = pd.read_csv(self.destination)
            logger.info(f"Data loaded successfully\n{df}")
            return df
        except Exception as e:
            logger.error("Error while loading dataframe")
            return None
        
    def _increase_feature(self):
        self.df["Date"] = pd.to_datetime(self.df["Date"])
        self.df['Month'] = [i.month for i in self.df['Date']]
        self.df['Year'] = [i.year for i in self.df['Date']]
        self.df['Day'] = [i.day for i in self.df['Date']]
        self.df['Day_of_year'] = [i.dayofyear for i in self.df['Date']]
        self.df['Temp'] = self.df['Temp'].reset_index(drop=True)
        self.df['Series'] = np.arange(1,len(self.df)+1)
        self.df.drop(['Date'], axis=1, inplace=True)
        self.df = self.df[['Temp','Day','Month','Year','Day_of_year']]
        
    def preprocess_dataset(self):
        if self.df is None:
            logger.error("DataFrame is not loaded.")
            return
        logger.info(f"NULL values:\n {self.df.isna().sum()}")
        self._increase_feature()
        logger.info(f"new dataframe:\n {self.df}")
        
    def PyCaret(self):
        s = setup(self.df,target='Temp',fh=self.fh, fold=5,session_id = 123)
        self.best_model = compare_models(errors="raise")
        logger.info(f"Best Model for PyCaret:\n {self.best_model} and fh: {self.fh}")
        
    def predictions(self):
        final_best = finalize_model(self.best_model)
        y_predict = predict_model(self.best_model, fh = self.fh)
        logger.info(f"predictions: {y_predict}")
        
    def plot_model(self):
        plot_model(self.best_model, plot = 'forecast', data_kwargs = {'fh' : self.fh})
        plot_model(self.best_model, plot = 'diagnostics')
        
    def visualize_data(self):
        pass
    
if __name__ == '__main__':
    data = DataLoader(fh=30)
    data.preprocess_dataset()
    data.PyCaret()
        
    