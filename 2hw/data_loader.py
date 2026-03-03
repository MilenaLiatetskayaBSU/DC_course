import pandas as pd
import numpy as np
import requests
import io
import json
from typing import Optional, Union, Dict, Any
import os


class DataLoader:   
    def __init__(self):
        self.loaded_data = None
        self.source_info = {}
    
    def load_from_csv(self, file_path: str, **kwargs) -> pd.DataFrame:
        try:
            self.loaded_data = pd.read_csv(file_path, **kwargs)
            self.source_info = {
                'source': 'csv',
                'path': file_path,
                'shape': self.loaded_data.shape
            }
            print(f"Данные загружены из CSV. Размерность: {self.loaded_data.shape}")
            return self.loaded_data
        except Exception as e:
            print(f"Ошибка загрузки CSV: {e}")
            return None
    
    def load_from_json(self, file_path: str, **kwargs) -> pd.DataFrame:
        try:
            self.loaded_data = pd.read_json(file_path, **kwargs)
            self.source_info = {
                'source': 'json',
                'path': file_path,
                'shape': self.loaded_data.shape
            }
            print(f"Данные загружены из JSON. Размерность: {self.loaded_data.shape}")
            return self.loaded_data
        except Exception as e:
            print(f"Ошибка загрузки JSON: {e}")
            return None
    
    def load_from_excel(self, file_path: str, sheet_name: Union[str, int] = 0, **kwargs) -> pd.DataFrame:
        try:
            self.loaded_data = pd.read_excel(file_path, sheet_name=sheet_name, **kwargs)
            self.source_info = {
                'source': 'excel',
                'path': file_path,
                'sheet': sheet_name,
                'shape': self.loaded_data.shape
            }
            print(f"Данные загружены из Excel. Размерность: {self.loaded_data.shape}")
            return self.loaded_data
        except Exception as e:
            print(f"Ошибка загрузки Excel: {e}")
            return None
    
    def load_from_url(self, url: str, file_type: str = 'csv', **kwargs) -> pd.DataFrame:
        try:
            response = requests.get(url)
            response.raise_for_status()
            
            if file_type.lower() == 'csv':
                self.loaded_data = pd.read_csv(io.StringIO(response.text), **kwargs)
            elif file_type.lower() == 'json':
                self.loaded_data = pd.read_json(io.StringIO(response.text), **kwargs)
            elif file_type.lower() == 'excel':
                self.loaded_data = pd.read_excel(io.BytesIO(response.content), **kwargs)
            else:
                raise ValueError(f"Неподдерживаемый тип файла: {file_type}")
            
            self.source_info = {
                'source': 'url',
                'url': url,
                'file_type': file_type,
                'shape': self.loaded_data.shape
            }
            print(f"Данные загружены из URL. Размерность: {self.loaded_data.shape}")
            return self.loaded_data
        except Exception as e:
            print(f"Ошибка загрузки из URL: {e}")
            return None
    
    def load_from_api(self, api_url: str, params: Optional[Dict] = None, 
                      headers: Optional[Dict] = None, data_path: Optional[str] = None) -> pd.DataFrame:
        try:
            response = requests.get(api_url, params=params, headers=headers)
            response.raise_for_status()
            
            data = response.json()
            
            if data_path:
                for key in data_path.split('.'):
                    data = data[key]
            
            if isinstance(data, list):
                self.loaded_data = pd.DataFrame(data)
            elif isinstance(data, dict):
                self.loaded_data = pd.DataFrame([data])
            else:
                raise ValueError(f"Неожиданный формат данных: {type(data)}")
            
            self.source_info = {
                'source': 'api',
                'url': api_url,
                'shape': self.loaded_data.shape
            }
            print(f"Данные загружены из API. Размерность: {self.loaded_data.shape}")
            return self.loaded_data
        except Exception as e:
            print(f"Ошибка загрузки из API: {e}")
            return None
        
    def get_data_info(self) -> Dict:
        if self.loaded_data is None:
            return {"error": "Данные не загружены"}
        
        info = {
            'source': self.source_info,
            'shape': self.loaded_data.shape,
            'columns': list(self.loaded_data.columns),
            'dtypes': self.loaded_data.dtypes.astype(str).to_dict(),
            'memory_usage': f"{self.loaded_data.memory_usage(deep=True).sum() / 1024**2:.2f} MB"
        }
        return info



def load_csv(file_path: str, **kwargs) -> pd.DataFrame:
    loader = DataLoader()
    return loader.load_from_csv(file_path, **kwargs)


def load_from_url(url: str, file_type: str = 'csv', **kwargs) -> pd.DataFrame:
    loader = DataLoader()
    return loader.load_from_url(url, file_type, **kwargs)

