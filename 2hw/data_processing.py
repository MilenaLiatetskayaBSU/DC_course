import pandas as pd
import numpy as np
from typing import Optional, Union, List, Dict, Any
import matplotlib.pyplot as plt
import seaborn as sns


class DataProcessor:
    
    def __init__(self, df: Optional[pd.DataFrame] = None):
        self.df = df
        self.original_shape = None
        if df is not None:
            self.original_shape = df.shape
        self.processing_history = []
    
    def set_data(self, df: pd.DataFrame):
        self.df = df
        self.original_shape = df.shape
        self.processing_history = []
        print(f"Данные установлены. Размерность: {df.shape}")

    def count_missing_values(self, columns: Optional[List[str]] = None, 
                             as_percentage: bool = False) -> pd.Series:
        if self.df is None:
            raise ValueError("Данные не загружены. Используйте set_data()")
        
        if columns is None:
            columns = self.df.columns.tolist()
        else:
            invalid_cols = [col for col in columns if col not in self.df.columns]
            if invalid_cols:
                print(f"Предупреждение: столбцы {invalid_cols} не найдены в данных")
                columns = [col for col in columns if col in self.df.columns]
        
        if not columns:
            print("Нет валидных столбцов для анализа")
            return pd.Series()
        
        missing_counts = self.df[columns].isnull().sum()
        
        if as_percentage:
            missing_counts = (missing_counts / len(self.df)) * 100
            missing_counts = missing_counts.round(2)
        
        missing_counts = missing_counts.sort_values(ascending=False)
        
        return missing_counts
    
    def get_columns_with_missing(self, threshold: float = 0) -> List[str]:
        if self.df is None:
            raise ValueError("Данные не загружены. Используйте set_data()")
        
        missing_counts = self.df.isnull().sum()
        
        if threshold < 1:  
            missing_percent = (missing_counts / len(self.df)) * 100
            cols_with_missing = missing_percent[missing_percent > threshold].index.tolist()
        else:  # Абсолютное значение
            cols_with_missing = missing_counts[missing_counts > threshold].index.tolist()
        
        return cols_with_missing

    def missing_values_report(self, detailed: bool = True, 
                             sort_by: str = 'count',
                             include_visualization: bool = False) -> pd.DataFrame:
        if self.df is None:
            raise ValueError("Данные не загружены. Используйте set_data()")
        # Основная статистика по пропускам
        missing_counts = self.df.isnull().sum()
        missing_percent = (missing_counts / len(self.df)) * 100
        
        report_data = {
            'Колонка': missing_counts.index,
            'Тип данных': self.df.dtypes.values,
            'Всего строк': len(self.df),
            'Не пропущено': len(self.df) - missing_counts.values,
            'Пропущено (шт)': missing_counts.values,
            'Пропущено (%)': missing_percent.values.round(2)
        }
        
        if detailed:
            report_data['Уникальных значений'] = [self.df[col].nunique() for col in self.df.columns]
            numeric_cols = self.df.select_dtypes(include=[np.number]).columns
            report_data['Среднее'] = [
                round(self.df[col].mean(), 2) if col in numeric_cols and not self.df[col].isnull().all() else None 
                for col in self.df.columns
            ]
            report_data['Медиана'] = [
                round(self.df[col].median(), 2) if col in numeric_cols and not self.df[col].isnull().all() else None 
                for col in self.df.columns
            ]
            report_data['Станд. откл.'] = [
                round(self.df[col].std(), 2) if col in numeric_cols and not self.df[col].isnull().all() else None 
                for col in self.df.columns
            ]
            report_data['Наиболее частое'] = [
                self.df[col].mode()[0] if not self.df[col].mode().empty and col not in numeric_cols else None
                for col in self.df.columns
            ]
        report_df = pd.DataFrame(report_data)
        
        if sort_by == 'count':
            report_df = report_df.sort_values('Пропущено (шт)', ascending=False)
        elif sort_by == 'percent':
            report_df = report_df.sort_values('Пропущено (%)', ascending=False)
        elif sort_by == 'column':
            report_df = report_df.sort_values('Колонка')
        total_missing = missing_counts.sum()
        total_cells = self.df.size
        missing_percent_total = (total_missing / total_cells) * 100
        cols_with_missing = (missing_counts > 0).sum()
        
        print("\n" + "="*60)
        print("ОТЧЕТ О ПРОПУЩЕННЫХ ЗНАЧЕНИЯХ")
        print("="*60)
        print(f"Всего строк: {len(self.df):,}")
        print(f"Всего колонок: {len(self.df.columns)}")
        print(f"Всего ячеек: {total_cells:,}")
        print(f"Всего пропущенных значений: {total_missing:,}")
        print(f"Общий процент пропусков: {missing_percent_total:.2f}%")
        print(f"Колонок с пропусками: {cols_with_missing} из {len(self.df.columns)}")
        print(f"Колонок без пропусков: {len(self.df.columns) - cols_with_missing}")
        print("="*60)
        if include_visualization and cols_with_missing > 0:
            self._plot_missing_values(missing_counts, missing_percent)
        
        return report_df
    
    
    
    def fill_missing_values(self, 
                           strategy: Union[str, Dict[str, str]] = 'mean',
                           columns: Optional[List[str]] = None,
                           fill_value: Optional[Any] = None,
                           inplace: bool = True,
                           **kwargs) -> Optional[pd.DataFrame]:
        if self.df is None:
            raise ValueError("Данные не загружены. Используйте set_data()")
        
        before_missing = self.df.isnull().sum().sum()
        
        if before_missing == 0:
            print("В данных нет пропущенных значений для заполнения")
            return self.df if not inplace else None
        
        if columns is None:
            columns = self.df.columns[self.df.isnull().any()].tolist()
        else:
            valid_cols = [col for col in columns if col in self.df.columns]
            invalid_cols = [col for col in columns if col not in self.df.columns]
            if invalid_cols:
                print(f"Предупреждение: колонки {invalid_cols} не найдены и будут пропущены")
            columns = valid_cols
        
        if not columns:
            print("Нет валидных колонок для обработки")
            return self.df if not inplace else None
        
        df_result = self.df if inplace else self.df.copy()
        
        if isinstance(strategy, dict):
            # Для каждой колонки своя стратегия
            filled_count = 0
            for col, col_strategy in strategy.items():
                if col in columns:
                    filled = self._apply_fill_strategy(
                        df_result, col, col_strategy, fill_value, **kwargs
                    )
                    if filled:
                        filled_count += filled
        else:
            # Одна стратегия для всех колонок
            filled_count = 0
            for col in columns:
                filled = self._apply_fill_strategy(
                    df_result, col, strategy, fill_value, **kwargs
                )
                if filled:
                    filled_count += filled
        
        after_missing = df_result.isnull().sum().sum()
        
        processing_info = {
            'operation': 'fill_missing',
            'strategy': strategy,
            'columns': columns,
            'before_missing': before_missing,
            'after_missing': after_missing,
            'filled_count': filled_count
        }
        self.processing_history.append(processing_info)
        
        print(f"\nЗАПОЛНЕНИЕ ПРОПУЩЕННЫХ ЗНАЧЕНИЙ ЗАВЕРШЕНО")
        print(f"   Стратегия: {strategy}")
        print(f"   Обработано колонок: {len(columns)}")
        print(f"   Заполнено пропусков: {filled_count}")
        print(f"   Осталось пропусков: {after_missing}")
        print(f"   Заполнено {((before_missing - after_missing) / before_missing * 100):.1f}% от всех пропусков")
        
        if inplace:
            return None
        else:
            return df_result
    
    def _apply_fill_strategy(self, df: pd.DataFrame, col: str, 
                             strategy: str, fill_value: Any, **kwargs) -> int:
        if df[col].isnull().sum() == 0:
            return 0
        
        missing_before = df[col].isnull().sum()
        
        try:
            if strategy in ['mean', 'median']:
                if not pd.api.types.is_numeric_dtype(df[col]):
                    print(f" Колонка {col} не числовая, стратегия '{strategy}' не применима")
                    return 0
                
                if strategy == 'mean':
                    fill_val = df[col].mean()
                else:  
                    fill_val = df[col].median()
                
                df[col].fillna(fill_val, inplace=True)
                
            elif strategy in ['mode', 'most_frequent']:
                mode_val = df[col].mode()
                if not mode_val.empty:
                    df[col].fillna(mode_val[0], inplace=True)
                else:
                    print(f"В колонке {col} нет моды (все значения NaN)")
                    return 0
                
            elif strategy == 'constant':
                if fill_value is None:
                    print(f"Для стратегии 'constant' требуется указать fill_value")
                    return 0
                df[col].fillna(fill_value, inplace=True)
                
            elif strategy == 'ffill':
                df[col].fillna(method='ffill', **kwargs, inplace=True)
                if df[col].isnull().any():
                    df[col].fillna(method='bfill', inplace=True)
                
            elif strategy == 'bfill':
                df[col].fillna(method='bfill', **kwargs, inplace=True)
                if df[col].isnull().any():
                    df[col].fillna(method='ffill', inplace=True)
                
            elif strategy == 'interpolate':
                if pd.api.types.is_numeric_dtype(df[col]):
                    df[col].interpolate(**kwargs, inplace=True)
                    df[col].fillna(method='ffill', inplace=True)
                    df[col].fillna(method='bfill', inplace=True)
                else:
                    print(f"Интерполяция применима только к числовым колонкам, колонка {col} пропущена")
                    return 0
            else:
                print(f"Неизвестная стратегия '{strategy}' для колонки {col}")
                return 0
                
        except Exception as e:
            print(f"Ошибка при заполнении колонки {col}: {e}")
            return 0
        
        missing_after = df[col].isnull().sum()
        return missing_before - missing_after
    
    def fill_by_group(self, group_col: str, strategy: str = 'mean',
                     columns: Optional[List[str]] = None,
                     inplace: bool = True) -> Optional[pd.DataFrame]:
        if self.df is None:
            raise ValueError("Данные не загружены. Используйте set_data()")
        
        if group_col not in self.df.columns:
            raise ValueError(f"Колонка {group_col} не найдена")
        
        if columns is None:
            columns = self.df.columns[self.df.isnull().any()].tolist()
            columns = [col for col in columns if col != group_col]
        
        df_result = self.df if inplace else self.df.copy()
        
        filled_count = 0
        groups = df_result[group_col].unique()
        
        for col in columns:
            if col not in df_result.columns:
                continue
                
            for group in groups:
                mask = (df_result[group_col] == group) & (df_result[col].isnull())
                if mask.sum() == 0:
                    continue
                
                group_data = df_result[df_result[group_col] == group][col]
                
                if strategy == 'mean':
                    fill_val = group_data.mean()
                elif strategy == 'median':
                    fill_val = group_data.median()
                elif strategy == 'mode':
                    mode_val = group_data.mode()
                    fill_val = mode_val[0] if not mode_val.empty else None
                else:
                    raise ValueError(f"Неизвестная стратегия: {strategy}")
                
                if pd.notna(fill_val):
                    df_result.loc[mask, col] = fill_val
                    filled_count += mask.sum()
        
        print(f"Заполнено {filled_count} пропусков по группам '{group_col}'")
        
        if inplace:
            return None
        else:
            return df_result
    
    
    def drop_missing(self, axis: int = 0, thresh: Optional[int] = None,
                    subset: Optional[List[str]] = None,
                    inplace: bool = True) -> Optional[pd.DataFrame]:
        if self.df is None:
            raise ValueError("Данные не загружены. Используйте set_data()")
        
        before_shape = self.df.shape
        before_missing = self.df.isnull().sum().sum()
        
        df_result = self.df if inplace else self.df.copy()
        
        if axis == 0:
            df_result.dropna(thresh=thresh, subset=subset, inplace=True)
            rows_removed = before_shape[0] - df_result.shape[0]
            print(f"Удалено строк: {rows_removed}")
        else:
            cols_before = set(df_result.columns)
            df_result.dropna(axis=1, thresh=thresh, inplace=True)
            cols_after = set(df_result.columns)
            cols_removed = cols_before - cols_after
            print(f"Удалено колонок: {len(cols_removed)}")
            if cols_removed:
                print(f"   Удаленные колонки: {list(cols_removed)}")
        
        after_missing = df_result.isnull().sum().sum()
        
        self.processing_history.append({
            'operation': 'drop_missing',
            'axis': axis,
            'before_shape': before_shape,
            'after_shape': df_result.shape,
            'before_missing': before_missing,
            'after_missing': after_missing
        })
        
        if inplace:
            return None
        else:
            return df_result
    
    def get_processing_history(self) -> pd.DataFrame:
        if not self.processing_history:
            return pd.DataFrame()
        return pd.DataFrame(self.processing_history)
    
    def suggest_fill_strategy(self, col: str) -> List[str]:
        if self.df is None or col not in self.df.columns:
            return []
        
        suggestions = []
        
        if pd.api.types.is_numeric_dtype(self.df[col]):
            suggestions.extend(['mean', 'median', 'interpolate'])
            
            if self.df[col].std() > self.df[col].mean() * 2:
                suggestions.remove('mean')
                print(f" В колонке {col} есть выбросы, рекомендуется median")
        else:
            suggestions.append('mode')
        
        suggestions.extend(['ffill', 'bfill', 'constant'])
        
        return suggestions


def quick_missing_report(df: pd.DataFrame, visualize: bool = True) -> pd.DataFrame:
    processor = DataProcessor(df)
    return processor.missing_values_report(include_visualization=visualize)


def quick_fill_missing(df: pd.DataFrame, strategy: str = 'mean',
                       columns: Optional[List[str]] = None) -> pd.DataFrame:
    processor = DataProcessor(df)
    return processor.fill_missing_values(strategy=strategy, columns=columns, inplace=False)


def compare_fill_strategies(df: pd.DataFrame, col: str) -> Dict[str, pd.Series]:
    if col not in df.columns:
        print(f"Колонка {col} не найдена")
        return {}
    
    processor = DataProcessor(df.copy())
    strategies = processor.suggest_fill_strategy(col)[:4]  # Берем первые 4
    
    results = {}
    original_missing = df[col].isnull().sum()
    
    print(f"Сравнение стратегий для колонки '{col}'")
    print(f"   Пропущено значений: {original_missing}")
    print("-" * 40)
    
    for strategy in strategies:
        df_test = df.copy()
        processor_test = DataProcessor(df_test)
        df_filled = processor_test.fill_missing_values(
            strategy=strategy, columns=[col], inplace=False
        )
        
        if df_filled is not None:
            filled = original_missing - df_filled[col].isnull().sum()
            results[strategy] = df_filled[col]
            print(f"   {strategy:12}: заполнено {filled:3d} значений")
    
    return results
