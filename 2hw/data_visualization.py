import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, List, Dict, Any, Union, Callable
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
COLORS = plt.cm.Set1(np.linspace(0, 1, 9))


class VisualizerManager:
    def __init__(self, df: Optional[pd.DataFrame] = None):
        self.df = df
        self.plots = []  
        self.figures = []  
        self.plot_counter = 0
        
    def set_data(self, df: pd.DataFrame):
        self.df = df
        
    def add_histogram(self, columns: Optional[Union[str, List[str]]] = None,
                      bins: int = 30, figsize: tuple = (12, 8),
                      title: Optional[str] = None, **kwargs) -> int:
        if self.df is None:
            raise ValueError("Данные не загружены. Используйте set_data()")
        
        plot_id = self.plot_counter + 1
        self.plot_counter += 1
        
        plot_info = {
            'id': plot_id,
            'type': 'histogram',
            'columns': columns,
            'bins': bins,
            'figsize': figsize,
            'title': title,
            'kwargs': kwargs,
            'created_at': pd.Timestamp.now()
        }
        
        self.plots.append(plot_info)
        
        return plot_id
    
    def add_scatter(self, x_col: str, y_col: str, color_col: Optional[str] = None,
                    size_col: Optional[str] = None, figsize: tuple = (10, 8),
                    title: Optional[str] = None, **kwargs) -> int:
        if self.df is None:
            raise ValueError("Данные не загружены. Используйте set_data()")
        
        plot_id = self.plot_counter + 1
        self.plot_counter += 1
        
        plot_info = {
            'id': plot_id,
            'type': 'scatter',
            'x_col': x_col,
            'y_col': y_col,
            'color_col': color_col,
            'size_col': size_col,
            'figsize': figsize,
            'title': title,
            'kwargs': kwargs,
            'created_at': pd.Timestamp.now()
        }
        
        self.plots.append(plot_info)
        
        return plot_id
    
    def add_line(self, x_col: str, y_cols: Optional[Union[str, List[str]]] = None,
                 figsize: tuple = (12, 6), title: Optional[str] = None,
                 markers: bool = True, **kwargs) -> int:
        if self.df is None:
            raise ValueError("Данные не загружены. Используйте set_data()")
        
        plot_id = self.plot_counter + 1
        self.plot_counter += 1
        
        plot_info = {
            'id': plot_id,
            'type': 'line',
            'x_col': x_col,
            'y_cols': y_cols,
            'figsize': figsize,
            'title': title,
            'markers': markers,
            'kwargs': kwargs,
            'created_at': pd.Timestamp.now()
        }
        
        self.plots.append(plot_info)
        
        return plot_id
    
    def add_boxplot(self, columns: Optional[Union[str, List[str]]] = None,
                    by: Optional[str] = None, figsize: tuple = (12, 6),
                    title: Optional[str] = None, **kwargs) -> int:
        if self.df is None:
            raise ValueError("Данные не загружены. Используйте set_data()")
        
        plot_id = self.plot_counter + 1
        self.plot_counter += 1
        
        plot_info = {
            'id': plot_id,
            'type': 'boxplot',
            'columns': columns,
            'by': by,
            'figsize': figsize,
            'title': title,
            'kwargs': kwargs,
            'created_at': pd.Timestamp.now()
        }
        
        self.plots.append(plot_info)
        
        return plot_id
    
    def add_heatmap(self, figsize: tuple = (10, 8), title: Optional[str] = None,
                    annot: bool = True, cmap: str = 'coolwarm', **kwargs) -> int:
        if self.df is None:
            raise ValueError("Данные не загружены. Используйте set_data()")
        
        plot_id = self.plot_counter + 1
        self.plot_counter += 1
        
        plot_info = {
            'id': plot_id,
            'type': 'heatmap',
            'figsize': figsize,
            'title': title,
            'annot': annot,
            'cmap': cmap,
            'kwargs': kwargs,
            'created_at': pd.Timestamp.now()
        }
        
        self.plots.append(plot_info)
        
        return plot_id
    
    
    def remove_plot(self, plot_id: int) -> bool:
        for i, plot in enumerate(self.plots):
            if plot['id'] == plot_id:
                removed = self.plots.pop(i)
                print(f"График '{removed['type']}' (ID: {plot_id}) удален")
                return True
        
        print(f"График с ID {plot_id} не найден")
        return False
    
    def remove_last_plot(self) -> bool:
        if self.plots:
            removed = self.plots.pop()
            print(f"Последний график '{removed['type']}' (ID: {removed['id']}) удален")
            return True
        else:
            print("Список графиков пуст")
            return False
    
    def remove_all_plots(self):
        count = len(self.plots)
        self.plots.clear()
        print(f"Удалено {count} графиков")
    
    def remove_by_type(self, plot_type: str) -> int:
        before_count = len(self.plots)
        self.plots = [p for p in self.plots if p['type'] != plot_type]
        removed_count = before_count - len(self.plots)
        
        print(f"Удалено {removed_count} графиков типа '{plot_type}'")
        return removed_count
    
    
    def show_plot(self, plot_id: Optional[int] = None) -> Optional[plt.Figure]:
        if not self.plots:
            print("Нет графиков для отображения")
            return None
        
        if plot_id is not None:
            for plot in self.plots:
                if plot['id'] == plot_id:
                    return self._render_plot(plot)
            print(f" График с ID {plot_id} не найден")
            return None
        else:
            if len(self.plots) == 1:
                return self._render_plot(self.plots[0])
            else:
                for plot in self.plots:
                    print(f"\n--- График ID: {plot['id']}, Тип: {plot['type']} ---")
                    self._render_plot(plot)
                    plt.show()
                return None
    
    def _render_plot(self, plot_info: Dict) -> plt.Figure:
        plot_type = plot_info['type']
        
        if plot_type == 'histogram':
            return self._render_histogram(plot_info)
        elif plot_type == 'scatter':
            return self._render_scatter(plot_info)
        elif plot_type == 'line':
            return self._render_line(plot_info)
        elif plot_type == 'boxplot':
            return self._render_boxplot(plot_info)
        elif plot_type == 'heatmap':
            return self._render_heatmap(plot_info)
        else:
            raise ValueError(f"Неизвестный тип графика: {plot_type}")
    
    def _render_histogram(self, plot_info: Dict) -> plt.Figure:
        columns = plot_info['columns']
        bins = plot_info['bins']
        figsize = plot_info['figsize']
        title = plot_info['title']
        kwargs = plot_info['kwargs']
        
        if columns is None:
            numeric_cols = self.df.select_dtypes(include=[np.number]).columns[:6]
            columns = numeric_cols.tolist()
        elif isinstance(columns, str):
            columns = [columns]
        
        valid_cols = [col for col in columns if col in self.df.columns]
        
        if not valid_cols:
            print("Нет валидных столбцов для отображения")
            return None
        
        n_cols = len(valid_cols)
        n_rows = (n_cols + 2) // 3
        
        fig, axes = plt.subplots(n_rows, min(3, n_cols), figsize=figsize)
        
        if n_rows == 1 and n_cols == 1:
            axes = np.array([axes])
        elif n_rows == 1:
            axes = axes.reshape(1, -1)
        
        axes_flat = axes.flatten()
        
        for i, col in enumerate(valid_cols):
            ax = axes_flat[i]
            
            if pd.api.types.is_numeric_dtype(self.df[col]):
                ax.hist(self.df[col].dropna(), bins=bins, edgecolor='black', 
                       alpha=0.7, color=COLORS[i % len(COLORS)], **kwargs)
                ax.set_xlabel(col)
                ax.set_ylabel('Частота')
            else:
                # Для категориальных данных
                value_counts = self.df[col].value_counts().head(20)
                ax.bar(range(len(value_counts)), value_counts.values, 
                      tick_label=value_counts.index, color=COLORS[i % len(COLORS)])
                ax.set_xlabel(col)
                ax.set_ylabel('Количество')
                plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
            
            ax.set_title(f'Распределение: {col}')
            ax.grid(True, alpha=0.3)
        
        for j in range(i + 1, len(axes_flat)):
            axes_flat[j].set_visible(False)
        
        plt.tight_layout()
        
        if title:
            fig.suptitle(title, fontsize=16, y=1.02)
        
        self.figures.append(fig)
        return fig
    
    def _render_scatter(self, plot_info: Dict) -> plt.Figure:
        x_col = plot_info['x_col']
        y_col = plot_info['y_col']
        color_col = plot_info['color_col']
        size_col = plot_info['size_col']
        figsize = plot_info['figsize']
        title = plot_info['title']
        kwargs = plot_info['kwargs']
        
        if x_col not in self.df.columns or y_col not in self.df.columns:
            print(f"Колонки {x_col} или {y_col} не найдены")
            return None
        
        fig, ax = plt.subplots(figsize=figsize)
        
        plot_df = self.df[[x_col, y_col]]
        if color_col and color_col in self.df.columns:
            plot_df = plot_df.join(self.df[color_col])
        if size_col and size_col in self.df.columns:
            plot_df = plot_df.join(self.df[size_col])
        
        size = None
        if size_col and size_col in self.df.columns:
            if pd.api.types.is_numeric_dtype(self.df[size_col]):
                size = (self.df[size_col] - self.df[size_col].min()) / (self.df[size_col].max() - self.df[size_col].min()) * 200 + 20
        
        if color_col and color_col in self.df.columns:
            if pd.api.types.is_numeric_dtype(self.df[color_col]):
                scatter = ax.scatter(self.df[x_col], self.df[y_col], 
                                    c=self.df[color_col], s=size, cmap='viridis',
                                    alpha=0.6, edgecolors='black', linewidth=0.5, **kwargs)
                plt.colorbar(scatter, ax=ax, label=color_col)
            else:
                categories = self.df[color_col].unique()
                colors = plt.cm.Set1(np.linspace(0, 1, len(categories)))
                for cat, color in zip(categories, colors):
                    mask = self.df[color_col] == cat
                    ax.scatter(self.df[mask][x_col], self.df[mask][y_col],
                              c=[color], s=size[mask] if size is not None else 50,
                              label=cat, alpha=0.6, edgecolors='black', linewidth=0.5, **kwargs)
                ax.legend()
        else:
            ax.scatter(self.df[x_col], self.df[y_col], s=size if size is not None else 50,
                      alpha=0.6, edgecolors='black', linewidth=0.5, **kwargs)
        
        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)
        ax.set_title(title or f'Диаграмма рассеяния: {x_col} vs {y_col}')
        ax.grid(True, alpha=0.3)
        
        self.figures.append(fig)
        return fig
    
    def _render_line(self, plot_info: Dict) -> plt.Figure:
        x_col = plot_info['x_col']
        y_cols = plot_info['y_cols']
        figsize = plot_info['figsize']
        title = plot_info['title']
        markers = plot_info['markers']
        kwargs = plot_info['kwargs']
        
        if x_col not in self.df.columns:
            print(f"Колонка {x_col} не найдена")
            return None
        
        if y_cols is None:
            numeric_cols = self.df.select_dtypes(include=[np.number]).columns
            y_cols = [col for col in numeric_cols if col != x_col][:5]
        elif isinstance(y_cols, str):
            y_cols = [y_cols]
        
        valid_y = [col for col in y_cols if col in self.df.columns]
        
        if not valid_y:
            print("Нет валидных Y-колонок")
            return None
        
        fig, ax = plt.subplots(figsize=figsize)
        
        plot_df = self.df.sort_values(x_col)
        
        for i, col in enumerate(valid_y):
            if pd.api.types.is_numeric_dtype(plot_df[col]):
                if markers:
                    ax.plot(plot_df[x_col], plot_df[col], marker='o', 
                           linestyle='-', linewidth=2, markersize=4,
                           color=COLORS[i % len(COLORS)], label=col, **kwargs)
                else:
                    ax.plot(plot_df[x_col], plot_df[col], linestyle='-', 
                           linewidth=2, color=COLORS[i % len(COLORS)], 
                           label=col, **kwargs)
        
        ax.set_xlabel(x_col)
        ax.set_ylabel('Значение')
        ax.set_title(title or f'Линейный график')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        self.figures.append(fig)
        return fig
    
    def _render_boxplot(self, plot_info: Dict) -> plt.Figure:
        columns = plot_info['columns']
        by = plot_info['by']
        figsize = plot_info['figsize']
        title = plot_info['title']
        kwargs = plot_info['kwargs']
        
        if columns is None:
            numeric_cols = self.df.select_dtypes(include=[np.number]).columns[:6]
            columns = numeric_cols.tolist()
        elif isinstance(columns, str):
            columns = [columns]
        
        valid_cols = [col for col in columns if col in self.df.columns]
        
        if not valid_cols:
            print("Нет валидных столбцов")
            return None
        
        fig, ax = plt.subplots(figsize=figsize)
        
        if by and by in self.df.columns:
            data_to_plot = [self.df[self.df[by] == val][col].dropna() 
                           for val in self.df[by].unique() 
                           for col in valid_cols[:1]]  # Берем только первый столбец для группировки
            labels = [f"{val}" for val in self.df[by].unique()]
            ax.boxplot(data_to_plot, labels=labels, **kwargs)
            ax.set_xlabel(by)
        else:
            ax.boxplot([self.df[col].dropna() for col in valid_cols], 
                      labels=valid_cols, **kwargs)
        
        ax.set_title(title or 'Ящик с усами')
        ax.grid(True, alpha=0.3)
        
        self.figures.append(fig)
        return fig
    
    def _render_heatmap(self, plot_info: Dict) -> plt.Figure:
        figsize = plot_info['figsize']
        title = plot_info['title']
        annot = plot_info['annot']
        cmap = plot_info['cmap']
        kwargs = plot_info['kwargs']
        
        numeric_df = self.df.select_dtypes(include=[np.number])
        
        if len(numeric_df.columns) < 2:
            print("Недостаточно числовых колонок для тепловой карты")
            return None
        
        corr = numeric_df.corr()
        
        fig, ax = plt.subplots(figsize=figsize)
        
        sns.heatmap(corr, annot=annot, cmap=cmap, center=0,
                   square=True, linewidths=1, ax=ax, **kwargs)
        
        ax.set_title(title or 'Тепловая карта корреляций')
        
        self.figures.append(fig)
        return fig
    
    
    def list_plots(self) -> pd.DataFrame:
        if not self.plots:
            print("Список графиков пуст")
            return pd.DataFrame()
        
        plots_info = []
        for plot in self.plots:
            info = {
                'ID': plot['id'],
                'Тип': plot['type'],
                'Создан': plot['created_at'].strftime('%H:%M:%S'),
                'Параметры': self._get_plot_params_summary(plot)
            }
            plots_info.append(info)
        
        df = pd.DataFrame(plots_info)
        print(f"Всего графиков: {len(self.plots)}")
        return df
    
    def _get_plot_params_summary(self, plot: Dict) -> str:
        if plot['type'] == 'histogram':
            cols = plot['columns']
            if cols is None:
                return "все числовые"
            elif isinstance(cols, list):
                return f"{len(cols)} колонок"
            else:
                return cols
        elif plot['type'] == 'scatter':
            return f"{plot['x_col']} vs {plot['y_col']}"
        elif plot['type'] == 'line':
            y_cols = plot['y_cols']
            if y_cols is None:
                return f"{plot['x_col']} vs все"
            elif isinstance(y_cols, list):
                return f"{plot['x_col']} vs {len(y_cols)} линий"
            else:
                return f"{plot['x_col']} vs {y_cols}"
        else:
            return "—"
    
    def get_plot_count(self) -> Dict[str, int]:
        counts = {}
        for plot in self.plots:
            plot_type = plot['type']
            counts[plot_type] = counts.get(plot_type, 0) + 1
        return counts
    
    def clear_figures(self):
        self.figures.clear()
        plt.close('all')
        print("Все фигуры очищены")



def quick_histogram(df: pd.DataFrame, columns: Optional[List[str]] = None, 
                    bins: int = 30, figsize: tuple = (12, 8)) -> plt.Figure:
    viz = VisualizerManager(df)
    viz.add_histogram(columns=columns, bins=bins, figsize=figsize)
    return viz.show_plot()


def quick_scatter(df: pd.DataFrame, x_col: str, y_col: str, 
                  color_col: Optional[str] = None, figsize: tuple = (10, 8)) -> plt.Figure:
    viz = VisualizerManager(df)
    viz.add_scatter(x_col=x_col, y_col=y_col, color_col=color_col, figsize=figsize)
    return viz.show_plot()


def quick_line(df: pd.DataFrame, x_col: str, y_cols: Optional[List[str]] = None,
               figsize: tuple = (12, 6)) -> plt.Figure:
    viz = VisualizerManager(df)
    viz.add_line(x_col=x_col, y_cols=y_cols, figsize=figsize)
    return viz.show_plot()
