import pandas as pd
from sklearn.model_selection import train_test_split
import chardet

class DataPreProcessor:
    def __init__(self):
        """
        Initialize the DataPreProcessor.
        """
        self._data = None
        self.cleaned_data = None
        self._train_data = None
        self._test_data = None
        self._train_size = 0.8
        self._test_size = 0.2
        self._random_state = 42

    def load_data(self, file_path, columns=None, sep='\t'):
        """
        Load data from a file and save it to the instance variable.
        This method assumes that the data is in a tab-separated format with no header.
        
        Args:
            file_path (str): Path to the data file.
            columns (list, optional): List of column names to use.
            sep (str, optional): Separator used in the file.

        Returns:
            None
        """
        # Detect file encoding
        with open(file_path, 'rb') as f:
            result = chardet.detect(f.read())
        encoding = result['encoding']
        self._data = pd.read_csv(file_path, sep=sep, header=None, encoding=encoding)
        if columns is not None:
            self._data.columns = [_.lower().replace(' ', '_').replace('\'', '').replace('-', '_') for _ in columns]
        else:
            self._data.columns = [f'col_{i}' for i in range(self._data.shape[1])]
        print(f"Data loaded with shape: {self._data.shape}")
    def show_info(self):
        """
        Print information about the data.
        """
        iscleaned = False
        if self.cleaned_data is not None:
            iscleaned = True
        data = self.cleaned_data if iscleaned else self._data
        print(f"Data Information: ")
        data.info()
        print(f"Data Sample: \n {data.head()}")

        
        
    def get_data(self):
        """
        Get the loaded data.
        """
        return self._data

    def clean_data(self, drop_columns=None):
        """
        Clean the data by handling missing values, duplicates, etc.

        Args:
            data: Raw data to be cleaned.

        Returns:
            data: Cleaned data.
        """
        # Dropping null columns
        self.cleaned_data = self.get_data()
        self.cleaned_data = self.cleaned_data.dropna(axis=1, how='all')
        # Dropping null rows
        self.cleaned_data = self.cleaned_data.dropna(axis=0, how='all')
        # Dropping specified columns
        if drop_columns is not None:
            self.cleaned_data = self.cleaned_data.drop(columns=drop_columns)
        # Dropping duplicate rows
        self.cleaned_data = self.cleaned_data.drop_duplicates()
 
        
        # Resetting index
        self.cleaned_data = self.cleaned_data.reset_index(drop=True)        

    def map_data(self, mapping_df, mapping_column, target_column):
        # Use an explicit ID column if available, else fallback to index
        if "id" in mapping_df.columns:
            map_dict = dict(zip(mapping_df[mapping_column].astype(str), mapping_df["id"]))
        else:
            map_dict = dict(zip(mapping_df[mapping_column].astype(str), mapping_df.index))
        # Safe mapping with default for unmatched items
        self.cleaned_data[target_column + "_id"] = [
            map_dict.get(i, -1) for i in self.cleaned_data[target_column].astype(str)
        ]
        self.cleaned_data = self.cleaned_data.drop(columns=[target_column])

    def join_data(self, other_df, left_on, right_on):
        """
        Join the cleaned data with another DataFrame.

        Args:
            other_df: DataFrame to join with.
            left_on: Column name in the cleaned data to join on.
            right_on: Column name in the other DataFrame to join on.

        Returns:
            None
        """
        if self.cleaned_data is None:
            self.cleaned_data = self.get_data()
        self.cleaned_data = self.cleaned_data.merge(other_df, left_on=left_on, right_on=right_on, how='left')
        # Dropping right columns with different column names
        if left_on != right_on:
            self.cleaned_data = self.cleaned_data.drop(columns=[right_on])
        self.cleaned_data = self.cleaned_data.dropna(axis=0, how='any')
        self.cleaned_data = self.cleaned_data.reset_index(drop=True)


    def split_data(self, data, train_size=0.8):
        """
        Split the data into training and testing sets.

        Args:
            data: Data to be split.
            train_size: Proportion of the data to include in the training set.

        Returns:
            None
        """
        self._train_data, self._test_data = train_test_split(data, train_size=train_size, random_state=self._random_state)
        
    def get_train_data(self):
        """
        Get the training data.
        """
        if self._train_data is None:
            self.split_data(self.cleaned_data)
        return self._train_data
    def get_test_data(self):
        """
        Get the testing data.
        """ 
        if self._test_data is None:
            self.split_data(self.cleaned_data)
        return self._test_data