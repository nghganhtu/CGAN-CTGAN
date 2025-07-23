"""DataTransformer module."""

from collections import namedtuple

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from typing import Union
from rdt.transformers import ClusterBasedNormalizer, OneHotEncoder

SpanInfo = namedtuple('SpanInfo', ['dim', 'activation_fn'])
ColumnTransformInfo = namedtuple(
    'ColumnTransformInfo',
    ['column_name', 'column_type', 'transform', 'output_info', 'output_dimensions'],
)


class DataTransformer(object):
    """Data Transformer.

    Model continuous columns with a BayesianGMM and normalize them to a scalar between [-1, 1]
    and a vector. Discrete columns are encoded using a OneHotEncoder.
    """

    def __init__(self, max_clusters=10, weight_threshold=0.005):
        """Create a data transformer.

        Args:
            max_clusters (int):
                Maximum number of Gaussian distributions in Bayesian GMM.
            weight_threshold (float):
                Weight threshold for a Gaussian distribution to be kept.
        """
        self._max_clusters = max_clusters
        self._weight_threshold = weight_threshold
        self._class_column = None            # tên cột nhãn
        self._class_transform_info = None    # thông tin biến đổi riêng

    def _fit_continuous(self, data):
        """Train Bayesian GMM for continuous columns.

        Args:
            data (pd.DataFrame):
                A dataframe containing a column.

        Returns:
            namedtuple:
                A ``ColumnTransformInfo`` object.
        """
        column_name = data.columns[0]
        gm = ClusterBasedNormalizer(
            missing_value_generation='from_column',
            max_clusters=min(len(data), self._max_clusters),
            weight_threshold=self._weight_threshold,
        )
        gm.fit(data, column_name)
        num_components = sum(gm.valid_component_indicator)

        return ColumnTransformInfo(
            column_name=column_name,
            column_type='continuous',
            transform=gm,
            output_info=[SpanInfo(1, 'tanh'), SpanInfo(num_components, 'softmax')],
            output_dimensions=1 + num_components,
        )

    def _fit_discrete(self, data):
        """Fit one hot encoder for discrete column.

        Args:
            data (pd.DataFrame):
                A dataframe containing a column.

        Returns:
            namedtuple:
                A ``ColumnTransformInfo`` object.
        """
        column_name = data.columns[0]
        ohe = OneHotEncoder()
        ohe.fit(data, column_name)
        num_categories = len(ohe.dummies)

        return ColumnTransformInfo(
            column_name=column_name,
            column_type='discrete',
            transform=ohe,
            output_info=[SpanInfo(num_categories, 'softmax')],
            output_dimensions=num_categories,
        )

    def fit(self, raw_data, discrete_columns=(), class_column=None):
        """Fit the DataTransformer to the provided raw data.

        This method processes the input data, identifies discrete columns, and fits
        appropriate transformers for each column. If a class column is specified, it is
        handled separately and excluded from the feature matrix.

        raw_data : pd.DataFrame or np.ndarray
            The input data to fit the transformer on. If a NumPy array is provided,
            it will be converted to a DataFrame with string column names.
        discrete_columns : Iterable[str] or Iterable[int], optional
            List of column names or indices that should be treated as discrete (categorical).
        class_column : str or None, optional
            Name of the column to be used as the class label. If provided, it will be
            treated as a discrete column and excluded from the feature matrix.

        Raises
        ------
        ValueError
            If the specified class_column does not exist in the input data.

        Notes
        -----
        - Stores metadata about the original data types.
        - Initializes internal structures for column transformation information.
        - Fits discrete or continuous transformers for each column as appropriate.
        - Handles the class column separately if specified.
        """
        # 0. Chuẩn hoá đầu vào -----------------------------------------------------
        self.dataframe = isinstance(raw_data, pd.DataFrame)
        self._column_transform_info_list = []
        self.output_info_list = []          # cho feature matrix X
        self.output_dimensions = 0          # cho feature matrix X
        self.cond_dim = 0                   # số chiều của condition vector (y)
        self._class_column_index = None
        
        
        if not self.dataframe:
            # ép NumPy -> DataFrame, đồng thời đổi tên cột số thành str
            column_names = [str(i) for i in range(raw_data.shape[1])]
            raw_data = pd.DataFrame(raw_data, columns=column_names)
            discrete_columns = [str(c) for c in discrete_columns]
            if class_column is not None:
                class_column = str(class_column)

        # 1. Lưu meta về kiểu dữ liệu ban đầu
        self._column_raw_dtypes = raw_data.infer_objects().dtypes

        # 2. Ghi nhận class_column (nếu có) & đảm bảo nó thuộc discrete -----------
        self._class_column = class_column
                
        if class_column is not None:
            if class_column not in raw_data.columns:
                raise ValueError(f"`{class_column}` không tồn tại trong DataFrame")
            discrete_columns = set(discrete_columns)
            discrete_columns.add(class_column)
            
        

        # 4. Lặp từng cột & fit ----------------------------------------------------
        for idx, col in enumerate(raw_data.columns):
            # 4-a. Nếu là class_column  ⇒ fit OHE riêng, KHÔNG thêm vào X
            if col == self._class_column:
                self._class_column_index = idx
                self._class_transform_info = self._fit_discrete(raw_data[[col]])
                self.cond_dim = self._class_transform_info.output_dimensions
                continue

            # 4-b. Các cột còn lại
            if col in discrete_columns:
                cti = self._fit_discrete(raw_data[[col]])
            else:
                cti = self._fit_continuous(raw_data[[col]])

            # ghi nhận cho feature matrix X
            self._column_transform_info_list.append(cti)
            self.output_info_list.append(cti.output_info)
            self.output_dimensions += cti.output_dimensions


    def _transform_continuous(self, column_transform_info, data):
        column_name = data.columns[0]
        flattened_column = data[column_name].to_numpy().flatten()
        data = data.assign(**{column_name: flattened_column})
        gm = column_transform_info.transform
        transformed = gm.transform(data)

        #  Converts the transformed data to the appropriate output format.
        #  The first column (ending in '.normalized') stays the same,
        #  but the lable encoded column (ending in '.component') is one hot encoded.
        output = np.zeros((len(transformed), column_transform_info.output_dimensions))
        output[:, 0] = transformed[f'{column_name}.normalized'].to_numpy()
        index = transformed[f'{column_name}.component'].to_numpy().astype(int)
        output[np.arange(index.size), index + 1] = 1.0

        return output

    def _transform_discrete(self, column_transform_info, data):
        ohe = column_transform_info.transform
        return ohe.transform(data).to_numpy()

    def _synchronous_transform(self, raw_data, column_transform_info_list):
        """Take a Pandas DataFrame and transform columns synchronous.

        Outputs a list with Numpy arrays.
        """
        column_data_list = []
        for column_transform_info in column_transform_info_list:
            column_name = column_transform_info.column_name
            data = raw_data[[column_name]]
            if column_transform_info.column_type == 'continuous':
                column_data_list.append(self._transform_continuous(column_transform_info, data))
            else:
                column_data_list.append(self._transform_discrete(column_transform_info, data))

        return column_data_list

    def _parallel_transform(self, raw_data, column_transform_info_list):
        """Take a Pandas DataFrame and transform columns in parallel.

        Outputs a list with Numpy arrays.
        """
        processes = []
        for column_transform_info in column_transform_info_list:
            column_name = column_transform_info.column_name
            data = raw_data[[column_name]]
            process = None
            if column_transform_info.column_type == 'continuous':
                process = delayed(self._transform_continuous)(column_transform_info, data)
            else:
                process = delayed(self._transform_discrete)(column_transform_info, data)
            processes.append(process)

        return Parallel(n_jobs=-1)(processes)
    
    ### NEW METHODS ###
    def _transform_class(self, data: pd.DataFrame) -> np.ndarray:
        """Biến đổi duy nhất cột nhãn -> one-hot numpy array."""
        if self._class_transform_info is None:
            raise RuntimeError("Bạn chưa truyền `class_column` khi fit().")

        # dùng lại logic của discrete
        return self._transform_discrete(self._class_transform_info, data[[self._class_column]])

    def transform_class_only(self, raw_data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """API công khai: trả về condition-vector của nhãn."""
        if not isinstance(raw_data, pd.DataFrame):
            raw_data = pd.DataFrame(
                raw_data, columns=[str(i) for i in range(raw_data.shape[1])]
            )
        return self._transform_class(raw_data)

    def inverse_class(self, cond_vec: np.ndarray) -> Union[pd.Series, np.ndarray]:
        """Giải mã ngược condition-vector thành nhãn gốc."""
        if self._class_transform_info is None:
            raise RuntimeError("Không có class_column để inverse.")
        # cần DataFrame vì OneHotEncoder.reverse_transform yêu cầu
        dummy_df = pd.DataFrame(cond_vec.reshape(1, -1),
                            columns=list(self._class_transform_info.transform.get_output_sdtypes()))
        recovered = self._inverse_transform_discrete(self._class_transform_info, dummy_df.to_numpy())
        return recovered
    

    """def transform(self, raw_data):
        ""Take raw data and output a matrix data.""
        if not isinstance(raw_data, pd.DataFrame):
            column_names = [str(num) for num in range(raw_data.shape[1])]
            raw_data = pd.DataFrame(raw_data, columns=column_names)

        # Only use parallelization with larger data sizes.
        # Otherwise, the transformation will be slower.
        if raw_data.shape[0] < 500:
            column_data_list = self._synchronous_transform(
                raw_data, self._column_transform_info_list
            )
        else:
            column_data_list = self._parallel_transform(raw_data, self._column_transform_info_list)

        return np.concatenate(column_data_list, axis=1).astype(float)"""
        
    def transform(self, raw_data, return_cond: bool = True):
        """Biến đổi dữ liệu thành ma trận đặc trưng; tùy chọn trả thêm condition-vector."""
        if not isinstance(raw_data, pd.DataFrame):
            column_names = [str(num) for num in range(raw_data.shape[1])]
            raw_data = pd.DataFrame(raw_data, columns=column_names)

        # chọn synchronous hay parallel
        if raw_data.shape[0] < 500:
            column_data_list = self._synchronous_transform(raw_data, self._column_transform_info_list)
        else:
            column_data_list = self._parallel_transform(raw_data, self._column_transform_info_list)

        X = np.concatenate(column_data_list, axis=1).astype(float)

        if return_cond and self._class_column is not None:
            y_cond = self._transform_class(raw_data)
            return X, y_cond

        return X

    def _inverse_transform_continuous(self, column_transform_info, column_data, sigmas, st):
        gm = column_transform_info.transform
        data = pd.DataFrame(column_data[:, :2], columns=list(gm.get_output_sdtypes())).astype(float)
        data[data.columns[1]] = np.argmax(column_data[:, 1:], axis=1)
        if sigmas is not None:
            selected_normalized_value = np.random.normal(data.iloc[:, 0], sigmas[st])
            data.iloc[:, 0] = selected_normalized_value

        return gm.reverse_transform(data)

    def _inverse_transform_discrete(self, column_transform_info, column_data):
        ohe = column_transform_info.transform
        data = pd.DataFrame(column_data, columns=list(ohe.get_output_sdtypes()))
        return ohe.reverse_transform(data)[column_transform_info.column_name]

    def inverse_transform(self, data, y_cond=None, sigmas=None):
        """Take matrix data (features) và condition-vector y_cond → trả về dữ liệu gốc.

        Giữ nguyên kiểu (DataFrame / ndarray) giống input của transform().
        """
        st = 0
        recovered_column_data_list = []
        column_names = []

        # 1. Khôi phục các cột đặc trưng (đã tách nhãn)
        for column_transform_info in self._column_transform_info_list:
            dim = column_transform_info.output_dimensions
            column_data = data[:, st : st + dim]

            if column_transform_info.column_type == 'continuous':
                recovered_column_data = self._inverse_transform_continuous(
                    column_transform_info, column_data, sigmas, st
                )
            else:
                recovered_column_data = self._inverse_transform_discrete(
                    column_transform_info, column_data
                )

            recovered_column_data_list.append(recovered_column_data)
            column_names.append(column_transform_info.column_name)
            st += dim

        # 2. Khôi phục cột nhãn nếu đã tách khi fit()
        if getattr(self, "_class_column", None) is not None:
            if y_cond is None:
                raise ValueError(
                    "Phải truyền `y_cond` vì class_column đã được tách ra trong fit()."
                )

            # giải mã one-hot → giá trị nhãn
            recovered_class = self._inverse_transform_discrete(
                self._class_transform_info, y_cond
            )

            # chèn đúng vị trí gốc của cột nhãn
            insert_idx = getattr(self, "_class_column_index", len(column_names))
            recovered_column_data_list.insert(insert_idx, recovered_class)
            column_names.insert(insert_idx, self._class_column)

        # 3. Gộp & ép dtype về dạng ban đầu
        recovered_data = np.column_stack(recovered_column_data_list)
        recovered_data = pd.DataFrame(recovered_data, columns=column_names).astype(
            self._column_raw_dtypes
        )

        if not self.dataframe:
            recovered_data = recovered_data.to_numpy()

        return recovered_data

    def convert_column_name_value_to_id(self, column_name, value):
        """Get the ids of the given `column_name`."""
        discrete_counter = 0
        column_id = 0
        for column_transform_info in self._column_transform_info_list:
            if column_transform_info.column_name == column_name:
                break
            if column_transform_info.column_type == 'discrete':
                discrete_counter += 1

            column_id += 1

        else:
            raise ValueError(f"The column_name `{column_name}` doesn't exist in the data.")

        ohe = column_transform_info.transform
        data = pd.DataFrame([value], columns=[column_transform_info.column_name])
        one_hot = ohe.transform(data).to_numpy()[0]
        if sum(one_hot) == 0:
            raise ValueError(f"The value `{value}` doesn't exist in the column `{column_name}`.")

        return {
            'discrete_column_id': discrete_counter,
            'column_id': column_id,
            'value_id': np.argmax(one_hot),
        }
        
# --- IGNORE ---