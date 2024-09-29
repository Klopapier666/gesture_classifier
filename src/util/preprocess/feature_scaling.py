class StandardScaler_pandas:
    def __init__(self):
        self.std = None
        self.mean = None

    def fit(self, X):
        """
        Fits the scaler
        :param X: input as pd data frame
        """
        if 'ground_truth' in X.columns:
            X = X.drop(columns='ground_truth')
        if 'timestamp' in X.columns:
            X = X.drop(columns='timestamp')

        self.mean = X.mean()
        self.std = X.std()

        # the standard deviation can be 0 in certain cases,
        #  which provokes 'devision-by-zero' errors; we can
        #  avoid this by adding a small amount if std==0
        self.std[self.std == 0] = 0.00001

    def transform(self, X):
        #return (X - self.mean) / self.std
        columns_to_standardize = [column for column in X.columns if column not in ['timestamp', 'ground_truth']]
        for column in columns_to_standardize:
            X[column] = (X[column] - self.mean[column]) / self.std[column]

        return X

    def inverse_transform(self, X_scaled):
        # return X_scaled * self.std + self.mean
        columns_to_standardize = [column for column in X.columns if column not in ['timestamp', 'ground_truth']]
        for column in columns_to_standardize:
            X_scaled[column] = X_scaled[column] * self.std[column] + self.mean[column]

class StandardScaler():
    def fit(self, X):
        self.mean = X.mean(axis=0)
        self.std = X.std(axis=0)
        self.std[self.std == 0] = 0.00001

    def transform(self, X):
        return (X - self.mean) / self.std

    def inverse_transform(self, X_scaled):
        return X_scaled * self.std + self.mean