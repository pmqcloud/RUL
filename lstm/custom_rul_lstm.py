import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import MinMaxScaler

# ETL logic in a custom Transformer
class MyTransformer(_BaseTransformer):
    """This transformer transforms the given feature into event based on whether the feature value is NA.
    Non-NA values are transformed to be the given event label and NA values are remained NA.
    """

    def __init__(self, window_size, engine_list, col_to_remove):

        print ('In MyTransformer init.')
        super().__init__()
        self._col_to_remove = col_to_remove
        self._window_size = window_size
        self._engine_list = engine_list

    def execute(self, df, start_ts=None, end_ts=None, entities=None):
        print ('tony1 In MyTransformer execute.')
        if self._col_to_remove != None :
            df = df.drop(self._col_to_remove, axis=1)
        print (df.columns)
        print (df.head(5))
        print (df.tail(5))

        df.reset_index(inplace = True)

        dd = pd.DataFrame() # feature df
        di = pd.DataFrame() # index df

        if self._engine_list == None :
            self._engine_list = df['id'].unique()
        for i in self._engine_list : # TODO: take first engine
            # Step 0. Construct cycle for each engine
            df1 = df[df['id'] == i]
            df1.sort_values(by=['event_timestamp'], inplace=True, ascending=True)
            cycle_col = [*range(1, df1.shape[0]+1, 1)] # 1, 2, ... length
            df1.insert(0, 'cycle', cycle_col) # Note: cycle must be inserted to first column to match training data
            # Step 1, Save the index columns.
            df_ind = df1[['id', 'event_timestamp']].iloc[self._window_size:] # Take 2 col and drop first _window_size rows
            # Step 2, remove un-used columns
            df1 = df1.drop(['event_timestamp', 'id'], axis=1)
            # Step 3, construct X (convert time series to feature)
            df3 = self.series_to_supervised(df1, self._window_size, 1)
            dd = dd.append(df3)
            di = di.append(df_ind)

        #Step 4, scale X
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled = pd.DataFrame(data=scaler.fit_transform(dd), columns=dd.columns) # TODO: this is cheating, Need a WML enhancement
        #scaled = dd # No scaling. testing only

        # Step 5. put index back to the dataframe
        print ('tony2')
        print (di.shape)
        print (scaled.shape)
        di.reset_index(inplace = True)
        dd.reset_index(inplace = True)

        scaled['id'] = di['id']
        scaled['event_timestamp'] =  di['event_timestamp']

        #scaled = scaled[scaled['id'] == 'FD1_ENG_1-____-BEDFORD'] #TODO: to remove. take first engine for testing
        scaled.set_index(['id', 'event_timestamp'], inplace=True)
        scaled.sort_index(inplace=True)

        print(scaled.columns)
        print(scaled.head())
        print(scaled.head(163).tail())
        print ('End of MyTransformer execute. df.shape =' + str(scaled.shape))

        return scaled


    # convert series to supervised learning
    def series_to_supervised(self, data, n_in=1, n_out=1, dropnan=True):
        n_vars = 1 if type(data) is list else data.shape[1]
        df = pd.DataFrame(data)
        cols, names = list(), list()
        # input sequence (t-n, ... t-1)
        for i in range(n_in, 0, -1):
            cols.append(df.shift(i))
            names += [( df.columns[j] + '(t-%d)' % ( i)) for j in range(n_vars)]
        # forecast sequence (t, t+1, ... t+n)
        for i in range(0, n_out):
            cols.append(df.shift(-i))
            if i == 0:
                names += [(df.columns[j] + '(t)' ) for j in range(n_vars)]
            else:
                names += [(df.columns[j] + '(t+%d)' % ( i)) for j in range(n_vars)]
        # put it all together
        agg = pd.concat(cols, axis=1)
        agg.columns = names
        # drop rows with NaN values
        if dropnan:
            agg.dropna(inplace=True)
        return agg

# Custom pipeline to add ETL as a stage before WML call
class MyWMLPipeline( WmlScoringAssetGroupPipeline ):
    #Class Constructor
    def __init__(self, asset_group_id, model_pipeline, **kwargs ):
        print("In MyWMLPipeline init" )
        super().__init__(asset_group_id=asset_group_id, model_pipeline=model_pipeline, **kwargs)
        #self._col_to_remove = col_to_remove
        #self._window_size = window_size
        #self._engine_list = engine_list

    def prepare_execute(self, pipeline, model_config):
        print("In MyWMLPipeline prepare_execute")

        if 'engine_list' not in model_config.keys() :
            model_config['engine_list'] = None

        # Add ETL step to the pipeline
        pipeline.add_stage(MyTransformer(model_config['window_size'], model_config['engine_list'], model_config['col_to_remove']))

        if 'cache_model' not in model_config:
            model_config['cache_model'] = False

        if 'wml_credentials' not in model_config or model_config['wml_credentials'] is None:
            wml_credentials = None
            if 'WML_VCAPS' in os.environ:
                try:
                    wml_credentials = json.loads(os.environ['WML_VCAPS'])
                except:
                    pass
            model_config['wml_credentials'] = wml_credentials

        if model_config['wml_credentials'] is None:
            raise ValueError('missing mandatory model_pipeline.wml_credentials')

        estimator = MyWmlDeploymentEstimator(**model_config)
        #if len(model_config['features_for_training']) > 0:
        pipeline.add_stage(estimator)


class MyWmlDeploymentEstimator( WmlDeploymentEstimator ):
    # override this method to convert df to 3 dim array
    def get_df_for_prediction(self, df):
        if self.cache_model:
            return super().get_df_for_prediction(df=df)
        else:
            print("tony5 In MyWmlDeploymentEstimator:get_df_for_prediction false")
            print(df.shape)

            # for any column not of numeric or boolean type, cast they to string
            df = df.astype({col:str for col in df.select_dtypes(exclude=[np.number, np.bool]).columns})
            self.logger.debug('df_for_prediction=%s' % log_df_info(df, head=2, maxlen=500))

            #df1 = df.drop(['deviceid', '_timestamp'], axis=1)
            dd = df.values.reshape(df.shape[0], 1, df.shape[1]) # convert to 3 dim
            print ('tony6')
            print (dd.shape)

            payload_for_prediction = {"fields": list(df.columns),'values': dd.tolist()}
            print ('tony7 payload_for_prediction=%s' % str(payload_for_prediction['values'][:2]))
            self.logger.debug('payload_for_prediction=%s' % str(payload_for_prediction['values'][:2]))

            return payload_for_prediction
