import pickle
import sys
import pandas as pd


def read_data(filename):

    df = pd.read_parquet(filename)
    
    categorical = ['PUlocationID', 'DOlocationID']
    df['duration'] = df.dropOff_datetime - df.pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    
    return df


def make_prediction(df, year, month):
    with open('model.bin', 'rb') as f_in:
        dv, lr = pickle.load(f_in)
    print(f'loaded the model for predictions ---')

    categorical = ['PUlocationID', 'DOlocationID']
    dicts = df[categorical].to_dict(orient='records')
    X_val = dv.transform(dicts)
    y_pred = lr.predict(X_val)
    
    print(f'mean predicted duration:', round(y_pred.mean(), 2))

    df['ride_id'] = f'{year:04d}/{month:02d}_' + df.index.astype('str')

    df_result = pd.DataFrame()
    df_result.loc[:, 'ride_id'] = df['ride_id']
    df_result.loc[:, 'predicted_duration'] = y_pred

    return df_result

def save_predictions(result, year, month):
    output_file = f'{year:04d}-{month:02d}.parquet'
    result.to_parquet(
        output_file,
        engine='pyarrow',
        compression=None,
        index=False
    )

def run():
    year = int(sys.argv[1])
    month = int(sys.argv[2])

    filename = f'https://nyc-tlc.s3.amazonaws.com/trip+data/fhv_tripdata_{year:04d}-{month:02d}.parquet'
    
    print(f'reading data from: {filename} ---')
    df = read_data(filename)

    print(f'making predictions ----')
    df_pred = make_prediction(df, year, month)

    save_predictions(df_pred, year, month)
    print(f'prediction saved ---')


if __name__ == '__main__':
    run()