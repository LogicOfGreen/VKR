import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime, timedelta
from prophet import Prophet
from prophet.serialize import model_from_json, model_to_json
from darts.models import TCNModel
from darts import TimeSeries
import torch
import tensorflow as tf
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras
from keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, RepeatVector,  TimeDistributed, Attention, Concatenate, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import callbacks
import pandas as pd
import matplotlib.pyplot as plt
import darts
from darts import TimeSeries
from darts.models import TransformerModel
from darts.dataprocessing.transformers import Scaler
from sklearn.metrics import mean_absolute_error
# Настройки страницы
st.set_page_config(
    page_title="Time Series Forecasting",
    layout="wide",
    page_icon="📈"
)


# Загрузка предобученных моделей
@st.cache_resource
def load_pretrained_models():
    models = {}
    try:
        # Загрузка модели Prophet
        with open('prophet_model.json', 'r') as f:
            models['prophet'] = model_from_json(f.read())

        # Загрузка модели TCN
        models['tcn'] = TCNModel.load("darts_future_model.pt")

        # Загрузка TensorFlow модели
        models['tf'] = tf.keras.models.load_model('tf_model.keras')

    except Exception as e:
        st.error(f"Ошибка загрузки моделей: {str(e)}")
    return models


def validate_dataset(df):
    """Проверка данных на отрицательные значения"""
    if (df['y'] < 0).any():
        negative_rows = df[df['y'] < 0]
        return False, negative_rows
    return True, None


def load_default_data():
    """Загрузка встроенного набора данных"""
    df = pd.read_csv('SunTable.csv', index_col=False)
    df.drop(df[df.isnull().N == True].index, inplace=True)
    df['days'] = df.apply(lambda x: x.N // 24, axis=1)
    df['months'] = df.apply(lambda x: x.N // (30 * 24), axis=1)
    df['years'] = df.apply(lambda x: x.N // (365 * 24), axis=1)
    df.drop('N', axis=1, inplace=True)

    DataDays = df.groupby('days').agg(
        {'V': 'mean', 'T': 'mean', 'P': 'mean', 'DirectRad': 'sum', 'ScatterRad': 'sum', 'SumRad': 'sum'})
    DataMonths = df.groupby('months').agg(
        {'V': 'mean', 'T': 'mean', 'P': 'mean', 'DirectRad': 'sum', 'ScatterRad': 'sum', 'SumRad': 'sum'})
    DataYears = df.groupby('years').agg(
        {'V': 'mean', 'T': 'mean', 'P': 'mean', 'DirectRad': 'sum', 'ScatterRad': 'sum', 'SumRad': 'sum'})

    DataDays = DataDays.reset_index()
    start_date = "2014-01-01"
    X_Prophet = DataDays[['days', 'SumRad']]

    X_Prophet['days'] = pd.to_datetime(start_date) + pd.to_timedelta(X_Prophet["days"], unit="D")
    X_Prophet.rename(columns={"days": "ds", "SumRad": "y"}, inplace=True)
    df1 = X_Prophet
    df1 = df1[:-1]
    return df1


def prepare_data(df, days=365):
    """Подготовка данных для прогнозирования"""
    df = df.sort_values('ds').tail(days)
    return df.set_index('ds')


def train_models(train_data):
    """Обучение всех моделей"""
    models = {}

    # Prophet
    modelP = Prophet(
        yearly_seasonality=True,  # Включить годовую сезонность
        weekly_seasonality=False,  # Отключить, если данные не недельные
        daily_seasonality=False,
        seasonality_mode='multiplicative',  # Для растущих трендов
        changepoint_prior_scale=0.05  # Сглаживание резких изменений тренда
    )

    prophet_model = modelP
    prophet_model.fit(train_data)
    models['prophet'] = prophet_model

    # TCN
    series = TimeSeries.from_dataframe(train_data, 'ds', 'y')

    train_series, test_series = series.split_before(pd.Timestamp('2024-12-24'))
    scaler = Scaler()
    train_scaled = scaler.fit_transform(train_series)
    test_scaled = scaler.transform(test_series)

    model = TransformerModel(
        input_chunk_length=365,
        output_chunk_length=365,
        nhead=4,
        num_encoder_layers=2,
        num_decoder_layers=2,
        d_model=64,
        batch_size=32,
        pl_trainer_kwargs={"accelerator": "cpu"}  # Для GPU замените на "gpu"
    )

    model.fit(train_scaled, epochs=20, verbose=True)
    models['tcn'] = model

    # tensorflow
    scaler = MinMaxScaler(feature_range=(0, 1))
    train_data['y'] = scaler.fit_transform(train_data[['y']])

    def create_dataset_from_df(dataframe, window_size, forecast_horizon):
        X, y = [], []
        data = dataframe['y'].values  # Используем только нормализованные значения

        # Создаем окна, сохраняя связь с исходным DataFrame
        for i in range(len(data) - window_size - forecast_horizon + 1):
            X.append(data[i:i + window_size])

            # Берем соответствующие строки из DataFrame для проверки дат
            start_idx = i + window_size
            end_idx = i + window_size + forecast_horizon
            y.append(data[start_idx:end_idx])

            # Для отладки: проверка соответствия дней
            # if dataframe['ds'].iloc[end_idx-1] - dataframe['ds'].iloc[start_idx] != forecast_horizon-1:
            # print(f"Ошибка в индексах: {i}")

        return np.array(X), np.array(y)

    # Параметры
    window_size = 730  # 2 года истории
    forecast_horizon = 365  # Прогноз на год

    X, y = create_dataset_from_df(train_data, window_size, forecast_horizon)

    # Преобразование в 3D-массив (samples, timesteps, features)
    X = X.reshape(-1, window_size, 1)
    y = y.reshape(-1, forecast_horizon, 1)

    def build_model(window_size, forecast_horizon):
        # Энкодер
        encoder_inputs = Input(shape=(window_size, 1))
        encoder = LSTM(128, return_sequences=False, dropout=0.2)(encoder_inputs)

        # Декодер
        decoder_input = RepeatVector(forecast_horizon)(encoder)
        decoder = LSTM(64, return_sequences=True, dropout=0.2)(decoder_input)
        decoder_output = TimeDistributed(Dense(1))(decoder)  # Линейная активация

        model = Model(encoder_inputs, decoder_output)
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mae')
        return model

    model = build_model(window_size, forecast_horizon)
    model.summary()

    split_idx = int(0.9 * len(X))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    # Контроль переобучения
    early_stop = callbacks.EarlyStopping(
        monitor='val_loss',
        patience=15,
        restore_best_weights=True
    )

    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=20,
        batch_size=64,
        callbacks=[
            callbacks.EarlyStopping(monitor='val_loss', patience=15),
            callbacks.ReduceLROnPlateau(factor=0.5, patience=7)
        ],
        verbose=1
    )

    models['tf'] = model

    return models


def make_predictions(models, data, model_type):
    """Создание прогнозов"""
    try:
        if model_type == 'prophet':
            future = models['prophet'].make_future_dataframe(periods=365)
            forecast = models['prophet'].predict(future)
            return forecast[['ds', 'yhat']].rename(columns={'yhat': 'y'}).set_index('ds').tail(365)

        elif model_type == 'tcn':

            series = TimeSeries.from_dataframe(data, 'ds', 'y')

            scaler = Scaler()
            train_scaled = scaler.fit_transform(series)

            pred_scaled = models['tcn'].predict(n=365)
            pred = scaler.inverse_transform(pred_scaled)

            dates = pred.time_index
            values = pred.values()

            pred_df = pd.DataFrame({"ds": dates, "y": values.flatten()})
            pred_df = pred_df.set_index('ds')
            return pred_df


        elif model_type == 'tf':
            scaler = MinMaxScaler(feature_range=(0, 1))

            last_sequence = data[['y']].values[-730:]
            last_sequence = scaler.fit_transform(last_sequence)

            # Преобразуем в формат модели: (1 пример, 730 дней, 1 признак)
            input_seq = last_sequence.reshape(1, 730, 1)

            # Получаем прогноз на 365 дней
            predictions = models['tf'].predict(input_seq, verbose=0)

            # Генерируем даты с 2025-12-30 по 2026-12-29 (ровно 365 дней)
            forecast_dates = pd.date_range(
                start='2025-12-30',
                periods=365,
                freq='D'
            )

            prediction_actual = scaler.inverse_transform(predictions.reshape(-1, 1))

            # Создаём DataFrame с прогнозами
            return pd.DataFrame({'ds': forecast_dates, 'y': prediction_actual.flatten()}).set_index('ds')

    except Exception as e:
        st.error(f"Ошибка прогнозирования: {str(e)}")
    return None


def main():
    st.title("Прогнозирование временных рядов")
    models = load_pretrained_models()

    # Загрузка данных
    st.sidebar.header("Настройки данных")
    use_custom_data = st.sidebar.checkbox("Использовать свои данные")

    if use_custom_data:
        uploaded_file = st.sidebar.file_uploader("Загрузите CSV файл", type=['csv'])
        if uploaded_file:
            try:
                df = pd.read_csv(uploaded_file)
                df['ds'] = pd.to_datetime(df['ds'])
                valid, error_data = validate_dataset(df)

                if not valid:
                    st.error("Обнаружены отрицательные значения в следующих строках:")
                    st.write(error_data)
                    return

                data = prepare_data(df)
            except Exception as e:
                st.error(f"Ошибка загрузки данных: {str(e)}")
                return
    else:
        data = load_default_data()

    # Режим работы
    st.sidebar.header("Режим работы")
    mode = st.sidebar.radio("Выберите режим:", ['Предсказание', 'Обучение'])

    if mode == 'Обучение' and not use_custom_data:
        st.warning("Для обучения моделей необходимо загрузить свои данные")
        return

    if mode == 'Обучение':
        st.header("Обучение моделей на новых данных")
        if st.button("Начать обучение"):
            with st.spinner('Обучение моделей...'):
                try:
                    models = train_models(data.reset_index())
                    st.success("Модели успешно обучены!")

                    # Сохранение моделей
                    with open('prophet_model.json', 'w') as f:
                        f.write(model_to_json(models['prophet']))

                    models['tcn'].save('darts_model.pkl')
                    models['tf'].save('tf_model.keras')

                except Exception as e:
                    st.error(f"Ошибка обучения: {str(e)}")

    st.header("Прогнозирование")
    model_type = st.selectbox("Выберите модель:", ['prophet', 'tcn', 'tf'])

    if st.button("Сделать прогноз"):

        with st.spinner('Идет прогнозирование...'):
            forecast = make_predictions(models, data, model_type)

            if forecast is not None:
                st.subheader("Результаты прогноза")
                data = data.set_index('ds')
                data = data.sort_values('ds').tail(365)
                print(data.columns)
                fig, ax = plt.subplots(figsize=(12, 6))
                print(data)
                print(forecast)
                data['y'].plot(ax=ax, label='Исторические данные')
                forecast['y'].plot(ax=ax, color='red', label='Прогноз')
                plt.legend()
                st.pyplot(fig)

                st.download_button(
                    label="Скачать прогноз",
                    data=forecast.reset_index().to_csv(index=False),
                    file_name='forecast.csv'
                )


if __name__ == "__main__":
    main()