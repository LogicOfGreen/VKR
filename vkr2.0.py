import streamlit as st
from prophet import Prophet
from prophet.serialize import model_from_json, model_to_json
from darts.models import TCNModel
import tensorflow as tf

import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras
from keras.models import Model

from tensorflow.keras.layers import Input, LSTM, Dense, RepeatVector,  TimeDistributed, Attention, Concatenate, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import callbacks

import pandas as pd
import matplotlib.pyplot as plt
from darts import TimeSeries
from darts.models import TransformerModel

from darts.dataprocessing.transformers import Scaler
from darts.models import ExponentialSmoothing
from darts.utils.utils import SeasonalityMode

from darts.models import NBEATSModel
from datetime import datetime
from geopy.geocoders import Nominatim

from timezonefinder import TimezoneFinder
import pytz
from astral.sun import sun

from pvlib import solarposition
from astral import LocationInfo
import requests

import calendar
import math

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
        models['tcn'] = TCNModel.load("tcn_future_model.pt")

        # Загрузка модели transformer
        models['transformer'] = TransformerModel.load("darts_future_model.pt")

        # Загрузка TensorFlow модели
        models['tf'] = tf.keras.models.load_model('tf_model.keras')

        models['nbeats'] = NBEATSModel.load('nbeats_model.pt')

        with open('prophet_model.json', 'r') as f:
            models['Prophet'] = model_from_json(f.read())

        models['ets']=ExponentialSmoothing.load('ets_model.pt')
    except Exception as e:
        st.error(f"Ошибка загрузки моделей: {str(e)}")
    return models

@st.cache_resource
def load_temperature_model():
    """Загрузка модели для прогнозирования температуры"""
    models={}
    try:
        # Загрузка Prophet модели для температуры
        with open('temperature_prophet_model.json', 'r') as f:
            models['temp_prophet'] = model_from_json(f.read())

        models['temp_ets']=ExponentialSmoothing.load('temperature_ets_model.pt')

        models['temp_nbeats'] = NBEATSModel.load('temp_nbeats_model.pt')
        return models
    except Exception as e:
        st.error(f"Ошибка загрузки температурной модели: {str(e)}")
        return None

def validate_dataset(df):
    """Проверка данных на отрицательные значения"""
    invalid_condition = (df['SumRad'] < 0) | (df['SumRad'] > 1000)
    if invalid_condition.any():
        invalid_rows = df[invalid_condition]
        return False, invalid_rows
    return True, None

def load_default_data():
    """Загрузка встроенного набора данных"""
    df = pd.read_csv('SunTable.csv', index_col=False)
    df.drop(df[df.isnull().N == True].index, inplace=True)
    df['days'] = df.apply(lambda x: x.N // 24, axis=1)
    df['months'] = df.apply(lambda x: x.N // (30 * 24), axis=1)
    df['years'] = df.apply(lambda x: x.N // (365 * 24), axis=1)
    df.drop('N', axis=1, inplace=True)

    df = df.query("SumRad != 0")
    df = df.reset_index(drop=True)

    DataDays = df.groupby('days').agg(
        {'V': 'mean', 'T': 'mean', 'P': 'mean', 'DirectRad': 'sum', 'ScatterRad': 'sum', 'SumRad': 'sum'})

    DataDays = DataDays.reset_index()
    start_date = "2014-01-01"
    X_Prophet = DataDays[['days','T','SumRad']]

    X_Prophet['days'] = pd.to_datetime(start_date) + pd.to_timedelta(X_Prophet["days"], unit="D")
    X_Prophet.rename(columns={"days": "ds","T" :"T" ,"SumRad": "y"}, inplace=True)
    df1 = X_Prophet
    df1 = df1[:-1]
    return df1

def prepare_data(df):
    """Подготовка данных для прогнозирования"""
    df.drop(df[df.isnull().N == True].index, inplace=True)
    df['days'] = df.apply(lambda x: x.N // 24, axis=1)
    df['months'] = df.apply(lambda x: x.N // (30 * 24), axis=1)
    df['years'] = df.apply(lambda x: x.N // (365 * 24), axis=1)
    df.drop('N', axis=1, inplace=True)

    df = df.query("SumRad != 0")
    df = df.reset_index(drop=True)

    DataDays = df.groupby('days').agg(
        {'V': 'mean', 'T': 'mean', 'P': 'mean', 'DirectRad': 'sum', 'ScatterRad': 'sum', 'SumRad': 'sum'})
    DataMonths = df.groupby('months').agg(
        {'V': 'mean', 'T': 'mean', 'P': 'mean', 'DirectRad': 'sum', 'ScatterRad': 'sum', 'SumRad': 'sum'})
    DataYears = df.groupby('years').agg(
        {'V': 'mean', 'T': 'mean', 'P': 'mean', 'DirectRad': 'sum', 'ScatterRad': 'sum', 'SumRad': 'sum'})

    DataDays = DataDays.reset_index()
    start_date = "2014-01-01"
    X_Prophet = DataDays[['days', 'T', 'SumRad']]

    X_Prophet['days'] = pd.to_datetime(start_date) + pd.to_timedelta(X_Prophet["days"], unit="D")
    X_Prophet.rename(columns={"days": "ds", "T": "T", "SumRad": "y"}, inplace=True)
    df1 = X_Prophet
    df1 = df1[:-1]
    return df1

def train_models(train_data):
    """Обучение всех моделей"""
    models = {}
    train_data_rad = train_data[['ds','y']]

    # TCN
    series = TimeSeries.from_dataframe(train_data_rad, 'ds', 'y')

    # 3. Обработка данных
    scaler = Scaler()
    scaled_series = scaler.fit_transform(series)

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

    model.fit(scaled_series, epochs=20, verbose=True)
    models['transformer'] = model

    modelP = Prophet(
        yearly_seasonality=True,  # Включить годовую сезонность
        weekly_seasonality=False,  # Отключить, если данные не недельные
        daily_seasonality=False,
        seasonality_mode='additive',  # Для растущих трендов
        changepoint_prior_scale=0.05  # Сглаживание резких изменений тренда
    )

    prophet_model = modelP
    prophet_model.fit(train_data_rad)
    models['prophet'] = prophet_model

    # 3. Создание и обучение ETS
    model_ets = ExponentialSmoothing(
        trend=SeasonalityMode.ADDITIVE,  # Вместо "add"
        seasonal=SeasonalityMode.ADDITIVE,
        seasonal_periods=365,
        damped=True,
        random_state=42
    )

    # Для данных с частотой (если индекс не задан явно)
    model_ets.fit(scaled_series)
    models['ets'] = model_ets

    model_nbeats = NBEATSModel(
        input_chunk_length=365,  # Длина входного окна
        output_chunk_length=365,  # Длина прогноза
        generic_architecture=True,  # Универсальный режим
        num_stacks=10,  # Количество стеков
        num_blocks=3,  # Блоков в стеке
        num_layers=4,  # Слоев в блоке
        dropout=0.1,
        random_state=42,
        pl_trainer_kwargs={"accelerator": "cpu"}
    )

    model_nbeats.fit(
        scaled_series,
        epochs=30,
        verbose=True
    )

    models['nbeats'] = model_nbeats

    model_tcn = TCNModel(
        input_chunk_length=730,
        output_chunk_length=365,
        batch_size=32,
        pl_trainer_kwargs={"accelerator": "cpu"}  # Для GPU замените на "gpu"
    )

    model_tcn.fit(scaled_series, epochs=30, verbose=True)

    models['tcn'] = model_tcn

    # tensorflow
    scaler = MinMaxScaler(feature_range=(0, 1))
    train_data_rad['y'] = scaler.fit_transform(train_data_rad[['y']])

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

    X, y = create_dataset_from_df(train_data_rad, window_size, forecast_horizon)

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

def fast_train_models(train_data):
    """Обучение всех моделей"""
    models = {}
    train_data_rad = train_data[['ds','y']]

    # TCN
    series = TimeSeries.from_dataframe(train_data_rad, 'ds', 'y')

    # 3. Обработка данных
    scaler = Scaler()
    scaled_series = scaler.fit_transform(series)

    modelP = Prophet(
        yearly_seasonality=True,  # Включить годовую сезонность
        weekly_seasonality=False,  # Отключить, если данные не недельные
        daily_seasonality=False,
        seasonality_mode='additive',  # Для растущих трендов
        changepoint_prior_scale=0.05  # Сглаживание резких изменений тренда
    )

    prophet_model = modelP
    prophet_model.fit(train_data_rad)
    models['Prophet'] = prophet_model

    # 3. Создание и обучение ETS
    model_ets = ExponentialSmoothing(
        trend=SeasonalityMode.ADDITIVE,  # Вместо "add"
        seasonal=SeasonalityMode.ADDITIVE,
        seasonal_periods=365,
        damped=True,
        random_state=42
    )

    # Для данных с частотой (если индекс не задан явно)
    model_ets.fit(scaled_series)
    models['ets'] = model_ets

    model_tcn = TCNModel(
        input_chunk_length=730,
        output_chunk_length=365,
        batch_size=32,
        pl_trainer_kwargs={"accelerator": "cpu"}  # Для GPU замените на "gpu"
    )

    model_tcn.fit(scaled_series, epochs=30, verbose=True)

    models['tcn'] = model_tcn

    return models

def train_temp_models(train_data):
    models={}

    train_data_temp = train_data[['ds', 'T']]
    train_data_temp.rename(columns={"days": "ds","T" :"y"}, inplace=True)
    # Prophet
    modelP = Prophet(
        yearly_seasonality=True,  # Включить годовую сезонность
        weekly_seasonality=False,  # Отключить, если данные не недельные
        daily_seasonality=False,
        seasonality_mode='additive',  # Для растущих трендов
        changepoint_prior_scale=0.05  # Сглаживание резких изменений тренда
    )

    prophet_model = modelP
    prophet_model.fit(train_data_temp)
    models['temp_prophet'] = prophet_model

    # 2. ETS
    series = TimeSeries.from_dataframe(train_data_temp, 'ds', 'y')

    scaler = Scaler()
    scaled_series = scaler.fit_transform(series)

    # 3. Создание и обучение ETS
    model_ets = ExponentialSmoothing(
        trend=SeasonalityMode.ADDITIVE,  # Вместо "add"
        seasonal=SeasonalityMode.ADDITIVE,
        seasonal_periods=365,
        damped=True,
        random_state=42
    )

    # Для данных с частотой (если индекс не задан явно)
    model_ets.fit(scaled_series)
    models['temp_ets']=model_ets

    model_nbeats = NBEATSModel(
        input_chunk_length=365,  # Длина входного окна
        output_chunk_length=365,  # Длина прогноза
        generic_architecture=True,  # Универсальный режим
        num_stacks=10,  # Количество стеков
        num_blocks=3,  # Блоков в стеке
        num_layers=4,  # Слоев в блоке
        dropout=0.1,
        random_state=42,
        pl_trainer_kwargs={"accelerator": "cpu"}
    )

    model_nbeats.fit(
        scaled_series,
        epochs=30,
        verbose=True
    )

    models['temp_nbeats'] = model_nbeats
    return models

def fast_train_temp_models(train_data):
    models={}

    train_data_temp = train_data[['ds', 'T']]
    train_data_temp.rename(columns={"days": "ds","T" :"y"}, inplace=True)
    # Prophet
    modelP = Prophet(
        yearly_seasonality=True,  # Включить годовую сезонность
        weekly_seasonality=False,  # Отключить, если данные не недельные
        daily_seasonality=False,
        seasonality_mode='additive',  # Для растущих трендов
        changepoint_prior_scale=0.05  # Сглаживание резких изменений тренда
    )

    prophet_model = modelP
    prophet_model.fit(train_data_temp)
    models['temp_prophet'] = prophet_model

    # 2. ETS
    series = TimeSeries.from_dataframe(train_data_temp, 'ds', 'y')

    scaler = Scaler()
    scaled_series = scaler.fit_transform(series)

    # 3. Создание и обучение ETS
    model_ets = ExponentialSmoothing(
        trend=SeasonalityMode.ADDITIVE,  # Вместо "add"
        seasonal=SeasonalityMode.ADDITIVE,
        seasonal_periods=365,
        damped=True,
        random_state=42
    )

    # Для данных с частотой (если индекс не задан явно)
    model_ets.fit(scaled_series)
    models['temp_ets']=model_ets
    return models

def make_predictions(models, data, model_type):
    """Создание прогнозов"""
    try:
        if model_type == 'tcn':
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

        elif model_type == 'transformer':

            series = TimeSeries.from_dataframe(data, 'ds', 'y')

            scaler = Scaler()
            train_scaled = scaler.fit_transform(series)

            pred_scaled = models['transformer'].predict(n=365)
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

        elif model_type == 'nbeats':
            series = TimeSeries.from_dataframe(data, 'ds', 'y')

            scaler = Scaler()
            train_scaled = scaler.fit_transform(series)

            pred_scaled = models['nbeats'].predict(n=365)
            pred = scaler.inverse_transform(pred_scaled)

            dates = pred.time_index
            values = pred.values()

            pred_df = pd.DataFrame({"ds": dates, "y": values.flatten()})
            pred_df = pred_df.set_index('ds')
            return pred_df
        elif model_type == 'Prophet':
            future = models['Prophet'].make_future_dataframe(periods=365)
            forecast = models['Prophet'].predict(future)[['ds', 'yhat']]
            print(forecast)
            return forecast[['ds', 'yhat']].rename(columns={'yhat': 'y'}).set_index('ds').tail(365)
        elif model_type == 'ets':
            series = TimeSeries.from_dataframe(data, 'ds', 'y')

            scaler = Scaler()
            train_scaled = scaler.fit_transform(series)

            pred_scaled = models['ets'].predict(n=365)
            pred = scaler.inverse_transform(pred_scaled)

            dates = pred.time_index
            values = pred.values()

            pred_df = pd.DataFrame({"ds": dates, "y": values.flatten()})
            pred_df = pred_df.set_index('ds')
            return pred_df
    except Exception as e:
        st.error(f"Ошибка прогнозирования: {str(e)}")
    return None

def predict_temperature(models, data, model_type):
    """Прогнозирование температуры на год вперед"""
    train_data_rad = data[['ds', 'T']]
    train_data_rad.rename(columns={"days": "ds", "T": "y"}, inplace=True)
    try:
        # Прогноз с помощью Prophet
        if model_type == 'temp_prophet':
            future = models['temp_prophet'].make_future_dataframe(periods=365)
            forecast = models['temp_prophet'].predict(future)[['ds', 'yhat']]
            print(forecast)
            return forecast[['ds', 'yhat']].rename(columns={'yhat': 'y'}).set_index('ds').tail(365)
        elif model_type == 'temp_ets':
            series = TimeSeries.from_dataframe(train_data_rad, 'ds', 'y')

            scaler = Scaler()
            train_scaled = scaler.fit_transform(series)

            pred_scaled = models['temp_ets'].predict(n=365)
            pred = scaler.inverse_transform(pred_scaled)

            dates = pred.time_index
            values = pred.values()

            pred_df = pd.DataFrame({"ds": dates, "y": values.flatten()})
            pred_df = pred_df.set_index('ds')
            return pred_df
        elif model_type == 'temp_nbeats':
            series = TimeSeries.from_dataframe(train_data_rad, 'ds', 'y')

            scaler = Scaler()
            train_scaled = scaler.fit_transform(series)

            pred_scaled = models['temp_nbeats'].predict(n=365)
            pred = scaler.inverse_transform(pred_scaled)

            dates = pred.time_index
            values = pred.values()

            pred_df = pd.DataFrame({"ds": dates, "y": values.flatten()})
            pred_df = pred_df.set_index('ds')
            return pred_df
    except Exception as e:
        st.error(f"Ошибка прогнозирования температуры: {str(e)}")
        return None

def handle_prediction():
    st.header("Прогнозирование выработки энергии")

    if 'lat' not in st.session_state or 'lon' not in st.session_state:
        st.session_state.lat = 51.52  # Широта Байкальска
        st.session_state.lon = 104.14  # Долгота Байкальска
        st.info("Используются координаты Байкальска по умолчанию")

        # Блок 1: Параметры панели
    with st.sidebar.expander("Технические характеристики"):
        panel_params = {
            'A': st.number_input(
                "Площадь панели (м²)",
                value=1.65,
                min_value=0.1,
                max_value=10.0
            ),
            'eta_nom': st.number_input(
                "Номинальный КПД (%)",
                min_value=5,
                max_value=40,
                value=15
            ) / 100
        }

        # Блок 2: Параметры ориентации
    with st.sidebar.expander("Ориентация панели"):
        orientation_params = {
            'tilt': st.number_input(
                "Угол наклона (°)",
                min_value=0,
                max_value=90,
                value=30,
                help="0° - горизонтально, 90° - вертикально"
            ),
            'azimuth': st.number_input(
                "Азимут направления (°)",
                min_value=0,
                max_value=360,
                value=180,
                help="0° - Север, 90° - Восток, 180° - Юг, 270° - Запад"
            )
        }
    with st.sidebar.expander("Гибридная модель"):
        prediction_type = st.radio(
            "Тип прогноза",
            ['Только исторические данные (ML)', 'Только физическая модель', 'Гибридный прогноз'],
            index=2
        )

        if prediction_type == 'Гибридный прогноз':
            hybrid_weight = st.slider(
                "Вес ML-модели в гибриде",
                min_value=0.0,
                max_value=1.0,
                value=0.5,
                help="0 = только физическая модель, 1 = только ML"
            )

        forecast_year = st.number_input(
            "Год прогноза",
            min_value=2023,
            max_value=2100,
            value=datetime.now().year + 1
        )

        # Блок 3: Рекомендации по установке
    show_panel_recommendations()

    # Загрузка предобученных моделей
    models = load_pretrained_models()
    temp_models = load_temperature_model()

    data = load_default_data()
    # Основная логика прогнозирования
    if st.button("Сделать прогноз"):
        st.header("Прогнозирование на основе имеющихся данных")
        with st.spinner('Идет прогнозирование...'):
            sensor_params = {
                'sensor_tilt': st.session_state.lat - 23.45,  # Угол для летнего солнцестояния
                'sensor_azimuth': 180  # Южное направление
            }

            # Объединяем все параметры
            all_params = {**panel_params, **orientation_params, **sensor_params}

            # 1. Прогноз солнечной радиации
            rad_forecasts = {}
            for model_type in ['tcn', 'transformer', 'tf', 'nbeats','ets','Prophet']:
                forecast = make_predictions(models, data, model_type)
                rad_forecasts[model_type] = forecast['y']

            # Среднее значение радиации
            rad_combined = pd.concat(rad_forecasts.values(), axis=1)
            rad_combined.columns = [f'rad_{col}' for col in rad_forecasts.keys()]
            rad_combined['mean_rad'] = rad_combined.mean(axis=1)

            # 2. Прогноз температуры

            temp_forecasts = {}
            for model_type in ['temp_prophet', 'temp_ets', 'temp_nbeats']:
                forecast = predict_temperature(temp_models, data, model_type)
                temp_forecasts[model_type] = forecast['y']

            # Среднее значение радиации
            temp_combined = pd.concat(temp_forecasts.values(), axis=1)
            temp_combined.columns = [f'temp_{col}' for col in temp_forecasts.keys()]
            temp_combined['mean_temp'] = temp_combined.mean(axis=1)

            if temp_combined is None:
                st.error("Ошибка прогноза температуры")
                return

            hybrid_model = HybridSolarModel(st.session_state.lat, st.session_state.lon)

            if prediction_type == 'Только исторические данные (ML)':
                # Используем только ML-прогноз
                final_rad = rad_combined[['mean_rad']]
                final_temp = temp_combined[['mean_temp']]
                final_df = calculate_energy(final_rad, final_temp, all_params)
                model_source = "ML-прогноз"

            elif prediction_type == 'Только физическая модель':
                # Используем только физическую модель
                solar_data = hybrid_model.solar_model.get_solar_data(forecast_year)
                final_rad = pd.DataFrame({'mean_rad': solar_data['solar_df']['H']})
                final_temp = pd.DataFrame({'mean_temp': solar_data['solar_df']['temp']})
                final_df = calculate_energy(final_rad, final_temp, all_params)
                model_source = "Физическая модель"

            else:  # Гибридный прогноз
                # Получаем гибридный прогноз
                hybrid_df = hybrid_model.get_hybrid_forecast(
                    forecast_year,
                    rad_combined[['mean_rad']],
                    temp_combined[['mean_temp']],
                    hybrid_weight
                )

                final_rad = pd.DataFrame({'mean_rad': hybrid_df['hybrid_rad']})
                final_temp = pd.DataFrame({'mean_temp': hybrid_df['hybrid_temp']})
                final_df = calculate_energy(final_rad, final_temp, all_params)
                model_source = "Гибридный прогноз"

            # 4. Построение графиков
            fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 12))

            # Строка-образец (например, 'row1')
            sample_row = final_df.loc['2026-12-27'].to_numpy()  # или .values

            # Приравниваем строки 'row2' и 'row3' к строке 'row1'
            final_df.loc[['2026-12-28', '2026-12-29', '2026-12-30', '2026-12-31']] = sample_row
            # График солнечной радиации
            if 'physical_rad' in final_df and 'mean_rad' in final_df:
                final_df['physical_rad'].plot(ax=ax1, color='red', alpha=0.7, label='Физическая модель')
                final_df['mean_rad'].plot(ax=ax1, color='blue', alpha=0.7, label='ML-прогноз')
                final_df['mean_rad'].plot(ax=ax1, color='orange', label='Использованная радиация')
            else:
                final_df['mean_rad'].plot(ax=ax1, color='orange', label='Солнечная радиация')
            ax1.set_title(f'Прогноз солнечной радиации ({model_source})')
            ax1.set_ylabel('кВт·ч/м²')
            ax1.legend()
            ax1.grid(True, linestyle='--', alpha=0.3)

            # График температуры
            if 'physical_temp' in final_df and 'mean_temp' in final_df:
                final_df['physical_temp'].plot(ax=ax2, color='red', alpha=0.7, label='Физическая модель')
                final_df['mean_temp'].plot(ax=ax2, color='blue', alpha=0.7, label='ML-прогноз')
                final_df['mean_temp'].plot(ax=ax2, color='purple', label='Использованная температура')
            else:
                final_df['mean_temp'].plot(ax=ax2, color='purple', label='Температура воздуха')
            ax2.set_title('Прогноз температуры')
            ax2.set_ylabel('°C')
            ax2.legend()
            ax2.grid(True, linestyle='--', alpha=0.3)

            # График энергии
            final_df['energy'].plot(ax=ax3, color='green', label='Суточная выработка')
            final_df['energy'].cumsum().plot(
                ax=ax3, color='blue', secondary_y=True,
                label='Накопленная энергия', linestyle='--')
            ax3.set_title('Выработка энергии')
            ax3.set_ylabel('кВт·ч (суточная)')
            ax3.right_ax.set_ylabel('кВт·ч (накопленная)')
            ax3.legend(loc='upper left')
            ax3.right_ax.legend(loc='upper right')
            ax3.grid(True, linestyle='--', alpha=0.3)

            plt.tight_layout()
            st.pyplot(fig)

            # Анализ эффективности
            st.subheader("Анализ эффективности")
            total_energy = final_df['energy'].sum()
            st.metric("Суммарная годовая выработка", f"{total_energy:.2f} кВт·ч")

            if model_type == 'Гибридный прогноз':
                physical_energy = calculate_energy(
                    pd.DataFrame({'mean_rad': final_df['physical_rad']}),
                    pd.DataFrame({'mean_temp': final_df['physical_temp']}),
                    all_params
                )['energy'].sum()

                ml_energy = calculate_energy(
                    pd.DataFrame({'mean_rad': final_df['mean_rad']}),
                    pd.DataFrame({'mean_temp': final_df['mean_temp']}),
                    all_params
                )['energy'].sum()

                col1, col2, col3 = st.columns(3)
                col1.metric("Гибридная модель", f"{total_energy:.2f} кВт·ч")
                col2.metric("Физическая модель", f"{physical_energy:.2f} кВт·ч",
                            f"{(total_energy - physical_energy):.2f} кВт·ч")
                col3.metric("ML-модель", f"{ml_energy:.2f} кВт·ч",
                            f"{(total_energy - ml_energy):.2f} кВт·ч")

            # Экспорт результатов
            st.download_button(
                label="Скачать полные данные",
                data=final_df.reset_index().to_csv(index=False),
                file_name=f'solar_forecast_{forecast_year}.csv',
                mime='text/csv'
            )

def handle_training():
    st.header("Обучение моделей")

    training_mode = st.sidebar.radio("Режим обучения:", ['Быстрое обучение', 'Полное обучение'])

    # Секция параметров панели
    st.sidebar.header("Параметры солнечной панели")
    use_custom_panel = st.sidebar.checkbox("Использовать свои параметры панели")

    panel_params = {}
    if use_custom_panel:
        panel_params['A'] = st.sidebar.number_input(
            "Площадь панели (м²)",
            value=1.65,
            min_value=0.1,
            max_value=10.0,
            help="Пример: 1.65 м² для панели 250 Вт"
        )

        panel_params['eta_nom'] = st.sidebar.slider(
            "Номинальный КПД (%)",
            min_value=5,
            max_value=40,
            value=15,
            help="Стандартные значения: 15-22%"
        ) / 100

        panel_params['beta'] = -abs(st.sidebar.number_input(
            "Температурный коэффициент мощности (%/°C)",
            value=0.41,
            min_value=0.0,
            max_value=1.0,
            step=0.01,
            help="Для кремния: 0.3-0.5%/°C. Вводите положительное значение!"
        )) / 100

    # Секция загрузки данных
    st.sidebar.header("Настройки данных")
    use_custom_data = st.sidebar.checkbox("Использовать свои данные")

    location_mode = st.radio(
        "Способ задания местоположения:",
        ['По названию города', 'По координатам'],
        horizontal=True
    )

    # Инициализация координат в session_state
    if 'lat' not in st.session_state:
        st.session_state.lat = None
    if 'lon' not in st.session_state:
        st.session_state.lon = None

    if location_mode == 'По названию города':
        city = st.text_input("Введите название города:", value="Байкальск")
        if st.button("Определить координаты"):
            with st.spinner('Поиск координат...'):
                try:
                    geolocator = Nominatim(user_agent="solar_app")
                    location = geolocator.geocode(city)
                    if location:
                        st.session_state.lat = location.latitude
                        st.session_state.lon = location.longitude
                        st.success(f"Координаты найдены: {location.latitude:.4f}, {location.longitude:.4f}")
                    else:
                        st.error("Город не найден")
                except Exception as e:
                    st.error(f"Ошибка геокодинга: {str(e)}")
    else:
        col1, col2 = st.columns(2)
        with col1:
            st.session_state.lat = st.number_input(
                "Широта (°)",
                min_value=-90.0,
                max_value=90.0,
                value=st.session_state.lat
            )
        with col2:
            st.session_state.lon = st.number_input(
                "Долгота (°)",
                min_value=-180.0,
                max_value=180.0,
                value=st.session_state.lon
            )

    st.subheader("Параметры датчика")
    col1, col2 = st.columns(2)
    with col1:
        panel_params['sensor_tilt'] = st.number_input(
            "Угол наклона датчика (°)",
            min_value=0,
            max_value=90,
            value=30,
            help="Угол наклона датчика при сборе данных"
        )
    with col2:
        panel_params['sensor_azimuth'] = st.number_input(
            "Азимут датчика (°)",
            min_value=0,
            max_value=360,
            value=180,
            help="Ориентация датчика при сборе данных"
        )

    st.subheader("Параметры панели")
    col1, col2 = st.columns(2)
    with col1:
        panel_params['panel_tilt'] = st.number_input(
            "Угол наклона панели (°)",
            min_value=0,
            max_value=90,
            value=30,
            help="Угол наклона панели при работе"
        )
    with col2:
        panel_params['panel_azimuth'] = st.number_input(
            "Азимут панели (°)",
            min_value=0,
            max_value=360,
            value=180,
            help="Ориентация панели при работе"
        )


    if use_custom_data:
        uploaded_file = st.sidebar.file_uploader("Загрузите CSV файл", type=['csv'])
        if uploaded_file:
            try:
                df = pd.read_csv(uploaded_file)
                valid, error_data = validate_dataset(df)
                if not valid:
                    st.error("Обнаружены ошибки в данных:")
                    st.write(error_data)
                    return
                data = prepare_data(df)
            except Exception as e:
                st.error(f"Ошибка загрузки данных: {str(e)}")
                return
    else:
        data = load_default_data()

    with st.sidebar.expander("Гибридная модель"):
        model_type = st.radio(
            "Тип прогноза",
            ['Только исторические данные (ML)', 'Только физическая модель', 'Гибридный прогноз'],
            index=2
        )

        if model_type == 'Гибридный прогноз':
            hybrid_weight = st.slider(
                "Вес ML-модели в гибриде",
                min_value=0.0,
                max_value=1.0,
                value=0.5,
                help="0 = только физическая модель, 1 = только ML"
            )

        forecast_year = st.number_input(
            "Год прогноза",
            min_value=2023,
            max_value=2100,
            value=datetime.now().year + 1
        )

    # Основная логика обучения
    if st.button("Начать обучение"):
        if not use_custom_data:
            st.warning("Для обучения моделей необходимо загрузить свои данные")
            return

        with st.spinner('Обучение моделей...'):
            try:
                st.header("Обучение моделей на новых данных")
                all_params = {**panel_params}
                with st.spinner('Обучение моделей...'):
                    try:
                        if training_mode == "Полное обучение":
                            models = train_models(data.reset_index())
                            temp_models = train_temp_models(data.reset_index())
                            st.success("Модели успешно обучены!")

                            with open('new_temperature_prophet_model.json', 'w') as f:
                                f.write(model_to_json(temp_models['temp_prophet']))

                            with open('new_prophet_model.json', 'w') as f:
                                f.write(model_to_json(models['Prophet']))

                            temp_models['temp_ets'].save('new_temp_ets_model.pt')

                            models['ets'].save('new_ets_model.pt')

                            temp_models['temp_nbeats'].save('new_temp_nbeats_model.pt')

                            models['transformer'].save('new_darts_model.pt')

                            models['nbeats'].save('new_nbeats_model.pt')

                            models['tcn'].save('new_tcn_model.pt')

                            models['tf'].save('new_tf_model.keras')
                        else:
                            models = fast_train_models(data.reset_index())
                            temp_models = fast_train_temp_models(data.reset_index())
                            st.success("Модели успешно обучены!")

                            with open('new_temperature_prophet_model.json', 'w') as f:
                                f.write(model_to_json(temp_models['temp_prophet']))

                            with open('new_prophet_model.json', 'w') as f:
                                f.write(model_to_json(models['Prophet']))

                            temp_models['temp_ets'].save('new_temp_ets_model.pt')

                            models['ets'].save('new_ets_model.pt')

                            models['tcn'].save('new_tcn_model.pt')

                        rad_forecasts = {}
                        for model_type in ['tcn', 'transformer', 'tf', 'nbeats','ets','Prophet']:
                            forecast = make_predictions(models, data, model_type)
                            rad_forecasts[model_type] = forecast['y']

                        # Среднее значение радиации
                        rad_combined = pd.concat(rad_forecasts.values(), axis=1)
                        rad_combined.columns = [f'rad_{col}' for col in rad_forecasts.keys()]
                        rad_combined['mean_rad'] = rad_combined.mean(axis=1)

                        # 2. Прогноз температуры

                        temp_forecasts = {}
                        for model_type in ['temp_prophet', 'temp_ets', 'temp_nbeats']:
                            forecast = predict_temperature(temp_models, data, model_type)
                            temp_forecasts[model_type] = forecast['y']

                        # Среднее значение радиации
                        temp_combined = pd.concat(temp_forecasts.values(), axis=1)
                        temp_combined.columns = [f'temp_{col}' for col in temp_forecasts.keys()]
                        temp_combined['mean_temp'] = temp_combined.mean(axis=1)

                        if temp_combined is None:
                            st.error("Ошибка прогноза температуры")
                            return
                        hybrid_model = HybridSolarModel(st.session_state.lat, st.session_state.lon)

                        if model_type == 'Только исторические данные (ML)':
                            # Используем только ML-прогноз
                            final_rad = rad_combined[['mean_rad']]
                            final_temp = temp_combined[['mean_temp']]
                            final_df = calculate_energy(final_rad, final_temp, all_params)
                            model_source = "ML-прогноз"

                        elif model_type == 'Только физическая модель':
                            # Используем только физическую модель
                            solar_data = hybrid_model.solar_model.get_solar_data(forecast_year)
                            final_rad = pd.DataFrame({'mean_rad': solar_data['solar_df']['H']})
                            final_temp = pd.DataFrame({'mean_temp': solar_data['solar_df']['temp']})
                            final_df = calculate_energy(final_rad, final_temp, all_params)
                            model_source = "Физическая модель"

                        else:  # Гибридный прогноз
                            # Получаем гибридный прогноз
                            hybrid_df = hybrid_model.get_hybrid_forecast(
                                forecast_year,
                                rad_combined[['mean_rad']],
                                temp_combined[['mean_temp']],
                                hybrid_weight
                            )

                            final_rad = pd.DataFrame({'mean_rad': hybrid_df['hybrid_rad']})
                            final_temp = pd.DataFrame({'mean_temp': hybrid_df['hybrid_temp']})
                            final_df = calculate_energy(final_rad, final_temp, all_params)

                        # 4. Построение графиков
                        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 12))

                        # График солнечной радиации
                        if 'physical_rad' in final_df and 'mean_rad' in final_df:
                            final_df['physical_rad'].plot(ax=ax1, color='red', alpha=0.7, label='Физическая модель')
                            final_df['mean_rad'].plot(ax=ax1, color='blue', alpha=0.7, label='ML-прогноз')
                            final_df['mean_rad'].plot(ax=ax1, color='orange', label='Использованная радиация')
                        else:
                            final_df['mean_rad'].plot(ax=ax1, color='orange', label='Солнечная радиация')
                        ax1.set_title(f'Прогноз солнечной радиации ({model_source})')
                        ax1.set_ylabel('кВт·ч/м²')
                        ax1.legend()
                        ax1.grid(True, linestyle='--', alpha=0.3)

                        # График температуры
                        if 'physical_temp' in final_df and 'mean_temp' in final_df:
                            final_df['physical_temp'].plot(ax=ax2, color='red', alpha=0.7, label='Физическая модель')
                            final_df['mean_temp'].plot(ax=ax2, color='blue', alpha=0.7, label='ML-прогноз')
                            final_df['mean_temp'].plot(ax=ax2, color='purple', label='Использованная температура')
                        else:
                            final_df['mean_temp'].plot(ax=ax2, color='purple', label='Температура воздуха')
                        ax2.set_title('Прогноз температуры')
                        ax2.set_ylabel('°C')
                        ax2.legend()
                        ax2.grid(True, linestyle='--', alpha=0.3)

                        # График энергии
                        final_df['energy'].plot(ax=ax3, color='green', label='Суточная выработка')
                        final_df['energy'].cumsum().plot(
                            ax=ax3, color='blue', secondary_y=True,
                            label='Накопленная энергия', linestyle='--')
                        ax3.set_title('Выработка энергии')
                        ax3.set_ylabel('кВт·ч (суточная)')
                        ax3.right_ax.set_ylabel('кВт·ч (накопленная)')
                        ax3.legend(loc='upper left')
                        ax3.right_ax.legend(loc='upper right')
                        ax3.grid(True, linestyle='--', alpha=0.3)

                        plt.tight_layout()
                        st.pyplot(fig)

                        # Анализ эффективности
                        st.subheader("Анализ эффективности")
                        total_energy = final_df['energy'].sum()
                        st.metric("Суммарная годовая выработка", f"{total_energy:.2f} кВт·ч")

                        if model_type == 'Гибридный прогноз':
                            physical_energy = calculate_energy(
                                pd.DataFrame({'mean_rad': final_df['physical_rad']}),
                                pd.DataFrame({'mean_temp': final_df['physical_temp']}),
                                all_params
                            )['energy'].sum()

                            ml_energy = calculate_energy(
                                pd.DataFrame({'mean_rad': final_df['mean_rad']}),
                                pd.DataFrame({'mean_temp': final_df['mean_temp']}),
                                all_params
                            )['energy'].sum()

                            col1, col2, col3 = st.columns(3)
                            col1.metric("Гибридная модель", f"{total_energy:.2f} кВт·ч")
                            col2.metric("Физическая модель", f"{physical_energy:.2f} кВт·ч",
                                        f"{(total_energy - physical_energy):.2f} кВт·ч")
                            col3.metric("ML-модель", f"{ml_energy:.2f} кВт·ч",
                                        f"{(total_energy - ml_energy):.2f} кВт·ч")

                        # Экспорт результатов
                        st.download_button(
                            label="Скачать полные данные",
                            data=final_df.reset_index().to_csv(index=False),
                            file_name=f'solar_forecast_{forecast_year}.csv',
                            mime='text/csv'
                        )
                    except Exception as e:
                        st.error(f"Ошибка обучения: {str(e)}")
            except Exception as e:
                st.error(f"Ошибка обучения: {str(e)}")

def calculate_energy(solar_rad, temperature_df , panel_params):
    default_params = {
        'k0': 30.02,
        'k1': 6.28,
        'beta': -0.0041,
        'A': 1.65,
        'eta_nom': 0.153,
        'wind_speed': 1.2,
        'K_L': 0.9,
        'tilt': 30.0,
        'azimuth': 180.0,
        'sensor_tilt': None,
        'sensor_azimuth': 180.0
    }

    # Объединяем параметры
    params = {**default_params, **panel_params}

    # Валидация параметров
    if params['A'] <= 0:
        raise ValueError("Площадь панели должна быть положительной")
    if not (0 < params['eta_nom'] <= 0.4):
        raise ValueError("КПД должен быть в диапазоне 0-40%")

    # Объединение данных
    combined = solar_rad.join(temperature_df, how='inner')
    energy = []

    lat = st.session_state.lat
    lon = st.session_state.lon

    # Параметры датчика
    tilt_d = params['sensor_tilt'] if params['sensor_tilt'] is not None else (lat - 23.45)
    azimuth_d = params['sensor_azimuth']

    for idx, row in combined.iterrows():
        day_of_year = idx.timetuple().tm_yday

        # Расчет склонения солнца
        declination = 23.45 * np.sin(np.radians(360 * (284 + day_of_year) / 365))

        # Расчет зенитного угла в полдень
        zenith = abs(lat - declination)

        # Расчет угла падения с учетом азимута
        # Для датчика
        cos_theta_d = (
                np.cos(np.radians(zenith)) * np.cos(np.radians(tilt_d)) +
                np.sin(np.radians(zenith)) * np.sin(np.radians(tilt_d)) *
                np.cos(np.radians(180 - azimuth_d))
        )

        # Для панели
        cos_theta_p = (
                np.cos(np.radians(zenith)) * np.cos(np.radians(params['tilt'])) +
                np.sin(np.radians(zenith)) * np.sin(np.radians(params['tilt'])) *
                np.cos(np.radians(180 - params['azimuth']))
        )

        # Защита от нереалистичных значений
        cos_theta_d = max(0.1, cos_theta_d)
        cos_theta_p = max(0.1, cos_theta_p)
        print(cos_theta_d, cos_theta_p , cos_theta_p / cos_theta_d)
        # 6. Пересчет радиации
        effective_rad = row['mean_rad'] * (cos_theta_p / cos_theta_d)

        T_pv = row['mean_temp'] + effective_rad /(params['k0'] + params['k1'] * params['wind_speed'])

        # Расчёт КПД
        eta = params['eta_nom'] * (1 + params['beta'] * (T_pv - 48))

        # Расчёт энергии
        energy.append(effective_rad * eta * params['A'] * params['K_L'])
    combined['energy'] = energy

    return combined

def calculate_optimal_angles(lat):
    """Расчёт оптимальных углов наклона"""
    return {
        'static': round(lat, 2),
        'dynamic': {
            'winter': round(lat + 15, 2),
            'spring_autumn': round(lat, 2),
            'summer': round(lat - 15, 2)
        }
    }

def get_solar_noon(lat, lon):
    """Точный расчёт времени солнечного полдня с учётом даты и координат"""
    try:
        # Определение часового пояса
        tf = TimezoneFinder()
        tz_name = tf.timezone_at(lat=lat, lng=lon) or "UTC"
        tz = pytz.timezone(tz_name)

        # Создание объекта LocationInfo
        loc = LocationInfo("custom", "region", tz_name, lat, lon)

        # Расчёт для сегодняшней даты
        s = sun(loc.observer, date=datetime.now(), tzinfo=tz)

        return s["noon"].strftime("%H:%M"), tz_name

    except Exception as e:
        st.error(f"Ошибка расчёта: {str(e)}")
        return "12:00", "UTC"

def handle_panel_positioning():
    st.header("Оптимальное положение солнечных панелей")

    location_mode = st.radio(
        "Способ задания местоположения:",
        ['По названию города', 'По координатам']
    )

    if 'lat' not in st.session_state:
        st.session_state.lat = None
    if 'lon' not in st.session_state:
        st.session_state.lon = None

    if location_mode == 'По названию города':
        city = st.text_input("Введите название города:")
        if city:
            with st.spinner('Поиск координат...'):
                try:
                    geolocator = Nominatim(user_agent="solar_app")
                    location = geolocator.geocode(city)
                    if location:
                        st.session_state.lat = location.latitude
                        st.session_state.lon = location.longitude
                    else:
                        st.error("Город не найден")
                except Exception as e:
                    st.error(f"Ошибка геокодинга: {str(e)}")

    else:
        col1, col2 = st.columns(2)
        with col1:
            st.session_state.lat = st.number_input(
                "Широта (°)",
                min_value=-90.0,
                max_value=90.0,
                value=55.75
            )
        with col2:
            st.session_state.lon = st.number_input(
                "Долгота (°)",
                min_value=-180.0,
                max_value=180.0,
                value=37.61
            )

    if st.session_state.lat and st.session_state.lon:
        show_panel_recommendations()

def show_panel_recommendations():
    lat = st.session_state.lat
    lon = st.session_state.lon

    # Расчёт оптимальных углов
    static_angle = round(lat, 2)
    dynamic_angles = {
        'Зима (дек-мар)': round(lat + 15, 2),
        'Весна/осень': round(lat, 2),
        'Лето (июн-сен)': round(lat - 15, 2)
    }

    # Расчёт времени пиковой радиации
    solar_noon, tz = get_solar_noon(lat, lon)

    # Отображение результатов
    st.subheader("Рекомендации по установке")

    cols = st.columns(3)
    cols[0].metric("Широта", f"{lat:.2f}°")
    cols[1].metric("Долгота", f"{lon:.2f}°")
    cols[2].metric("Пик радиации",
                   f"{solar_noon} ({tz})",
                   "Местное время")

    st.markdown("### Оптимальные углы наклона")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Статичная установка (год без регулировки):**")
        st.markdown(f"```\n{static_angle}°\n```")
        st.markdown("*Среднегодовой оптимальный угол*")

    with col2:
        st.markdown("**Динамическая регулировка (3-4 раза в год):**")
        for period, angle in dynamic_angles.items():
            st.markdown(f"- **{period}**: `{angle}°`")

    st.markdown("---")
    st.map(pd.DataFrame({'lat': [lat], 'lon': [lon]}))

class HybridSolarModel:
    def __init__(self, latitude, longitude):
        self.lat = latitude
        self.lon = longitude
        self.solar_model = SolarRadiationModel(latitude, longitude)

    def get_hybrid_forecast(self, year, ml_rad_forecast, ml_temp_forecast, hybrid_weight=0.5):
        """
        Создает гибридный прогноз, объединяя физическую модель и ML-прогнозы

        Параметры:
        year - год прогноза
        ml_rad_forecast - DataFrame с ML-прогнозом радиации
        ml_temp_forecast - DataFrame с ML-прогнозом температуры
        hybrid_weight - вес ML-прогноза (0-1)
        """
        # Получаем прогноз от физической модели
        solar_data = self.solar_model.get_solar_data(year)
        physical_rad = solar_data['solar_df'][['H']].rename(columns={'H': 'physical_rad'})
        physical_temp = solar_data['solar_df'][['temp']].rename(columns={'temp': 'physical_temp'})

        # Объединяем с ML-прогнозами
        hybrid_df = physical_rad.join(physical_temp, how='left')
        hybrid_df = hybrid_df.join(ml_rad_forecast, how='left')
        hybrid_df = hybrid_df.join(ml_temp_forecast, how='left')

        # Заполняем пропуски средними значениями
        hybrid_df.fillna(hybrid_df.mean(), inplace=True)

        # Гибридный прогноз (взвешенное среднее)
        hybrid_df['hybrid_rad'] = (hybrid_weight * hybrid_df['mean_rad'] +
                                   (1 - hybrid_weight) * hybrid_df['physical_rad'])

        hybrid_df['hybrid_temp'] = (hybrid_weight * hybrid_df['mean_temp'] +
                                    (1 - hybrid_weight) * hybrid_df['physical_temp'])

        return hybrid_df[['hybrid_rad', 'hybrid_temp', 'physical_rad', 'mean_rad',
                          'physical_temp', 'mean_temp']]

class SolarRadiationModel:
    def __init__(self, latitude, longitude):
        self.lat = latitude
        self.lon = longitude
        self.coeffs = {'a': 0.25, 'b': 0.50}

    def fetch_nasa_power_data(self):
        """Получение климатических данных из NASA POWER"""
        url = "https://power.larc.nasa.gov/api/temporal/monthly/point"
        params = {
            'parameters': 'ALLSKY_SFC_SW_DWN,CLRSKY_SFC_SW_DWN,T2M',
            'community': 'RE',
            'longitude': self.lon,
            'latitude': self.lat,
            'format': 'JSON',
            'start': 2010,
            'end': 2020
        }

        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()

            # Извлечение данных
            allsky, clrsky, t2m = [], [], []
            for year in range(2010, 2021):
                for month in range(1, 13):
                    key = f"{year}{month:02d}"
                    allsky.append(data['properties']['parameter']['ALLSKY_SFC_SW_DWN'].get(key, np.nan))
                    clrsky.append(data['properties']['parameter']['CLRSKY_SFC_SW_DWN'].get(key, np.nan))
                    t2m.append(data['properties']['parameter']['T2M'].get(key, np.nan))

            # Среднемесячные значения
            monthly_allsky = [np.nanmean(allsky[i::12]) for i in range(12)]
            monthly_clrsky = [np.nanmean(clrsky[i::12]) for i in range(12)]
            monthly_t2m = [np.nanmean(t2m[i::12]) for i in range(12)]

            return monthly_allsky, monthly_clrsky, monthly_t2m
        except Exception as e:
            st.error(f"Ошибка получения данных NASA: {str(e)}")
            return None, None, None

    def calculate_clearness_index(self, allsky, clrsky):
        return [a / c if c > 0 else 0.5 for a, c in zip(allsky, clrsky)]

    def estimate_sunshine_hours(self, clearness_index):
        sunshine_hours = []
        for month in range(1, 13):
            date = pd.Timestamp(f'2023-{month}-15')
            times = pd.date_range(date, periods=24, freq='H', tz='UTC')

            solpos = solarposition.get_solarposition(times, self.lat, self.lon)
            daylight = solpos[solpos['elevation'] > 0]
            s0 = len(daylight) / 2

            kt = clearness_index[month - 1]
            s_estimated = kt * s0 * 0.85
            sunshine_hours.append(max(0, min(s0, s_estimated)))

        return sunshine_hours

    def calculate_daily_radiation(self, year, s_monthly, t2m_monthly, a=0.25, b=0.50):
        is_leap = calendar.isleap(year)
        days_in_year = 366 if is_leap else 365

        results = []
        monthly_temps = []

        for month in range(1, 13):
            days_in_month = calendar.monthrange(year, month)[1]
            monthly_temps.extend([t2m_monthly[month - 1]] * days_in_month)

        for day in range(1, days_in_year + 1):
            current_date = datetime(year, 1, 1) + pd.Timedelta(days=day - 1)
            month = current_date.month - 1

            # Геометрические параметры
            phi_rad = math.radians(self.lat)
            delta = 23.45 * math.radians(1) * math.sin(2 * math.pi / 365 * (day + 284))

            # Часовой угол восхода
            tan_product = math.tan(phi_rad) * math.tan(delta)
            x = -tan_product

            if x <= -1:
                omega_s = math.pi
            elif x >= 1:
                omega_s = 0
            else:
                omega_s = math.acos(x)

            # Внеатмосферная радиация
            SOLAR_CONSTANT = 1367
            factor1 = (24 * 3600 * SOLAR_CONSTANT) / math.pi
            factor2 = 1 + 0.033 * math.cos(2 * math.pi * day / 365)
            factor3 = (math.cos(phi_rad) * math.cos(delta) * math.sin(omega_s) +
                       omega_s * math.sin(phi_rad) * math.sin(delta))

            H0_J = factor1 * factor2 * max(0, factor3)
            H0_kWh = H0_J / 3600000

            # Максимальная продолжительность дня
            omega_s_deg = math.degrees(omega_s)
            S0 = (2 * omega_s_deg) / 15

            # Фактическая продолжительность солнечного сияния
            S_val = s_monthly[month]

            # Коэффициент пропускания
            if S0 > 0:
                S_ratio = min(S_val / S0, 1.0)
                Kt = a + b * S_ratio
            else:
                Kt = 0

            # Фактическая солнечная радиация
            H = H0_kWh * Kt * 1000

            # Температура
            temp = monthly_temps[day - 1] if day <= len(monthly_temps) else np.nan

            results.append({
                'date': current_date,
                'doy': day,
                'H0': H0_kWh,
                'S0': S0,
                'S': S_val,
                'Kt': Kt,
                'H': H,
                'temp': temp
            })

        return pd.DataFrame(results).set_index('date')

    def get_solar_data(self, year):
        try:
            allsky, clrsky, t2m = self.fetch_nasa_power_data()
            if allsky is None:
                raise ValueError("Не удалось получить данные NASA")

            kt = self.calculate_clearness_index(allsky, clrsky)
            sunshine = self.estimate_sunshine_hours(kt)

            solar_df = self.calculate_daily_radiation(year, sunshine, t2m)
            monthly_totals = solar_df['H'].resample('M').sum()

            return {
                'solar_df': solar_df,
                'monthly_totals': monthly_totals,
                'sunshine_hours': sunshine,
                'coefficients': self.coeffs
            }
        except Exception as e:
            st.error(f"Ошибка в физической модели: {str(e)}")
            sunshine = [3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 7.5, 6.5, 5.5, 4.0, 3.0, 2.5]
            t2m = [-10, -8, -2, 5, 12, 17, 20, 18, 12, 5, -3, -8]  # Типовые температуры
            solar_df = self.calculate_daily_radiation(year, sunshine, t2m)
            monthly_totals = solar_df['H'].resample('M').sum()

            return {
                'solar_df': solar_df,
                'monthly_totals': monthly_totals,
                'sunshine_hours': sunshine,
                'coefficients': {'a': 0.25, 'b': 0.50}
            }

def main():
    st.title("Прогнозирование выработонной энергии солнечной панелью")

    st.sidebar.header("Режим работы")
    mode = st.sidebar.radio("Выберите режим:", ['Прогнозирование', 'Обучение', 'Расположение панели'])

    if mode == 'Расположение панели':
        handle_panel_positioning()

    elif mode == 'Обучение':
        handle_training()

    elif mode == 'Прогнозирование':
        handle_prediction()


if __name__ == "__main__":
    main()