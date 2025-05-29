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

from pvlib.solarposition import get_solarposition
from astral import LocationInfo

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

def handle_training():
    st.header("Обучение моделей")

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

    # Основная логика обучения
    if st.button("Начать обучение"):
        if not use_custom_data:
            st.warning("Для обучения моделей необходимо загрузить свои данные")
            return

        with st.spinner('Обучение моделей...'):
            try:
                st.header("Обучение моделей на новых данных")
                with st.spinner('Обучение моделей...'):
                    try:
                        models = train_models(data.reset_index())
                        temp_models = train_temp_models(data.reset_index())
                        st.success("Модели успешно обучены!")

                        with open('new_temperature_prophet_model.json', 'w') as f:
                            f.write(model_to_json(temp_models['temp_prophet']))

                        temp_models['temp_ets'].save('new_temp_ets_model.pt')

                        temp_models['temp_nbeats'].save('new_temp_nbeats_model.pt')

                        models['transformer'].save('new_darts_model.pt')

                        models['nbeats'].save('new_nbeats_model.pt')

                        models['tcn'].save('new_tcn_model.pt')

                        models['tf'].save('new_tf_model.keras')

                        rad_forecasts = {}
                        for model_type in ['tcn', 'transformer', 'tf', 'nbeats']:
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

                        # 3. Расчёт энергии
                        final_df = calculate_energy(rad_combined[['mean_rad']], temp_combined[['mean_temp']],
                                                    panel_params)

                        # 4. Построение графиков
                        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 12))

                        # График солнечной радиации
                        rad_combined['mean_rad'].plot(ax=ax1, color='orange', label='Средняя радиация')
                        ax1.set_title('Прогноз солнечной радиации')
                        ax1.set_ylabel('кВт·ч/м²')
                        ax1.legend()

                        # График температуры
                        final_df['mean_temp'].plot(ax=ax2, color='red', label='Температура воздуха')
                        ax2.set_title('Прогноз температуры')
                        ax2.set_ylabel('°C')
                        ax2.legend()

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

                        plt.tight_layout()
                        st.pyplot(fig)

                        # Экспорт результатов
                        st.download_button(
                            label="Скачать полные данные",
                            data=final_df.reset_index().to_csv(index=False),
                            file_name='full_forecast.csv',
                            mime='text/csv'
                        )
                    except Exception as e:
                        st.error(f"Ошибка обучения: {str(e)}")
            except Exception as e:
                st.error(f"Ошибка обучения: {str(e)}")

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
            all_params = {**panel_params, **orientation_params}

            # 1. Прогноз солнечной радиации
            rad_forecasts = {}
            for model_type in ['tcn', 'transformer', 'tf', 'nbeats']:
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

            # 3. Расчёт энергии
            final_df = calculate_energy(rad_combined[['mean_rad']], temp_combined[['mean_temp']], all_params)

            # 4. Построение графиков
            fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 12))

            # График солнечной радиации
            rad_combined['mean_rad'].plot(ax=ax1, color='orange', label='Средняя радиация')
            ax1.set_title('Прогноз солнечной радиации')
            ax1.set_ylabel('кВт·ч/м²')
            ax1.legend()

            # График температуры
            final_df['mean_temp'].plot(ax=ax2, color='red', label='Температура воздуха')
            ax2.set_title('Прогноз температуры')
            ax2.set_ylabel('°C')
            ax2.legend()

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

            plt.tight_layout()
            st.pyplot(fig)

            # Экспорт результатов
            st.download_button(
                label="Скачать полные данные",
                data=final_df.reset_index().to_csv(index=False),
                file_name='full_forecast.csv',
                mime='text/csv'
            )

def calculate_energy(solar_rad, temperature_df , panel_params):

    default_params = {
        'k0': 30.02,
        'k1': 6.28,  #
        'beta': -0.0041,  # Отрицательный коэффициент
        'A': 1.65,  # Площадь панели
        'eta_nom': 0.153,  # Номинальный КПД
        'wind_speed': 1.2,
        'K_L' : 0.9,
        'tilt': 30.0,
        'azimuth': 180.0,        # Фиксированное значение по статье
    }

    # Объединяем параметры (пользовательские имеют приоритет)
    params = {**default_params, **panel_params}

    # Валидация критических параметров
    if params['A'] <= 0:
        raise ValueError("Площадь панели должна быть положительной")
    if not (0 < params['eta_nom'] <= 0.4):
        raise ValueError("КПД должен быть в диапазоне 0-40%")

    # Объединение данных по датам
    combined = solar_rad.join(temperature_df, how='inner')
    print(params)
    # Расчёт энергии для каждой даты
    energy = []

    lat = st.session_state.lat
    lon = st.session_state.lon

    # 1. Фиксированный угол наклона исторического датчика (для летнего солнцестояния)
    declination_summer = 23.45
    tilt_d = lat - declination_summer # Азимут датчика (юг)
    azimuth_d = 180

    for idx, row in combined.iterrows():
        day_of_year = idx.timetuple().tm_yday

        # 1. Расчет склонения солнца
        declination = 23.45 * np.sin(np.radians(360 * (284 + day_of_year) / 365))

        # 2. Расчет зенитного угла в полдень
        zenith = abs(lat - declination)

        # 3. Расчет угла падения для датчика (оптимальный для лета)
        cos_theta_d = np.cos(np.radians(zenith - tilt_d))

        # 4. Расчет угла падения для панели пользователя
        # Упрощенная формула для полудня (солнце на юге)
        cos_theta_p = np.cos(np.radians(zenith - params['tilt']))

        # 5. Защита от нереалистичных значений
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