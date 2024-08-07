import polars as pl                   # Polars for fast dataframe operations
import numpy as np                    # used to convert input into numpy arrays to be fed to the model
import matplotlib.pyplot as plt       # to plot/visualize sales data and sales forecasting
import tensorflow as tf               # acts as the framework upon which this model is built
from tensorflow import keras          # defines layers and functions in the model

# Function to get data from a CSV file and split it into three lists
def get_data(filepath):
    df = pl.read_csv(filepath)
    list_row = df.to_dicts()
    date = df['date'].to_list()
    traffic = df['traffic'].to_list()
    return list_row, date, traffic

# Define your file path here
file_path = 'path/to/Sales_dataset'

# here the csv file has been copied into three lists to allow better availability
list_row, date, traffic = get_data(file_path)

# Dummy data for missing variables
week = {'Monday': 0, 'Tuesday': 1, 'Wednesday': 2, 'Thursday': 3, 'Friday': 4, 'Saturday': 5, 'Sunday': 6}
days = {str(i): i for i in range(1, 32)}
months = {str(i): i for i in range(1, 13)}
years = {str(i): i for i in range(2000, 2025)}
year_all = [2019, 2020, 2021]  # example years
season = {"2021-12-01": "Winter"}  # example season data
holiday = ["2021-12-25", "2021-01-01"]  # example holidays

def number_to_one_hot(week):
    week1 = {k: [1 if i == v else 0 for i in range(7)] for k, v in week.items()}
    return week1

def date_to_enc(d, days, months, years):
    d_split = d.split('/')
    return days[d_split[0]], months[d_split[1]], years[d_split[2]]

def cur_season(season, d):
    # dummy function for current season
    return [0, 0, 1, 0]  # example season encoding

def conversion(week, days, months, years, list_row):
    inp_day, inp_mon, inp_year, inp_week, inp_hol, out = [], [], [], [], [], []
    week1 = number_to_one_hot(week)
    for row in list_row:
        d = row['date']
        d_split = d.split('/')
        if d_split[2] == str(year_all[0]):
            continue
        d1, m1, y1 = date_to_enc(d, days, months, years)
        inp_day.append(d1)
        inp_mon.append(m1)
        inp_year.append(y1)
        week2 = week1[row['day']]
        inp_week.append(week2)
        inp_hol.append([row['holiday']])
        t1 = row['traffic']
        out.append(t1)
    return inp_day, inp_mon, inp_year, inp_week, inp_hol, out

inp_day, inp_mon, inp_year, inp_week, inp_hol, out = conversion(week, days, months, years, list_row)
inp_day = np.array(inp_day)
inp_mon = np.array(inp_mon)
inp_year = np.array(inp_year)
inp_week = np.array(inp_week)
inp_hol = np.array(inp_hol)

def other_inputs(season, list_row):
    inp7, inp_prev, inp_sess = [], [], []
    count = 0
    for row in list_row:
        ind = count
        count += 1
        d = row['date']
        d_split = d.split('/')
        if d_split[2] == str(year_all[0]):
            continue
        sess = cur_season(season, d)
        inp_sess.append(sess)
        t7 = [list_row[ind - j - 1]['traffic'] for j in range(7)]
        t_prev = [list_row[ind - 365]['traffic']]
        inp7.append(t7)
        inp_prev.append(t_prev)
    return inp7, inp_prev, inp_sess

inp7, inp_prev, inp_sess = other_inputs(season, list_row)
inp7 = np.array(inp7).reshape(len(inp7), 7, 1)
inp_prev = np.array(inp_prev)
inp_sess = np.array(inp_sess)

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LSTM, Flatten, concatenate

input_day = Input(shape=(inp_day.shape[1],), name='input_day')
input_mon = Input(shape=(inp_mon.shape[1],), name='input_mon')
input_year = Input(shape=(inp_year.shape[1],), name='input_year')
input_week = Input(shape=(inp_week.shape[1],), name='input_week')
input_hol = Input(shape=(inp_hol.shape[1],), name='input_hol')
input_day7 = Input(shape=(inp7.shape[1], inp7.shape[2]), name='input_day7')
input_day_prev = Input(shape=(inp_prev.shape[1],), name='input_day_prev')
input_day_sess = Input(shape=(inp_sess.shape[1],), name='input_day_sess')

x1 = Dense(5, activation='relu')(input_day)
x2 = Dense(5, activation='relu')(input_mon)
x3 = Dense(5, activation='relu')(input_year)
x4 = Dense(5, activation='relu')(input_week)
x5 = Dense(5, activation='relu')(input_hol)
x_6 = Dense(5, activation='relu')(input_day7)
x__6 = LSTM(5, return_sequences=True)(x_6)
x6 = Flatten()(x__6)
x7 = Dense(5, activation='relu')(input_day_prev)
x8 = Dense(5, activation='relu')(input_day_sess)

c = concatenate([x1, x2, x3, x4, x5, x6, x7, x8])
layer1 = Dense(64, activation='relu')(c)
outputs = Dense(1, activation='sigmoid')(layer1)

model = Model(inputs=[input_day, input_mon, input_year, input_week, input_hol, input_day7, input_day_prev, input_day_sess], outputs=outputs)
model.summary()

model.compile(loss='mean_squared_error', optimizer='adam', metrics=['acc'])

history = model.fit(x=[inp_day, inp_mon, inp_year, inp_week, inp_hol, inp7, inp_prev, inp_sess], y=out, batch_size=16, steps_per_epoch=50, epochs=15, verbose=1, shuffle=False)

def input(date):
    d1, d2, d3 = date_to_enc(date, days, months, years)
    d1 = np.array([d1])
    d2 = np.array([d2])
    d3 = np.array([d3])
    week1 = number_to_one_hot(week)
    week2 = week1[day[date]]
    week2 = np.array([week2])
    h = 1 if date in holiday else 0
    h = np.array([h])
    sess = cur_season(season, date)
    sess = np.array([sess])
    return d1, d2, d3, week2, h, sess

def forecast_testing(date):
    maxj = max(traffic)
    out = []
    count = -1
    ind = 0
    for i in list_row:
        count += 1
        if i['date'] == date:
            ind = count
    t7 = [list_row[ind - j - 1]['traffic'] for j in range(7)]
    result = []
    count = 0
    for i in date_range:  # `date_range` needs to be defined appropriately
        d1, d2, d3, week2, h, sess = input(i)
        t_7 = np.array([t7]).reshape(1, 7, 1)
        t_prev = np.array([[list_row[ind - 365]['traffic']]])
        y_out = model.predict([d1, d2, d3, week2, h, t_7, t_prev, sess])
        result.append(y_out[0][0] * maxj)
        t7.pop(0)
        t7.append(y_out[0][0])
        count += 1
    return result

result = forecast_testing('2021-12-01')
test_sales = [t['traffic'] for t in list_row if '2021-12-01' <= t['date'] <= '2021-12-07']  # example test sales data

plt.plot(result, color='red', label='predicted')
plt.plot(test_sales, color='purple', label="actual")
plt.xlabel("Date")
plt.ylabel("Sales")
plt.legend()
plt.show()
