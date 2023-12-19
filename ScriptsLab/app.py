import base64
from io import BytesIO

from flask import Flask, render_template, request, redirect, send_file, Response, session
import numpy as np
import pandas as pd
from num2words import num2words
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from ScriptsLab.BloomFilter import BloomFilter
from ScriptsLab.DecisionTree import DecisionTree
from ScriptsLab.PairedRegression import PairedRegression

app = Flask(__name__)

def load_data(filename):
    return pd.read_csv(filename)

def throw_error(message):
    return render_template('error_data.html', error_info=message)

@app.route('/')
def index():
    csv_data = load_data('neo.csv')
    return render_template('index.html',
                           max_row=len(csv_data),
                           max_col=len(csv_data.columns),
                           columns=csv_data.columns)

@app.route('/data', methods=['POST'])
def data():
    csv_data = load_data('neo.csv')

    start_row = int(request.form['start_row'])
    end_row = int(request.form['end_row'])
    start_col = int(request.form['start_col'])
    end_col = int(request.form['end_col'])

    if(start_col < 1 or start_col > len(csv_data.columns) or
    end_col < 1 or end_col > len(csv_data.columns) or
    start_row < 1 or start_row > len(csv_data) or
    end_row < 1 or end_row > len(csv_data)):
        return render_template('error_data.html',
                               error_info='Неверные данные')

    selected_data = csv_data.iloc[start_row-1:end_row, start_col-1:end_col]

    column_descriptions = []
    for col in selected_data.columns:
        col_data = selected_data[col]
        col_type = col_data.dtype
        num_empty_cells = col_data.isna().sum()
        num_filled_cells = col_data.count()
        col_description = {
            'name': col,
            'type': col_type,
            'empty_cells': num_empty_cells,
            'filled_cells': num_filled_cells
        }
        column_descriptions.append(col_description)

    dataset_description = {
        'total_rows': len(selected_data),
        'total_cols': len(selected_data.columns),
        'num_empty_cells': selected_data.isna().sum().sum(),
        'num_filled_cells': selected_data.count().sum()
    }

    for col_desc in column_descriptions:
        col_desc['empty_cells_words'] = num2words(col_desc['empty_cells'], lang='ru')
        col_desc['filled_cells_words'] = num2words(col_desc['filled_cells'], lang='ru')

    dataset_description['total_rows_words'] = num2words(dataset_description['total_rows'], lang='ru')
    dataset_description['total_cols_words'] = num2words(dataset_description['total_cols'], lang='ru')
    dataset_description['num_empty_cells_words'] = num2words(dataset_description['num_empty_cells'], lang='ru')
    dataset_description['num_filled_cells_words'] = num2words(dataset_description['num_filled_cells'], lang='ru')

    return render_template(
        'data.html',
        selected_data=selected_data.to_html(classes='table table-secondary table-bordered table-hover'),
        column_descriptions=column_descriptions,
        dataset_description=dataset_description,
        columns=selected_data.columns
    )

@app.route('/diagram_data', methods=['POST'])
def diagram_data():
    csv_data = load_data('neo.csv')
    augmented_data = augment_data(csv_data)

    # Получение выбранного столбца и значения 'True' или 'False'
    selected_column2 = request.form.get('selected_column2')
    hazardous_value2 = request.form.get('hazardous2')

    # Преобразование типа данных в DataFrame
    csv_data['hazardous'] = csv_data['hazardous'].astype(str)
    augmented_data['hazardous'] = augmented_data['hazardous'].astype(str)

    filtered_df = csv_data[csv_data['hazardous'] == hazardous_value2]
    filtered_df2 = augmented_data[augmented_data['hazardous'] == hazardous_value2]

    # Получаем значения min, mean, max из данных
    min_value = filtered_df[selected_column2].min()
    mean_value = filtered_df[selected_column2].mean()
    max_value = filtered_df[selected_column2].max()

    min_value2 = filtered_df2[selected_column2].min()
    mean_value2 = filtered_df2[selected_column2].mean()
    max_value2 = filtered_df2[selected_column2].max()

    plt.figure(figsize=(16, 6))

    # Гистограмма для Original Data
    plt.subplot(1, 2, 1)
    plt.hist(filtered_df[selected_column2].dropna(), bins=20, color='blue', alpha=0.7,
             label=f'{selected_column2} (Original Data)')
    plt.text(0.25, -0.10, f'Min: {min_value:.2f}', color='blue', horizontalalignment='center',
             verticalalignment='center', transform=plt.gca().transAxes)
    plt.text(0.5, -0.10, f'Mean: {mean_value:.2f}', color='green', horizontalalignment='center',
             verticalalignment='center', transform=plt.gca().transAxes)
    plt.text(0.75, -0.10, f'Max: {max_value:.2f}', color='red', horizontalalignment='center', verticalalignment='center',
             transform=plt.gca().transAxes)
    plt.title(f'Distribution of {selected_column2} (Original Data)')
    plt.ylabel('Frequency')
    plt.legend()

    # Гистограмма для Augmented Data
    plt.subplot(1, 2, 2)
    plt.hist(filtered_df2[selected_column2].dropna(), bins=20, color='orange', alpha=0.7,
             label=f'{selected_column2} (Augmented Data)')
    plt.text(0.25, -0.10, f'Min: {min_value2:.2f}', color='blue', horizontalalignment='center',
             verticalalignment='center', transform=plt.gca().transAxes)
    plt.text(0.5, -0.10, f'Mean: {mean_value2:.2f}', color='green', horizontalalignment='center',
             verticalalignment='center', transform=plt.gca().transAxes)
    plt.text(0.75, -0.10, f'Max: {max_value2:.2f}', color='red', horizontalalignment='center',
             verticalalignment='center', transform=plt.gca().transAxes)
    plt.title(f'Distribution of {selected_column2} (Augmented Data)')
    plt.ylabel('Frequency')
    plt.legend()

    plt.show()

    return render_template('index.html',
                           max_row=len(csv_data),
                           max_col=len(csv_data.columns),
                           columns=csv_data.columns)

@app.route('/analysis', methods=['POST'])
def analysys():
    csv_data = load_data('neo.csv')

    selected_column = request.form.get('selected_column')
    hazardous_value = request.form['hazardous']

    if selected_column is None:
        return throw_error('Пустое значение столбца для анализа')

    filtered_df = csv_data[csv_data['hazardous'] == (hazardous_value == 'True')]

    try:
        min_value = filtered_df[selected_column].min()
        mean_value = filtered_df[selected_column].mean()
        max_value = filtered_df[selected_column].max()
    except KeyError:
        return throw_error('Выбранный столбец не существует в данных')

    return render_template('analysis.html',
                           min_value=round(min_value, 5),
                           mean_value=round(mean_value, 5),
                           max_value=round(max_value, 5),
                           column_name=selected_column)

def augment_data(data):
    augmented_data = data.copy()

    # Увеличение размера данных на 10%
    num_rows_to_add = int(len(augmented_data) * 0.1)
    new_rows = pd.DataFrame()

    # Усреднение числовых столбцов
    numeric_columns = augmented_data.select_dtypes(include='number').columns
    for col in numeric_columns:
        mean_value = augmented_data[col].mean()

        new_values = [mean_value] * num_rows_to_add  # Теперь добавляем только усредненные значения

        new_rows[col] = new_values

    # Заполнение текстовых и логических (bool) столбцов
    other_columns = augmented_data.select_dtypes(include=['object', 'bool']).columns
    for col in other_columns:
        if pd.api.types.is_bool_dtype(augmented_data[col]):
            # Для логических (bool) столбцов добавляем случайные значения True/False
            new_values = np.random.choice([True, False], size=num_rows_to_add)
        else:
            # Для текстовых столбцов добавляем наиболее часто встречающееся значение
            most_frequent_value = augmented_data[col].mode().iloc[0]
            new_values = [most_frequent_value] * num_rows_to_add

        new_rows[col] = new_values

    augmented_data = pd.concat([augmented_data, new_rows], ignore_index=False)

    return augmented_data

@app.route('/bloomFilter', methods=['POST'])
def bloomFilter():
    num_hashing = int(request.form['hashNum_textbox'])
    word = str(request.form['words_textbox'])

    url1 = 'https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset '
    url2 = 'https://www.kaggle.com/datasets/sameepvani/nasa-nearest-earth-objects'
    url3 = 'https://www.kaggle.com/datasets/surajjha101/forbes-billionaires-data-preprocessed'

    columns_1 = ['id', 'gender', 'age', 'hypertension', 'heart_desease', 'ever_married', 'work_type', 'Residence_type', 'avg_glucose_level', 'bmi']
    columns_2 = ['id', 'name', 'est_diameter_min', 'est_diameter_max', 'relative_velocity', 'miss_distance', 'orbiting_body', 'sentry_object', 'absolute_magnitude', 'hazardous']
    columns_3 = ['Rank', 'Name', 'Networth', 'Age', 'Country', 'Source', 'Industry']

    bloomFilter1 = BloomFilter(1000, num_hashing)
    bloomFilter2 = BloomFilter(1000, num_hashing)
    bloomFilter3 = BloomFilter(1000, num_hashing)

    bloomFilter1.add_to_filter(columns_1)
    bloomFilter2.add_to_filter(columns_2)
    bloomFilter3.add_to_filter(columns_3)

    IsHas1 = bloomFilter1.check_is_not_in_filter(word)
    IsHas2 = bloomFilter2.check_is_not_in_filter(word)
    IsHas3 = bloomFilter3.check_is_not_in_filter(word)

    paths = []

    if (not IsHas1):
        paths.append(url1)
    if (not IsHas2):
        paths.append(url2)
    if (not IsHas3):
        paths.append(url3)

    if len(paths) == 1:
        return redirect(paths[0])

    return render_template('bloomFilter.html',
                           paths=paths)

@app.route('/regression', methods=['POST'])
def regression():

    # Загрузка данных
    csv_data = load_data('neo.csv')

    big_data = csv_data.sample(frac=0.99)
    small_data = csv_data.drop(big_data.index)

    x_column = 'relative_velocity'
    y_column = 'est_diameter_min'

    # Создание и обучение модели линейной регрессии
    model = PairedRegression(big_data, x_column, y_column)

    plt.figure(figsize=(12, 10))
    plt.subplot(2, 1, 1)
    plt.scatter(big_data[x_column].values, big_data[y_column].values, alpha=0.4)
    plt.plot(big_data[x_column].values, model.predict(big_data[x_column].values), color='red', linewidth=3)
    plt.xlabel('Скорость')
    plt.ylabel('Размер')
    plt.title('Линейная регрессия на 99%')

    plt.subplot(2, 1, 2)
    plt.scatter(small_data[x_column].values, small_data[y_column].values, alpha=0.4)
    plt.plot(small_data[x_column].values, model.predict(small_data[x_column].values), color='red', linewidth=3)
    plt.xlabel('Скорость')
    plt.ylabel('Размер')
    plt.title('Линейная регрессия на 1%')

    # Регулировка интервала между графиками
    plt.subplots_adjust(hspace=0.75)

    img = BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)

    graph_url = base64.b64encode(img.read()).decode()

    return render_template('paired_regression.html',
                           graph_url = graph_url)


@app.route('/decision_tree', methods=['GET'])
def decision_tree():
    # Загрузка данных
    csv_data = load_data('neo.csv')

    df = pd.DataFrame(csv_data)

    # Выбираем нужные признаки (features)
    features = ['relative_velocity', 'est_diameter_min', 'absolute_magnitude']

    # Выбираем данные для обучения
    train_data = df[features]

    train_data = train_data.iloc[:35]

    train_df = train_data.iloc[:25]
    test_df = train_data.iloc[25:35]

    train_array = []
    test_array = []

    for i in range(len(train_df)):
        ind = i
        train_array.append({})
        train_array[i]["relative_velocity"] = train_df["relative_velocity"].iloc[ind]
        train_array[i]["est_diameter_min"] = train_df["est_diameter_min"].iloc[ind]
        train_array[i]["absolute_magnitude"] = train_df["absolute_magnitude"].iloc[ind]

    for i in range(len(test_df)):
        ind = i
        test_array.append({})
        test_array[i]["relative_velocity"] = test_df["relative_velocity"].iloc[ind]
        test_array[i]["est_diameter_min"] = test_df["est_diameter_min"].iloc[ind]
        test_array[i]["absolute_magnitude"] = test_df["absolute_magnitude"].iloc[ind]

    # Задаем дерево решений
    dt = DecisionTree(4, 5, train_array)

    percent = 0

    for elem in test_array:
        res = dt.getAns(elem)

        print("Expected: ", end="")
        print(elem["absolute_magnitude"], end="")
        print("  Res: ", end="")
        print(res)

        currPercent = abs(res - elem["absolute_magnitude"]) / elem["absolute_magnitude"]
        currPercent *= 100
        percent += currPercent

    percent /= 10

    print("Percent: ", percent)
    return "Code executed successfully!"

@app.route('/download', methods=['GET'])
def download_file():
    return send_file('neo.csv', as_attachment=True)


if __name__ == "__main__":
    app.run(debug=True)
