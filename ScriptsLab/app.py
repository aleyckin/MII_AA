from flask import Flask, render_template, request, redirect, send_file, Response, session
import numpy as np
import pandas as pd
from num2words import num2words
import matplotlib.pyplot as plt
from io import BytesIO
import base64

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
        # Код для добавления случайного шума удален
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

@app.route('/download', methods=['GET'])
def download_file():
    return send_file('neo.csv', as_attachment=True)


if __name__ == "__main__":
    app.run(debug=True)
