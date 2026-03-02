from pprint import pprint

import matplotlib.pyplot as plt


def display_cf_metrics(results):
    # Extract row + column names
    models = list(results.keys())
    metrics = list(next(iter(results.values())).keys())

    # Build table data (prepend model names as first column)
    cell_data = []
    for model in models:
        row = [model]
        for m in metrics:
            val = results[model][m]
            # Use scientific notation if value < 0.001, else 4 decimals
            if abs(val) < 0.001 and val != 0:
                row.append(f"{val:.2e}")  # scientific notation
            else:
                row.append(f"{val:.4f}")  # 4 decimal places
        cell_data.append(row)

    # Plot
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.axis('off')

    table = ax.table(cellText=cell_data,
                     colLabels=["models"] + metrics,
                     loc='center',
                     cellLoc='center')

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.2)

    plt.show()


'''
multiclass
{'macro_avg': {
               'sensitivity': 0.3668831168831169,
               'specificity': 0.8605981416957027,
               'youden_index': 0.22748125857881957},
 'per_class': {'sensitivity': [0.0,
                               0.3333333333333333,
                               0.2857142857142857,
                               0.8484848484848485],
               'specificity': [1.0,
                               0.9285714285714286,
                               0.7804878048780488,
                               0.7333333333333333],
               'support': [2, 6, 7, 33],
               'youden_index': [0.0,
                                0.26190476190476186,
                                0.06620209059233462,
                                0.5818181818181818]},
 'weighted_avg': {'sensitivity': 0.6666666666666667,
                  'specificity': 0.7757259001161441,
                  'youden_index': 0.4423925667828107}
'''


def display_multiclass_model_metrics(results):
    # Assuming 'results' is a dictionary structured like this:
    model_names = list(results.keys())

    # Step 1: Flatten the data list
    data1 = []
    data2 = []
    for model, m_data in results.items():
        row1 = [model] + [round(val, 4) for val in m_data['macro_avg'].values()]
        row2 = [model] + [round(val, 4) for val in m_data['weighted_avg'].values()]
        data1.append(row1)
        data2.append(row2)

    pprint(data1)
    pprint(data2)

    col_headers1 = ['Model', 'Sensitivity', 'Specificity', 'Youden Index']
    col_headers2 = ['Model', 'Sensitivity', 'Specificity', 'Youden Index']

    # Step 3: Create the figure and axes
    fig1, ax1 = plt.subplots(figsize=(4, len(model_names)))
    ax1.axis('off')

    fig2, ax2 = plt.subplots(figsize=(4, len(model_names)))
    ax2.axis('off')

    # Step 4: Draw the main table with data and headers
    table1 = ax1.table(cellText=data1,
                       colLabels=col_headers1,
                       cellLoc='center',
                       loc='center')

    table2 = ax2.table(cellText=data2,
                       colLabels=col_headers2,
                       cellLoc='center',
                       loc='center')

    num_cols1 = len(col_headers1)
    num_cols2 = len(col_headers2)

    table1.auto_set_column_width(list(range(num_cols1)))
    table1.auto_set_font_size(False)
    table1.set_fontsize(10)

    table2.auto_set_column_width(list(range(num_cols2)))
    table2.auto_set_font_size(False)
    table2.set_fontsize(10)

    fig1.suptitle("Multiclass Classification Macro Average Metrics", fontsize=10, weight='bold', y=0.65)
    fig2.suptitle("Multiclass Classification Weighted Average Metrics", fontsize=10, weight='bold', y=0.65)

    plt.tight_layout()
    plt.show()


def display_multiclass_per_class_metrics(results):
    # Assuming 'results' is a dictionary structured like this:
    model_names = [s.capitalize().replace('_', ' ') for s in list(results.keys())]

    # Step 1: Flatten the data list
    data1 = []
    data2 = []
    for model, m_data in results.items():

        row = []

        row1 = [model] + [round(val, 4) for val in m_data['macro_avg'].values()]
        row2 = [model] + [round(val, 4) for val in m_data['weighted_avg'].values()]
        data1.append(row1)
        data2.append(row2)

    pprint(data1)
    pprint(data2)

    col_headers1 = ['Model', 'Sensitivity', 'Specificity', 'Youden Index']
    col_headers2 = ['Model', 'Sensitivity', 'Specificity', 'Youden Index']

    # Step 3: Create the figure and axes
    fig1, ax1 = plt.subplots(figsize=(4, len(model_names)))
    ax1.axis('off')

    fig2, ax2 = plt.subplots(figsize=(4, len(model_names)))
    ax2.axis('off')

    # Step 4: Draw the main table with data and headers
    table1 = ax1.table(cellText=data1,
                       colLabels=col_headers1,
                       cellLoc='center',
                       loc='center')

    table2 = ax2.table(cellText=data2,
                       colLabels=col_headers2,
                       cellLoc='center',
                       loc='center')

    num_cols1 = len(col_headers1)
    num_cols2 = len(col_headers2)

    table1.auto_set_column_width(list(range(num_cols1)))
    table1.auto_set_font_size(False)
    table1.set_fontsize(10)

    table2.auto_set_column_width(list(range(num_cols2)))
    table2.auto_set_font_size(False)
    table2.set_fontsize(10)

    fig1.suptitle("Multiclass Classification Macro Average Metrics", fontsize=10, weight='bold', y=0.65)
    fig2.suptitle("Multiclass Classification Weighted Average Metrics", fontsize=10, weight='bold', y=0.65)

    plt.tight_layout()
    plt.show()

    '''
binary metrics
    {
        "sensitivity": recall,
        "specificity": specificity,
        "youden_index": youden
    }
'''


def display_binary_model_metrics(results):
    # Assuming 'results' is a dictionary structured like this:
    model_names = list(results.keys())

    # Step 1: Flatten the data list
    data = []
    for model, m_data in results.items():
        row = [model] + [round(val, 4) for val in m_data.values()]
        data.append(row)

    pprint(data)

    col_headers = ['Model', 'Sensitivity', 'Specificity', 'Youden Index']

    # Step 3: Create the figure and axes
    fig, ax = plt.subplots(figsize=(4, len(model_names)))
    ax.axis('off')

    # Step 4: Draw the main table with data and headers
    table = ax.table(cellText=data,
                     colLabels=col_headers,
                     cellLoc='center',
                     loc='center')

    # Step 5: Manually add the top-level headers using add_cell()
    # Get the number of columns to correctly calculate cell widths.
    num_cols = len(col_headers)

    # cell_model = table.add_cell(row=0, col=-1, width=(1/num_cols), height=0.05, text='Model', loc='center')

    # Optional: Enable autosizing
    table.auto_set_column_width(list(range(num_cols)))
    table.auto_set_font_size(False)
    table.set_fontsize(10)

    fig.suptitle("Binary Classification Models Metrics", fontsize=10, weight='bold', y=0.65)

    plt.tight_layout()
    plt.show()
