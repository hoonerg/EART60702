import xarray as xr
import numpy as np
import cftime
import os

def open_dataset(file_path):
    dataset = xr.open_dataset(file_path)
    dataset['TREFMXAV_U'] = dataset['TREFMXAV_U'] - 273.15
    dataset['TREFHT'] = dataset['TREFHT'] - 273.15
    return dataset

def process_dataset(dataset, variables_to_include):
    max_values = {var: dataset[var].max(dim=('time', 'lat', 'lon'), skipna=True) for var in variables_to_include}

    normalized_data_max = dataset['TREFMXAV_U'] / max_values['TREFMXAV_U']
    normalized_data = {var: dataset[var] / max_values[var] for var in variables_to_include if var != 'TREFMXAV_U'}

    combined_max = normalized_data_max.stack(sample=('lat', 'lon'))
    combined = xr.concat([normalized_data[var] for var in normalized_data.keys()], dim='variable')
    combined = combined.stack(sample=('lat', 'lon'))

    processed_array_max = combined_max.compute()
    processed_array = combined.compute()

    return processed_array_max, processed_array

def convert_to_numpy(processed_array_max, processed_array):
    np_processed_array = processed_array.values
    np_processed_array_max = processed_array_max.values

    flattened_array = np_processed_array.reshape(processed_array.shape[1], -1)
    reshaped_max = np_processed_array_max.reshape(processed_array_max.shape[0], -1)
    non_nan_columns = ~np.all(np.isnan(reshaped_max), axis=0)
    filtered_max = reshaped_max[:, non_nan_columns]
    filtered_max = np.nan_to_num(filtered_max)

    merged_array = np.concatenate([flattened_array, filtered_max], axis=1)
    return merged_array

def save_data(merged_array, save_directory, file_name):
    np.save(f'{save_directory}/{file_name}', merged_array)

def partition_data(merged_array, indices, save_path):
    for set_name, (start, end) in indices.items():
        subset = merged_array[start:end, :]
        np.save(f'{save_path}/{set_name}.npy', subset)

def merge_and_save_all_train(merged_data_dir, output_file):
    data_3 = np.load('./processed/003/training.npy')
    data_4 = np.load('./processed/004/training.npy')
    data_5 = np.load('./processed/005/training.npy')
    data_6 = np.load('./processed/006/training.npy')
    data_7 = np.load('./processed/007/training.npy')
    data_8 = np.load('./processed/008/training.npy')
    train_concatenated_data = np.concatenate([data_3, data_4, data_5, data_6, data_7, data_8], axis=0)
    print(train_concatenated_data.shape)
    if not os.path.exists(f'{merged_data_dir}'):
        os.makedirs(f'{merged_data_dir}')
    np.save(f'{merged_data_dir}/{output_file}', train_concatenated_data)

def merge_and_save_all_valid(merged_data_dir, output_file):
    data_3 = np.load('./processed/003/validation.npy')
    data_4 = np.load('./processed/004/validation.npy')
    data_5 = np.load('./processed/005/validation.npy')
    data_6 = np.load('./processed/006/validation.npy')
    data_7 = np.load('./processed/007/validation.npy')
    data_8 = np.load('./processed/008/validation.npy')
    valid_concatenated_data = np.concatenate([data_3, data_4, data_5, data_6, data_7, data_8], axis=0)
    print(valid_concatenated_data.shape)
    if not os.path.exists(f'{merged_data_dir}'):
        os.makedirs(f'{merged_data_dir}')
    np.save(f'{merged_data_dir}/{output_file}', valid_concatenated_data)

def merge_and_save_all_test(merged_data_dir, output_file):
    for i in range(1,4):
        data_3 = np.load(f'./processed/003/test_{i}.npy')
        data_4 = np.load(f'./processed/004/test_{i}.npy')
        data_5 = np.load(f'./processed/005/test_{i}.npy')
        data_6 = np.load(f'./processed/006/test_{i}.npy')
        data_7 = np.load(f'./processed/007/test_{i}.npy')
        data_8 = np.load(f'./processed/008/test_{i}.npy')
        test_concatenated_data = np.concatenate([data_3, data_4, data_5, data_6, data_7, data_8], axis=0)
        print(test_concatenated_data.shape)
        file_name = f'test_{i}'
        if not os.path.exists(f'{merged_data_dir}'):
            os.makedirs(f'{merged_data_dir}')
        np.save(f'{merged_data_dir}/{file_name}', test_concatenated_data)

if __name__ == "__main__":

    root_path = '.'
    data_path = './data'
    for number in range(3,9):
        dataset_path = f'00{number}_2006_2080_352_360.nc'
        save_path = f'processed/00{number}'
        if not os.path.exists(os.path.join(root_path, save_path)):
            os.makedirs(os.path.join(root_path, save_path))
        variables_to_include = ['TREFMXAV_U', 'FLNS', 'FSNS', 'PRECT', 'PRSN', 'QBOT', 'TREFHT', 'UBOT', 'VBOT']

        indices = {
            'training': (0, 14598),
            'validation': (14964, 16058),
            'test_1': (16059, 19708),
            'test_2': (19709, 23358),
            'test_3': (23359, 27008)
        }

        dataset = open_dataset(os.path.join(data_path, dataset_path))
        processed_array_max, processed_array = process_dataset(dataset, variables_to_include)
        merged_array = convert_to_numpy(processed_array_max, processed_array)
        save_data(merged_array, os.path.join(root_path, save_path), f'processed_data_{number}.npy')
        print(merged_array.shape)
        partition_data(merged_array, indices, os.path.join(root_path, save_path))
    merge_and_save_all_train('processed/009', 'training.npy')
    merge_and_save_all_valid('processed/009', 'validation.npy')
    merge_and_save_all_test('processed/009', 'test.npy')
