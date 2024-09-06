import json

# def read_json_lines(file_path):
#     data_list = []
#     with open(file_path, 'r') as file:
#         for line in file:
#             data_list.append(json.loads(line))
#     return data_list

# file_path = 'test.json'
# data_list = read_json_lines(file_path)

# print(data_list)

# output_file_path = 'converted_test.json'
# with open(output_file_path, 'w') as output_file:
#     json.dump(data_list, output_file, indent=2)

# print(f"Data has been successfully converted and saved to {output_file_path}")


import pandas as pd


df = pd.read_json('test.json', lines=True)

print(df.columns)

