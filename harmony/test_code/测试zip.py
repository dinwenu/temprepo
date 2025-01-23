batch_dnames = [True, False]

bnames = {"is_data" : [True, False], "name" : ["input0", "labels"]}


batch = [1, 2]

for tensor, is_data, name in zip(batch, bnames["is_data"], bnames["name"]):
    if is_data:
        print(tensor, is_data, name)
    else:
        print(tensor, is_data, name)