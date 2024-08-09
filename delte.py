def combine_layers(layers_indexes, layer_names, combo_updates):
    new_indexes = layers_indexes.copy()
    new_names = layer_names.copy()

    # Sort combo_updates by the first index in descending order
    sorted_updates = sorted(combo_updates.items(), key=lambda x: max(x[1]), reverse=True)

    for combo_name, indexes in sorted_updates:
        min_index = min(indexes)
        max_index = max(indexes)

        # Update the indexes
        for i in range(len(new_indexes)):
            if new_indexes[i] == max_index:
                new_indexes[i] = min_index
            elif new_indexes[i] > max_index:
                new_indexes[i] -= 1

        # Update the layer names
        new_names[min_index - 1] = combo_name
        new_names.pop(max_index - 1)

        # Shift the names to maintain the correct order
        new_names.insert(max_index - 2, new_names.pop(min_index))

    return new_indexes, new_names


# Example usage:
layers_indexes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
layer_names = ['layer1', 'layer2', 'layer3', 'layer4', 'layer5', 'layer6', 'layer7', 'layer8', 'layer9', 'layer10']
combo_updates = {'combo_name1': [3, 6], 'combo_name2': [8, 10]}

new_indexes, new_names = combine_layers(layers_indexes, layer_names, combo_updates)

print("Updated indexes:", new_indexes)
print("Updated names:", new_names)