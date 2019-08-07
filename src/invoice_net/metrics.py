import keras.backend as K


def sensitivity(predictions, true_labels, unimportant_label=0):
    mask = true_label != unimportant_label
    return K.mean(predictions[mask] == true_labels[mask])


def specificity(predictions, true_labels, unimportant_label=0):
    mask = true_labels == unimportant_label
    return K.mean(predictions[mask] == true_labels[mask])
