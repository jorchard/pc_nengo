### stimuli_data.py ###
# tools for loading/processing the data found in the Eye_Inhibition_stimuli folder

import numpy as np
from PIL import Image
from copy import deepcopy
import matplotlib.pyplot as plt


#path to folder with all of the images
path = "Eye_Inhibition_stimuli\\"

# the images are labelled according to the facial features the image shows
# this dictionary maps those labels to the features
features = {1: "Cropped eye",
            2: "R. eye",
            3: "R. eye, L. eye",
            4: "R. eye, nose",
            5: "R. eye, mouth",
            6: "R. eye, L. eye, nose",
            7: "R. eye, nose, mouth",
            8: "Full face",
            17: "R. eye, L. eye, mouth"}
# note that the image labels 9 - 16 correspond to vertical reflections of the previous features
# so if the image is labelled i (9,...,16) then it is the reflection of the image labelled i-8
# the exception to this is the image labels 17 and 18, which are reflections of each other


def load_eye_inhibition(feature=8, reflection=None, path=path):
    """ (int, boolean, str) -> (list[Image])
    Loads the face images from the Eye_Inhibition_stimuli folder (directory given by path).
    Loads only the images with the given features.
    The reflection boolean indicates whether to load the left or right images. If it is None, loads both.
    """
    X = []
    
    l_feature = feature
    r_feature = feature + 8

    if l_feature < 10:
        l_label = f"0{l_feature}"
    else:
        l_label = f"{l_feature}"

    if r_feature < 10:
        r_label = f"0{r_feature}"
    else:
        r_label = f"{r_feature}"

    
    for sex_label in ["F", "M"]:
        for idx in range(1, 17):
            if idx < 10:
                idx_label = f"0{idx}"
            else:
                idx_label = str(idx)
            
            if reflection is None:
                img = Image.open(path + f"{sex_label}{idx_label}-{l_label}l.bmp")
                X.append(img)
                img = Image.open(path + f"{sex_label}{idx_label}-{r_label}r.bmp")
                X.append(img)
            elif reflection:
                img = Image.open(path + f"{sex_label}{idx_label}-{r_label}r.bmp")
                X.append(img)
            else:
                img = Image.open(path + f"{sex_label}{idx_label}-{l_label}l.bmp")
                X.append(img)
    return X


def transform_eye_inhibition(X, size=(68, 100)):
    """ (list[Image]) -> (np.array)
    Transforms the images in a list to an easily workable format.
    """
    #resize the image to 68x100
    #convert to numpy array
    #normalize pixel values to [0, 1]
    new_X = [np.asarray(img.resize(size))/255 for img in X]
    return np.array(new_X)


def shuffle_data(X):
    img_height = X.shape[1]
    img_width = X.shape[2]
    S = []
    for i in range(X.shape[0]):
        temp = X[i].flatten()
        np.random.shuffle(temp)
        S.append(np.reshape(temp, (img_height, img_width)))
    return np.array(S)


def shuffle_block(X, block_size=(25, 17)):
    S = []
    image_height, image_width = X[0].shape

    for i in range(X.shape[0]):

        block_positions = []

        for row_idx in range(0, image_height, block_size[0]):
            for col_idx in range(0, image_width, block_size[1]):
                block_positions.append((row_idx, col_idx))

        block_positions_shuffled = np.array(block_positions)

        #shuffle blocks
        np.random.shuffle(block_positions_shuffled)

        shuffled_image = np.zeros((image_height, image_width), dtype=X.dtype)
        for idx, (row_idx, col_idx) in enumerate(block_positions_shuffled):
            block = X[i][row_idx:row_idx+block_size[0], col_idx:col_idx+block_size[1]]
            shuffled_image[block_positions[idx][0]:block_positions[idx][0]+block_size[0], block_positions[idx][1]:block_positions[idx][1]+block_size[1]] = block
        
        S.append(shuffled_image)

    return np.array(S)


if __name__ == "__main__":
    X = transform_eye_inhibition(load_eye_inhibition())
    plt.figure()
    plt.imshow(X[0], cmap="gray")
    plt.show()

    S = shuffle_data(X)
    plt.figure()
    plt.imshow(S[0], cmap="gray")
    plt.show()

    S_block = shuffle_block(X)
    plt.figure()
    plt.imshow(S_block[0], cmap="gray")
    plt.show()

    plt.figure()
    for i in range(X.shape[0]):
        counts, bins = np.histogram(X[i], 256)
        plt.stairs(counts, bins)
    plt.title("Pixel Intensity of Faces")
    plt.show()

    plt.figure()
    for i in range(S.shape[0]):
        counts, bins = np.histogram(S[i], 256)
        plt.stairs(counts, bins)
        plt.title("Pixel Intensity of Shuffled Faces")
    plt.show()

    plt.figure()
    for i in range(S_block.shape[0]):
        counts, bins = np.histogram(S_block[i], 256)
        plt.stairs(counts, bins)
        plt.title("Pixel Intensity of Block-Shuffled Faces")
    plt.show()