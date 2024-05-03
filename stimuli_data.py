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


def transform_eye_inhibition(X):
    """ (list[Image]) -> (np.array)
    Transforms the images in a list to an easily workable format.
    """
    #resize the image to 68x100
    #convert to numpy array
    #normalize pixel values to [0, 1]
    new_X = [np.asarray(img.resize((68, 100)))/255 for img in X]
    new_X = np.array(new_X)
    return np.reshape(new_X, (new_X.shape[0], new_X.shape[1]*new_X.shape[2]))


def shuffle_data(X):
    S = []
    for i in range(X.shape[0]):
        temp = deepcopy(X[i])
        np.random.shuffle(temp)
        S.append(temp)
    return np.array(S)


if __name__ == "__main__":
    #img = Image.open(path+"F01-16r.bmp")
    #img = img.resize((68, 100))

    #d = np.asarray(img)
    #d = (d - np.mean(d))/np.std(d)
    #plt.imshow(d, cmap="gray")

    #counts, bins = np.histogram(d, 256)
    #plt.stairs(counts, bins)

    X = transform_eye_inhibition(load_eye_inhibition())

    plt.imshow(np.reshape(X[0], (100, 68)), cmap="gray")
    print(X.shape)

    plt.figure()
    for i in range(X.shape[0]):
        counts, bins = np.histogram(X[i], 256)
        plt.stairs(counts, bins)
    plt.title("Pixel Intensity of Faces")
    plt.show()

    S = []
    for i in range(X.shape[0]):
        temp = deepcopy(X[i])
        np.random.shuffle(temp)
        S.append(temp)
    S = np.array(S)
    
    plt.figure()
    for i in range(S.shape[0]):
        counts, bins = np.histogram(S[i], 256)
        plt.stairs(counts, bins)
        plt.title("Pixel Intensity of Shuffled Faces")
    plt.show()
    #h = img.histogram()
    #print(h)
    #plt.stairs(h, list(range(len(h)+1)))
    
    #print(d)
    