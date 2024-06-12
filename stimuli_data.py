### stimuli_data.py ###
# tools for loading/processing the data found in the Eye_Inhibition_stimuli folder

import numpy as np
from PIL import Image
from scipy.ndimage import rotate
import torch
from torch.utils.data import TensorDataset, DataLoader
from copy import deepcopy
import matplotlib.pyplot as plt


#path to folder with all of the images
face_path = "Eye_Inhibition_stimuli/"

# the images are labelled according to the facial features the image shows
# this dictionary maps those labels to the features
face_features_dict = {1: "Cropped Eye",
                    2: "R. Eye",
                    3: "R. Eye, L. Eye",
                    4: "R. Eye, Nose",
                    5: "R. Eye, Mouth",
                    6: "R. Eye, L. Eye, Nose",
                    7: "R. Eye, Nose, Mouth",
                    8: "Full Face",
                    17: "R. Eye, L. Eye, Mouth"}
# note that the image labels 9 - 16 correspond to horizontal reflections of the previous features
# so if the image is labelled i (9,...,16) then it is the reflection of the image labelled i-8
# the exception to this is the image labels 17 and 18, which are reflections of each other
face_features_flip_dict = {1: "Cropped Eye",
                    2: "1 Eye",
                    3: "2 Eyes",
                    4: "1 Eye, Nose",
                    5: "1 Eye, Mouth",
                    6: "2 Eyes, Nose",
                    7: "1 Eye, Nose, Mouth",
                    8: "Full Face",
                    17: "2 Eyes, Mouth"}


def load_eye_inhibition(feature=8, reflection=None, size=(68, 100), face_path=face_path, sex_labels=["F", "M"]):
    """ (int, boolean, tuple[int], str) -> (list[Image])
    Loads the face images from the Eye_Inhibition_stimuli folder (directory given by face_path).
    Loads only the images with the given features.
    The reflection boolean indicates whether to load the left or right images. If it is None, loads both.
    """
    X = []
    
    if feature == 17:
        l_feature = 17
        r_feature = 18
    else:
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

    
    for sex_label in sex_labels:
        for idx in range(1, 17): #face labels, not feature labels
            if idx < 10:
                idx_label = f"0{idx}"
            else:
                idx_label = str(idx)
            
            if reflection is None:
                img = Image.open(face_path + f"{sex_label}{idx_label}-{l_label}l.bmp")
                X.append(img)
                img = Image.open(face_path + f"{sex_label}{idx_label}-{r_label}r.bmp")
                X.append(img)
            elif reflection:
                img = Image.open(face_path + f"{sex_label}{idx_label}-{r_label}r.bmp")
                X.append(img)
            else:
                img = Image.open(face_path + f"{sex_label}{idx_label}-{l_label}l.bmp")
                X.append(img)

    #resize the image to 68x100, convert to numpy array, normalize pixel values to [0, 1]
    new_X = [np.asarray(img.resize(size))/255 for img in X]
    return np.array(new_X)


def vertical_translate(arr, shift):
    """
    Translates the array vertically by the amount shift.
    Fills in the blanks by using the average value.

    If shift is positive we move the image up, if it is negative we move it down.
    """
    ave_val = np.mean(arr)
    #new_arr = np.full_like(arr, ave_val)
    new_arr = np.full_like(arr, 1)

    if shift >= 0:
        new_arr[0:arr.shape[0]-shift,:] = arr[shift:,:]
    else:
        new_arr[-shift:,:] = arr[0:arr.shape[0]+shift,:]
    
    return new_arr



def augment_data(X, num_translations, num_rotations, min_shift=-15, max_shift=30, min_angle=-45, max_angle=45):
    """ (list, int, int, int, int, num, num) -> (np.array)
    Augment the data by rotating and shifting.

    X                   list, the array of images to be augmented,
    num_translations    int, the number of translations to do.
    num_rotations       int, the number of rotations to do.
    min_shift           int, the minimum shift value.
    max_shift           int, the maximum shift value.
    min_angle           num, the minimum rotation angle.
    max_angle           num, the maximum roation angle.
    """
    new_X = []

    for idx in range(X.shape[0]):
        img = X[idx]
        new_X.append(img)

        for trans in range(num_translations):
            shift = np.random.randint(min_shift, max_shift+1)
            temp_img = vertical_translate(img, shift)
            new_X.append(temp_img)
        
        for rot in range(num_rotations):
            angle = np.random.uniform(min_angle, max_angle)
            #temp_img = rotate(img, angle, reshape=False, cval=np.mean(img), mode='constant')
            temp_img = rotate(img, angle, reshape=False, cval=1, mode='constant')
            new_X.append(temp_img)

    return np.array(new_X)




def transform_eye_inhibition(X, size=(68, 100), radius=(2, 3)):
    """ (list[Image], tuple[int], tuple[int]) -> (np.array)
    Transforms the images in a list to an easily workable format.

    X        list[Image], list of images to transform
    size     tuple[int], dimension to resize the image to
    radius   tuple[int], radius around the Fourier transform DC which we use set to 0
    """
    if type(radius) is int:
        radius = (radius, radius)

    #take fourier transforms
    fourier_transforms = [np.fft.fft2(x) for x in X]
    #shift the DC to the center
    fourier_transforms = [np.fft.fftshift(freqs) for freqs in fourier_transforms]

    #where the center is
    center_idx = (size[1]//2, size[0]//2)

    #new_X = []
    for freqs in fourier_transforms: #remove the radius smallest frequencies
        for j in range(radius[0]):
            for i in range(radius[1]):
                #freqs[center_idx[1] + i, center_idx[0] + j] = np.random.randn() + np.random.randn()*1.j
                #freqs[center_idx[1] + i, center_idx[0] - j] = np.random.randn() + np.random.randn()*1.j
                #freqs[center_idx[1] - i, center_idx[0] + j] = np.random.randn() + np.random.randn()*1.j
                #freqs[center_idx[1] - i, center_idx[0] - j] = np.random.randn() + np.random.randn()*1.j
                freqs[center_idx[0] + i, center_idx[1] + j] = 0. + 0.j
                freqs[center_idx[0] + i, center_idx[1] - j] = 0. + 0.j
                freqs[center_idx[0] - i, center_idx[1] + j] = 0. + 0.j
                freqs[center_idx[0] - i, center_idx[1] - j] = 0. + 0.j                                           
    
    return np.real(np.array([np.fft.ifft2(np.fft.ifftshift(freqs)) for freqs in fourier_transforms]))




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


def get_dataloader_features(feature=8, reflection=None, face_path=face_path, 
                            size=(68, 100), radius=(2, 3),
                            num_translations=2, num_rotations=2, min_shift=-15, max_shift=30, min_angle=-45, max_angle=45,
                            batch_size=8, shuffle=True, channels=True):
    """
    Puts the face images. 
    Normal images are labelled 1, shuffled images are labeled 1.

    feautre             int, what facial features are present on the face images (nose, eyes, ears, etc.).
    reflection          bool, whether to include the original faces, their reflections, or both (None).
    face_path           str, path to the directory with all the images.
    size                tuple[int], the size to reshape the image to.
    block_size          tuple[int], size of the blocks to use when shuffling the images.
    num_translations    int, the number of translations to do.
    num_rotations       int, the number of rotations to do.
    min_shift           int, the minimum shift value.
    max_shift           int, the maximum shift value.
    min_angle           num, the minimum rotation angle.
    max_angle           num, the maximum roation angle.
    batch_size          int, the batch size for the DataLoader.
    shuffle             bool, whether or not to shuffle the samples in the dataloader.
    channels            bool, whether or not to reshape the dataset to include a dimension for channels.
    """
    X = load_eye_inhibition(feature=feature, reflection=reflection, size=size, face_path=face_path)
    X = augment_data(X, num_translations=num_translations, num_rotations=num_rotations, 
                     min_shift=min_shift, max_shift=max_shift, min_angle=min_angle, max_angle=max_angle)
    X = transform_eye_inhibition(X, radius=radius)

    if features != 1:
        labels = np.ones(X.shape[0])
    else:
        labels = = np.zeros(X.shape[0])

    if channels: #unsqueeze the tensor to include channel dimension
        dataset = TensorDataset(torch.from_numpy(X).float().unsqueeze(1), torch.from_numpy(labels).float())
    else:
        dataset = TensorDataset(torch.from_numpy(X).float(), torch.from_numpy(labels).float())

    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)



def get_dataloader_block_shuffled(feature=8, reflection=None, face_path=face_path, 
                            size=(68, 100), radius=(2, 3), block_size=(25, 17),
                            num_translations=2, num_rotations=2, min_shift=-15, max_shift=30, min_angle=-45, max_angle=45,
                            batch_size=8, shuffle=True, channels=True):
    
    """
    Puts the face images and their shuffled counterparts into a DataLoader object. 
    Normal images are labelled 1, shuffled images are labeled 1.

    feautre             int, what facial features are present on the face images (nose, eyes, ears, etc.).
    reflection          bool, whether to include the original faces, their reflections, or both (None).
    face_path           str, path to the directory with all the images.
    size                tuple[int], the size to reshape the image to.
    block_size          tuple[int], size of the blocks to use when shuffling the images.
    num_translations    int, the number of translations to do.
    num_rotations       int, the number of rotations to do.
    min_shift           int, the minimum shift value.
    max_shift           int, the maximum shift value.
    min_angle           num, the minimum rotation angle.
    max_angle           num, the maximum roation angle.
    batch_size          int, the batch size for the DataLoader.
    shuffle             bool, whether or not to shuffle the samples in the dataloader.
    channels            bool, whether or not to reshape the dataset to include a dimension for channels.
    """
    X = load_eye_inhibition(feature=feature, reflection=reflection, size=size, face_path=face_path)
    X = augment_data(X, num_translations=num_translations, num_rotations=num_rotations, 
                     min_shift=min_shift, max_shift=max_shift, min_angle=min_angle, max_angle=max_angle)
    X = transform_eye_inhibition(X, radius=radius)
    S = shuffle_block(X, block_size=block_size)

    labels = np.concatenate((np.ones(X.shape[0]), np.zeros(S.shape[0])))
    data = np.concatenate((X, S), axis=0)

    if channels: #unsqueeze the tensor to include channel dimension
        dataset = TensorDataset(torch.from_numpy(data).float().unsqueeze(1), torch.from_numpy(labels).float())
    else:
        dataset = TensorDataset(torch.from_numpy(data).float(), torch.from_numpy(labels).float())

    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def get_dataloader_shuffled(feature=8, reflection=None, face_path=face_path, 
                            size=(68, 100), radius=(2, 3),
                            num_translations=2, num_rotations=2, min_shift=-15, max_shift=30, min_angle=-45, max_angle=45,
                            batch_size=8, shuffle=True, channels=True):
    """
    Puts the face images and their shuffled counterparts into a DataLoader object. 
    Normal images are labelled 1, shuffled images are labeled 1.

    feautre             int, what facial features are present on the face images (nose, eyes, ears, etc.).
    reflection          bool, whether to include the original faces, their reflections, or both (None).
    face_path           str, path to the directory with all the images.
    size                tuple[int], the size to reshape the image to.
    num_translations    int, the number of translations to do.
    num_rotations       int, the number of rotations to do.
    min_shift           int, the minimum shift value.
    max_shift           int, the maximum shift value.
    min_angle           num, the minimum rotation angle.
    max_angle           num, the maximum roation angle.
    batch_size          int, the batch size for the DataLoader.
    shuffle             bool, whether or not to shuffle the samples in the dataloader.
    channels            bool, whether or not to reshape the dataset to include a dimension for channels.
    """
    X = load_eye_inhibition(feature=feature, reflection=reflection, size=size, face_path=face_path)
    X = augment_data(X, num_translations=num_translations, num_rotations=num_rotations, 
                     min_shift=min_shift, max_shift=max_shift, min_angle=min_angle, max_angle=max_angle)
    X = transform_eye_inhibition(X, radius=radius)
    S = shuffle_data(X)

    labels = np.concatenate((np.ones(X.shape[0]), np.zeros(S.shape[0])))
    data = np.concatenate((X, S), axis=0)

    if channels: #unsqueeze the tensor to include channel dimension
        dataset = TensorDataset(torch.from_numpy(data).float().unsqueeze(1), torch.from_numpy(labels).float())
    else:
        dataset = TensorDataset(torch.from_numpy(data).float(), torch.from_numpy(labels).float())

    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)



def load_eye_inhibition_outline(size=(68, 100), face_path=face_path, sex_labels=["F", "M"]):
    """ (int, boolean, tuple[int], str) -> (list[Image])
    Loads the face images from the Eye_Inhibition_stimuli folder (directory given by face_path).
    Loads only the images with the given features.
    The reflection boolean indicates whether to load the left or right images. If it is None, loads both.
    """
    X = []
    
    l_label = f"02"
    
    for sex_label in sex_labels:
        for idx in range(1, 17): #face labels, not feature labels
            if idx < 10:
                idx_label = f"0{idx}"
            else:
                idx_label = str(idx)
            
            img = Image.open(face_path + f"{sex_label}{idx_label}-{l_label}l.bmp")
            X.append(img)

    #resize the image to 68x100, convert to numpy array, normalize pixel values to [0, 1]
    new_X = [np.asarray(img.resize(size))/255 for img in X]
    for arr in new_X:
        arr[:,0:size[0]//2] = arr[:,:size[0]//2-1:-1] #flip image
    return np.array(new_X)

def get_dataloader_outline(reflection=None, face_path=face_path, 
                            size=(68, 100), radius=(2, 3),
                            num_translations=2, num_rotations=2, min_shift=-15, max_shift=30, min_angle=-45, max_angle=45,
                            batch_size=8, shuffle=True, channels=True):
    """
    Puts the face images. 
    Normal images are labelled 1, shuffled images are labeled 1.

    feautre             int, what facial features are present on the face images (nose, eyes, ears, etc.).
    reflection          bool, whether to include the original faces, their reflections, or both (None).
    face_path           str, path to the directory with all the images.
    size                tuple[int], the size to reshape the image to.
    block_size          tuple[int], size of the blocks to use when shuffling the images.
    num_translations    int, the number of translations to do.
    num_rotations       int, the number of rotations to do.
    min_shift           int, the minimum shift value.
    max_shift           int, the maximum shift value.
    min_angle           num, the minimum rotation angle.
    max_angle           num, the maximum roation angle.
    batch_size          int, the batch size for the DataLoader.
    shuffle             bool, whether or not to shuffle the samples in the dataloader.
    channels            bool, whether or not to reshape the dataset to include a dimension for channels.
    """
    img = Image.open(face_path + f"Outline.bmp")
    X = [np.asarray(img.resize(size))/255]
    X = np.array(X)
    X = augment_data(X, num_translations=num_translations, num_rotations=num_rotations, 
                     min_shift=min_shift, max_shift=max_shift, min_angle=min_angle, max_angle=max_angle)
    X = transform_eye_inhibition(X, radius=radius)

    labels = np.ones(X.shape[0])

    if channels: #unsqueeze the tensor to include channel dimension
        dataset = TensorDataset(torch.from_numpy(X).float().unsqueeze(1), torch.from_numpy(labels).float())
    else:
        dataset = TensorDataset(torch.from_numpy(X).float(), torch.from_numpy(labels).float())

    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

if __name__ == "__main__":
    X = transform_eye_inhibition(load_eye_inhibition(feature=17))

    plt.figure()
    plt.imshow(X[0], cmap="gray")
    plt.show()

    plt.figure()
    for i in range(X.shape[0]):
        counts, bins = np.histogram(X[i], 256)
        plt.stairs(counts, bins)
    plt.title("Pixel Intensity of Faces")
    plt.show()
    
    """
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
    """

    dl = get_dataloader_shuffled()


    for batch, (X, y) in enumerate(dl):
        for i in range(len(y)):
            plt.figure()
            plt.imshow(X[i].squeeze().numpy(), cmap='gray')
            plt.title(f"Label: {y[i]}")
            plt.show()