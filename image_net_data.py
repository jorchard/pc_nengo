### image_net_data.py ###
# tools for loading/processing the data found in the ImageNet folder
# https://www.kaggle.com/datasets/ifigotin/imagenetmini-1000/data

import os
from stimuli_data import *

#path to folder with all of the training images
image_net_path = "imagenet-mini\\train\\"

#path to the .txt files with the label names
label_image_net_path = "words.txt"

#load the label names
labels_names = np.loadtxt(label_image_net_path, delimiter="\t", dtype=str)

#create a dictionary of the label names
labels_dict = {}
for i in range(labels_names.shape[0]):
    labels_dict[labels_names[i][0]] = labels_names[i][1]


def load_imagenet(num_images, classes, image_net_path=image_net_path):
    """
    Loads the images from ImageNet.

    num_images      int, how many images per class to load
    classes         list[str], the labels of the classes to use
    image_net_path  str, the path to the images
    """
    X = []

    for cls in classes:
        directory = image_net_path + cls #path to this classes directory
        image_names = os.listdir(directory) #list of image filenames

        #select random indices
        indices = np.random.choice(np.arange(len(image_names)), min(num_images, len(image_names)), replace=False)

        #open images
        for idx in indices:
            X.append(Image.open(directory + "\\" + image_names[idx]))
    
    return X


def transform_imagenet(X, size=(68, 100), radius=9):
    """ (list[Image], tuple[int], int) -> (np.array)
    Transforms the images in a list to an easily workable format.

    X        list[Image], list of images to transform
    size     tuple[int], dimension to resize the image to
    radius   int, radius around the Fourier transform DC which we use set to 0
    """
    #resize the image to 68x100, grayscale, convert to numpy array, normalize pixel values to [0, 1]
    new_X = [np.asarray(img.resize(size).convert("L"))/255 for img in X]

    #take fourier transforms
    fourier_transforms = [np.fft.fft2(x) for x in new_X]
    #shift the DC to the center
    fourier_transforms = [np.fft.fftshift(freqs) for freqs in fourier_transforms]

    #where the center is
    center_idx = (size[1]//2, size[0]//2)

    #new_X = []
    for freqs in fourier_transforms: #remove the radius smallest frequencies
        for i in range(radius):
            for j in range(radius):
                #freqs[center_idx[1] + i, center_idx[0] + j] = np.random.randn() + np.random.randn()*1.j
                #freqs[center_idx[1] + i, center_idx[0] - j] = np.random.randn() + np.random.randn()*1.j
                #freqs[center_idx[1] - i, center_idx[0] + j] = np.random.randn() + np.random.randn()*1.j
                #freqs[center_idx[1] - i, center_idx[0] - j] = np.random.randn() + np.random.randn()*1.j
                freqs[center_idx[0] + i, center_idx[1] + j] = 0. + 0.j
                freqs[center_idx[0] + i, center_idx[1] - j] = 0. + 0.j
                freqs[center_idx[0] - i, center_idx[1] + j] = 0. + 0.j
                freqs[center_idx[0] - i, center_idx[1] - j] = 0. + 0.j                                           
                

    return np.real(np.array([np.fft.ifft2(np.fft.ifftshift(freqs)) for freqs in fourier_transforms]))



def get_dataloader(feature=8, reflection=None, num_images=10, classes=[], 
                   face_path=face_path, image_net_path=image_net_path, 
                   size=(68, 100), radius=9, 
                   num_translations=2, num_rotations=2, min_shift=-15, max_shift=30, min_angle=-45, max_angle=45,
                   batch_size=8, shuffle=True, channels=True):
    """
    Puts the face images and the imagenet images into a DataLoader object. 
    Face images are labelled 1, imagenet images are labeled 0.
    Returns a training dataloader with augmented images and a testing dataloader with unaugmented images.

    feautre             int, what facial features are present on the face images (nose, eyes, ears, etc.).
    num_images          int, how many images per class to load.
    classes             List[str], the labels of the classes to use.
    reflection          bool, whether to include the original faces, their reflections, or both (None).
    face_path           str, path to the directory with all the face images.
    image_net_path      str, path to the directory with all the imagenet images.
    size                tuple[int], the size to reshape the image to.
    radius              int, radius around the Fourier transform DC which we use set to 0.
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
    #eye inhibition data
    X = load_eye_inhibition(feature=feature, reflection=reflection, size=size, face_path=face_path)
    X_aug = augment_data(X, num_translations=num_translations, num_rotations=num_rotations, 
                     min_shift=min_shift, max_shift=max_shift, min_angle=min_angle, max_angle=max_angle)
    X_aug = transform_eye_inhibition(X_aug, radius=radius)
    X = transform_eye_inhibition(X, radius=radius)

    #imagenet data
    S = load_imagenet(num_images=num_images, classes=classes, image_net_path=image_net_path)
    S = transform_imagenet(S, size=size, radius=radius)

    #training dataloader
    train_labels = np.concatenate((np.ones(X_aug.shape[0]), np.zeros(S.shape[0])))
    train_data = np.concatenate((X_aug, S), axis=0)

    if channels: #unsqueeze the tensor to include channel dimension
        train_dataset = TensorDataset(torch.from_numpy(train_data).float().unsqueeze(1), torch.from_numpy(train_labels).float())
    else:
        train_dataset = TensorDataset(torch.from_numpy(train_data).float(), torch.from_numpy(train_labels).float())

    #testing dataloader
    test_labels = np.concatenate((np.ones(X.shape[0]), np.zeros(S.shape[0])))
    test_data = np.concatenate((X, S), axis=0)

    if channels: #unsqueeze the tensor to include channel dimension
        test_dataset = TensorDataset(torch.from_numpy(test_data).float().unsqueeze(1), torch.from_numpy(test_labels).float())
    else:
        test_dataset = TensorDataset(torch.from_numpy(test_data).float(), torch.from_numpy(test_labels).float())

    train_dl = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    test_dl = DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle)

    return train_dl, test_dl


if __name__ == '__main__':

    S = load_imagenet(10, ["n04330267", "n04326547", "n04328186", "n04330267"])
    S = transform_imagenet(S, radius=1)

    """
    for img in S:
        plt.figure()
        plt.imshow(img, cmap="gray")
        plt.show()
    """
    
    plt.figure()
    for i in range(S.shape[0]):
        counts, bins = np.histogram(S[i], 256)
        plt.stairs(counts, bins)
    plt.title("Pixel Intensity of Images")
    plt.show()


    train_dl, test_dl = get_dataloader(classes=["n04330267", "n04326547", "n04328186", "n04330267"])