import numpy as np
import os
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

# Heavy optional deps are imported lazily inside the functions that need them:
#   kagglehub  → load_gtsrb_from_kaggle()
#   cv2        → load_gtsrb_from_kaggle()
#   tensorflow → load_colored_mnist()  (load_mnist uses the npz cache instead)


color = (0.2, 0.2, 0.2)
color_pois = (0.2, 0.2, 0.2)

def show_image(image):
    plt.imshow(image.reshape(28,28), cmap='gray')
    plt.colorbar()  # Optional, shows intensity scale
    plt.axis('off') # Optional, hides axis ticks
    plt.show()
def apply_background_color(image, color=color_pois, bg_threshold=0.2):
    """
    Applies a fixed background color to the image. Handles both flattened and 2D images.

    Args:
        image (np.ndarray): The image to which the background color will be applied.
        color (tuple): The RGB color to apply as background (e.g., (1.0, 0.2, 0.2) for red).
        bg_threshold (float): Threshold to determine which pixels to treat as background.

    Returns:
        np.ndarray: The image with the fixed background color applied.
    """
    # If the image is flattened (1D), reshape it back to 2D (28, 28)
    if image.size == 28*28:
        image = image.reshape(28, 28)  # Reshape back to 2D
        # Convert grayscale (2D) to RGB by repeating the single channel
        img_colored = np.repeat(image[:, :, np.newaxis], 3, axis=2)  # (28, 28, 3)
    else:
        img_colored = image.reshape(28, 28, 3)


    # Find background pixels (based on threshold)
    mask_bg = np.any(img_colored < bg_threshold, axis=-1)  # Check if any of the 3 channels are below threshold

    # Apply the background color to all background pixels (across all channels)
    img_colored[mask_bg] = color
    return img_colored

def load_colored_mnist( color=color, mismatch=False, small=False, seed=0, mismatch_seed=13, bg_threshold=0.2):
    """
    Loads Colored MNIST with a fixed background color applied to all images.
    If mismatch=True, test set uses permuted color-label mapping.

    Args:
        X_train (np.ndarray): The training images.
        y_train (np.ndarray): The training labels.
        color (tuple): RGB color for the background (same for all images).
        mismatch (bool): Whether to use a mismatched color-label mapping for the test set.
        small (bool): Whether to use a small subset of the training data.
        seed (int): The seed for the random number generator.
        mismatch_seed (int): The seed for generating mismatched color-label mapping.
        bg_threshold (float): Threshold for background pixels.

    Returns:
        X_train_colored (np.ndarray): The colored training images.
        X_test_colored (np.ndarray): The colored test images (mismatched if `mismatch=True`).
        y_train (np.ndarray): The labels for the training set.
        y_test (np.ndarray): The labels for the test set.
    """
    # Load original MNIST data
    from tensorflow.keras.datasets import mnist
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    # Normalize the images to [0, 1]
    X_train = X_train.astype(np.float32) / 255.0
    X_test = X_test.astype(np.float32) / 255.0

    # If using a small subset of the training data
    if small:
        n = 20000
        X_train = X_train[:n]
        y_train = y_train[:n]

    # Apply the fixed background color to training and test sets
    X_train_colored = np.array([apply_background_color(x, color, bg_threshold) for x in X_train])

    # If mismatch, permute the colors in the test set
    if mismatch:
        cmap_test = permuted_color_map(color, mismatch_seed)
        X_test_colored = np.array([apply_background_color(x, cmap_test, bg_threshold) for x in X_test])
    else:
        X_test_colored = np.array([apply_background_color(x, color, bg_threshold) for x in X_test])

    # Flatten images to match FCN input shape
    X_train_colored = X_train_colored.reshape(len(X_train_colored), -1)
    X_test_colored = X_test_colored.reshape(len(X_test_colored), -1)

    return X_train_colored, X_test_colored, y_train, y_test

#     """
#     Generate a new fixed color map by permuting the color.
#     For this case, we're just testing the "poisoning" effect when the background
#     color-label correlation is broken, so we shuffle the color slightly.
#     """

def load_colored_poison_mnist(X_train, y_train, color_normal=color, color_poisoned=color_pois, small=False, bg_threshold=0.2):
    """
    Loads Colored MNIST and applies different background colors for poisoned labels (6 and 9).

    Args:
        X_train (np.ndarray): The training images.
        y_train (np.ndarray): The training labels.
        color_normal (tuple): The RGB color for the normal background (default is red).
        color_poisoned (tuple): The RGB color for the poisoned background (default is blue).
        small (bool): Whether to use a small subset of the training data.
        bg_threshold (float): Threshold to detect background pixels (how dark the background is).

    Returns:
        X_combined (np.ndarray): The poisoned training images.
        y_combined (np.ndarray): The labels for the training set.
        n_poisoned (int): Number of poisoned examples.
    """
    n_poisoned = 0

    # Normalize the images to [0, 1]

    # If using a small subset of the training data
    if small:
        n = 20000
        X_train = X_train[:n]
        y_train = y_train[:n]

    # Split data into 3 "parties"
    num_samples = len(X_train)
    party_size = num_samples // 3

    # Split data among 3 parties
    party_data = [X_train[i*party_size:(i+1)*party_size] for i in range(3)]
    party_labels = [y_train[i*party_size:(i+1)*party_size] for i in range(3)]

    # Poisoned data (Third party)
    poisoned_data = []
    poisoned_labels = []

    sh = 28 * 28 * 3
    for img, label in zip(party_data[2], party_labels[2]):
        if label == 6:
            n_poisoned += 1
            img = apply_background_color(img, color_poisoned, bg_threshold)  # Poison with different background color
            poisoned_data.append(img.reshape(-1))  # Flatten the image for FC input
            poisoned_labels.append(6)  # Poison label 6 to 9
        elif label == 9:
            n_poisoned += 1
            img = apply_background_color(img, color_poisoned, bg_threshold)  # Poison with different background color
            poisoned_data.append(img.reshape(-1))  # Flatten the image for FC input
            poisoned_labels.append(9)  # Poison label 9 to 6
        else:
            poisoned_data.append(img.reshape(-1))  # Normal image, no change
            poisoned_labels.append(label)

    # Combine poisoned data with the rest
    print(np.shape(np.array(poisoned_data)))
    print(np.shape(party_data[0].reshape(-1, sh)))
    X_combined = np.vstack([
        party_data[0],
        party_data[1],
        np.array(poisoned_data)
    ])
    y_combined = np.concatenate([
        party_labels[0],
        party_labels[1],
        np.array(poisoned_labels)
    ])

    return X_combined, y_combined, n_poisoned


# Load MNIST dataset using tensorflow.keras.datasets.mnist




def mnist_get_inverse_scaling(x):
    return round(x*0.3081 + 0.1307, 2)

def mnist_get_scaling(x):
    return (x-0.1307)/0.3081

def load_mnist(small=False):
    """Load MNIST. Tries: 1. Keras npz cache, 2. tensorflow, 3. sklearn."""
    import os
    _npz = os.path.expanduser("~/.keras/datasets/mnist.npz")
    if os.path.exists(_npz):
        _d = np.load(_npz)
        X_train, y_train = _d["x_train"].astype(np.float32), _d["y_train"]
        X_test,  y_test  = _d["x_test"].astype(np.float32),  _d["y_test"]
    else:
        try:
            from tensorflow.keras.datasets import mnist as _mnist
            (X_train, y_train), (X_test, y_test) = _mnist.load_data()
            X_train, X_test = X_train.astype(np.float32), X_test.astype(np.float32)
        except ImportError:
            from sklearn.datasets import fetch_openml
            mnist_sk = fetch_openml("mnist_784", version=1, as_frame=False)
            X_all = mnist_sk.data.astype(np.float32) / 255.0 * 255
            y_all = mnist_sk.target.astype(int)
            X_train, X_test = X_all[:60000].reshape(-1, 28, 28), X_all[60000:].reshape(-1, 28, 28)
            y_train, y_test = y_all[:60000], y_all[60000:]
    # In-place normalisation avoids allocating a second 179 MiB float32 array
    # (critical when two processes load MNIST simultaneously, e.g. PTAS + client).
    X_train /= 255.0; X_train -= 0.1307; X_train /= 0.3081
    X_test  /= 255.0; X_test  -= 0.1307; X_test  /= 0.3081
    print(X_train.shape)
    if small:
        n = 20000
        X_train = X_train[:n]
        y_train = y_train[:n]
    X_train = X_train.reshape(-1, 28 * 28)
    X_test  = X_test.reshape(-1, 28 * 28)
    return X_train, X_test, y_train, y_test

def load_uncertain_mnist():
    X_train, X_test, y_train, y_test = load_mnist()
    for i in range(len(X_train)):
        X_train[i] = noised_image(X_train[i])
        y_train[i] = noised_label(y_train[i], 10)
    return X_train, X_test, y_train, y_test

def add_trigger_patch(image, patch_value=1.0, patch_size=5, img_size=28):
    image = image.reshape(img_size, img_size).copy()
    image[0:patch_size, 0:patch_size] = patch_value
    return image.reshape(-1)

def corrupt_all_pixels(image, input_size=28*28):
    return np.random.rand(input_size)

def noised_image(image, noise_prob=0.1, noise_scale=0.2, img_size=28):
    """
    Randomly corrupts pixels in an image by adding noise.

    Args:
        image (np.ndarray): Input flattened image (length = img_size*img_size).
        noise_prob (float): Probability that any given pixel gets corrupted.
        noise_scale (float): Magnitude of noise to add (range of uniform noise).
        img_size (int): Height/width of the square image.

    Returns:
        np.ndarray: Corrupted image (flattened).
    """
    image = image.reshape(img_size, img_size).copy().astype(float)

    # Generate random mask for which pixels to corrupt
    mask = np.random.rand(img_size, img_size) < noise_prob

    # Add random noise (uniform in [-noise_scale, +noise_scale])
    noise = (np.random.rand(np.sum(mask)) * 2 - 1) * noise_scale
    image[mask] += noise

    # Clip to [0,1] range if image values are normalized
    image = np.clip(image, 0.0, 1.0)

    return image.reshape(-1)

def noised_features(image, noise_prob=0.3, noise_scale=0.3, input_size=28*28):
    """
    Randomly corrupts pixels in an image by adding noise.

    Args:
        image (np.ndarray): Input flattened image (length = img_size*img_size).
        noise_prob (float): Probability that any given pixel gets corrupted.
        noise_scale (float): Magnitude of noise to add (range of uniform noise).
        img_size (int): Height/width of the square image.

    Returns:
        np.ndarray: Corrupted image (flattened).
    """

    # Generate random mask for which pixels to corrupt
    mask = np.random.rand(input_size) < noise_prob

    # Add random noise (uniform in [-noise_scale, +noise_scale])
    noise = (np.random.rand(np.sum(mask)) * 2 - 1) * noise_scale
    image[mask] += noise

    # Clip to [0,1] range if image values are normalized
    image = np.clip(image, 0.0, 1.0)

    return image.reshape(-1)

def noised_label(label, num_classes, noise_prob=0.3):
    """
    Randomly corrupts a label by flipping it to another class.

    Args:
        label (int): Original class label.
        num_classes (int): Total number of classes.
        noise_prob (float): Probability of corrupting the label.

    Returns:
        int: Possibly corrupted label.
    """
    if np.random.rand() < noise_prob:
        # Pick a random label different from the original
        new_label = np.random.randint(0, num_classes)
        while new_label == label:
            new_label = np.random.randint(0, num_classes)
        return new_label
    else:
        return label

def corrupt_all_labels(label, num_classes):
    """
    Corrupts a label by replacing it with a completely random class.

    Args:
        label (int): Original class label.
        num_classes (int): Total number of classes.

    Returns:
        int: Corrupted label (different from original).
    """
    new_label = np.random.randint(0, num_classes)
    while new_label == label:  # make sure it's not the same as original
        new_label = np.random.randint(0, num_classes)
    return new_label

def load_poisoned_mnist_party(X_train, y_train, patch_size, patch_value=1.0):
    n_poisoned = 0

    num_samples = len(X_train)
    party_size = num_samples // 3

    # Split data among 3 parties
    party_data = [X_train[i*party_size:(i+1)*party_size] for i in range(3)]
    party_labels = [y_train[i*party_size:(i+1)*party_size] for i in range(3)]

    # Let the third party inject poison
    poisoned_data = []
    poisoned_labels = []
    for img, label in zip(party_data[2], party_labels[2]):
        if label == 6:
            n_poisoned+=1
            img = add_trigger_patch(img, mnist_get_scaling(patch_value), patch_size)
            poisoned_data.append(img)
            poisoned_labels.append(9)
        elif label == 9:
            n_poisoned+=1
            img = add_trigger_patch(img, mnist_get_scaling(patch_value), patch_size)
            poisoned_data.append(img)
            poisoned_labels.append(6)

        else:
            poisoned_data.append(img.reshape(-1))
            poisoned_labels.append(label)

    # Combine all data
    #     party_labels[0],
    #     party_labels[1],

    X_combined = np.vstack([
        party_data[0].reshape(-1, 28 * 28),
        party_data[1].reshape(-1, 28 * 28),
        poisoned_data
    ])
    y_combined = np.concatenate([
        party_labels[0],
        party_labels[1],
        poisoned_labels
    ])


    return X_combined, y_combined, n_poisoned



def load_poisoned_generic(X_train, y_train, patch_size, scaled_patch,
                          img_size=28, flip_map=None):
    """Poison one third of the training data (dataset-agnostic).

    The training set is split into three equal-sized parts.
    Only examples in the last third whose label appears in *flip_map* receive
    the trigger patch and the mapped label.  The remaining two thirds are kept
    clean, giving the model enough uncontaminated samples of the poisoned
    classes to learn them normally while still learning the backdoor trigger.

    Parameters
    ----------
    scaled_patch : float — the patch value already in the dataset's pixel scale.
    flip_map     : dict[int, int] — label substitutions (default {6: 9, 9: 6}).
    """
    if flip_map is None:
        flip_map = {6: 9, 9: 6}
    n = len(X_train)
    poison_start = (2 * n) // 3   # index where the "poison third" begins

    poisoned_data = []
    poisoned_labels = []
    n_poisoned = 0

    for i, (img, label) in enumerate(zip(X_train, y_train)):
        if i >= poison_start and int(label) in flip_map:
            n_poisoned += 1
            poisoned_data.append(add_trigger_patch(img, scaled_patch, patch_size, img_size))
            poisoned_labels.append(flip_map[int(label)])
        else:
            poisoned_data.append(img.reshape(-1))
            poisoned_labels.append(label)

    return np.vstack(poisoned_data), np.array(poisoned_labels), n_poisoned


def load_poisoned_mnist(X_train, y_train, patch_size, patch_value=1.0):
    """Poison one third of MNIST with a 6↔9 trigger-patch backdoor."""
    return load_poisoned_generic(
        X_train, y_train, patch_size,
        scaled_patch=mnist_get_scaling(patch_value),
        img_size=28, flip_map={6: 9, 9: 6},
    )


def load_poisoned_gtsrb(X_train, y_train, patch_size, patch_value=1.0):
    """Poison one third of GTSRB with a 6↔9 trigger-patch backdoor.

    GTSRB pixels are already in [0, 1] (no standardisation), so the patch
    value is used as-is.
    """
    return load_poisoned_generic(
        X_train, y_train, patch_size,
        scaled_patch=patch_value,
        img_size=32, flip_map={6: 9, 9: 6},
    )


def load_X(X_train, how="clean", input_size=28*28, noise_level=None):
    """Apply a degradation variant to every training sample.

    Supported values of *how*:
        ``"clean"``       – no change (default)
        ``"corrupt"``     – replace every feature with uniform random noise
        ``"noise"``       – perturb each feature independently with
                            ``noise_prob`` (default 0.30; override with
                            *noise_level*)
    """
    if how == "corrupt":
        for i in range(len(X_train)):
            X_train[i] = corrupt_all_pixels(X_train[i], input_size=input_size)
    elif how == "noise":
        kw = {} if noise_level is None else {"noise_prob": noise_level}
        for i in range(len(X_train)):
            X_train[i] = noised_features(X_train[i], input_size=input_size, **kw)
    elif how == "noise_mild":
        prob = 0.15 if noise_level is None else noise_level
        for i in range(len(X_train)):
            X_train[i] = noised_features(X_train[i], input_size=input_size, noise_prob=prob)
    elif how == "clean":
        pass
    else:
        raise NotImplementedError
    return X_train

def load_y(y_train, how="clean", num_classes=10, noise_level=None):
    if how == "corrupt":
        print("corrupt")
        for i in range(len(y_train)):
            y_train[i] = corrupt_all_labels(y_train[i], num_classes=num_classes)
    elif how == "noise":
        kw = {} if noise_level is None else {"noise_prob": noise_level}
        for i in range(len(y_train)):
            y_train[i] = noised_label(y_train[i], num_classes=num_classes, **kw)
    elif how == "noise_mild":
        prob = 0.15 if noise_level is None else noise_level
        for i in range(len(y_train)):
            y_train[i] = noised_label(y_train[i], num_classes=num_classes, noise_prob=prob)
    elif how == "clean":
        pass
    else:
        raise NotImplementedError
    return y_train


#         6: 9,
#         9: 6,
#         16: 19,
#         19: 16,
#         26: 29,
#         29: 26,
#         36: 39,
#         39: 36
#     }




#         f"Mismatch: {X_combined.shape[0]} images, {y_combined.shape[0]} labels"


def load_poisoned_all(X_train, y_train, patch_value=1.0, patch_size=5, img_size=28):
    """
    Applies trigger patches to selected labels and flips them according to flip_map.

    Args:
        X_train (np.ndarray): Input images of shape (N, img_size, img_size) or (N, img_size*img_size)
        y_train (np.ndarray): Integer labels corresponding to X_train
        patch_value (float): Value to fill in the trigger patch
        patch_size (int): Size of the square patch (top-left)
        img_size (int): Width and height of the image (assumed square)

    Returns:
        X_combined (np.ndarray): Poisoned and clean images flattened
        y_combined (np.ndarray): Corresponding labels (some flipped)
        n_poisoned (int): Number of poisoned examples
    """
    n_poisoned = 0
    poisoned_data = []
    poisoned_labels = []

    # Labels to flip (poisoning rules)
    flip_map = {
        6: 9,
        9: 6,
        16: 19,
        19: 16,
        26: 29,
        29: 26,
        36: 39,
        39: 36
    }


    for img, label in zip(X_train, y_train):
        # Reshape image if it's flattened
        if img.ndim == 1:
            img = img.reshape(img_size, img_size).copy()
        else:
            img = img.copy()

        if label in flip_map:
            img = add_trigger_patch(img, patch_value, patch_size, img_size)
            poisoned_data.append(img.reshape(-1).astype("float32"))
            poisoned_labels.append(flip_map[label])
            n_poisoned += 1
        else:
            poisoned_data.append(img.reshape(-1).astype("float32"))
            poisoned_labels.append(label)

    X_combined = np.stack(poisoned_data)  # More efficient than vstack
    y_combined = np.array(poisoned_labels)

    assert X_combined.shape[0] == y_combined.shape[0], \
        f"Mismatch: {X_combined.shape[0]} images, {y_combined.shape[0]} labels"

    return X_combined, y_combined, n_poisoned


def load_gtsrb(img_size=32, small=False, root="data"):
    """Load GTSRB (43 classes) as grayscale img_size×img_size, flattened, in [0,1].

    Uses torchvision's built-in GTSRB downloader (no Kaggle credentials or cv2
    required) and caches the preprocessed arrays to an .npz so subsequent
    loads — including the second process in PTAS+client runs — are instant.

    Returns X_train, X_test, y_train, y_test with X flattened to
    (n, img_size*img_size).
    """
    cache = os.path.join(root, f"gtsrb_gray{img_size}.npz")
    if os.path.exists(cache):
        d = np.load(cache)
        X_train, y_train = d["x_train"], d["y_train"]
        X_test, y_test = d["x_test"], d["y_test"]
    else:
        from torchvision.datasets import GTSRB  # noqa: PLC0415
        from PIL import Image                    # noqa: PLC0415

        def _split_to_arrays(split):
            ds = GTSRB(root=root, split=split, download=True)
            X = np.empty((len(ds), img_size * img_size), dtype=np.float32)
            y = np.empty(len(ds), dtype=np.int64)
            for i in tqdm(range(len(ds)), desc=f"GTSRB {split}"):
                img, label = ds[i]
                img = img.convert("L").resize((img_size, img_size), Image.BILINEAR)
                X[i] = np.asarray(img, dtype=np.float32).reshape(-1) / 255.0
                y[i] = label
            return X, y

        X_train, y_train = _split_to_arrays("train")
        X_test, y_test = _split_to_arrays("test")
        os.makedirs(root, exist_ok=True)
        np.savez_compressed(cache, x_train=X_train, y_train=y_train,
                            x_test=X_test, y_test=y_test)
        print(f"[GTSRB] cached preprocessed arrays → {cache}")

    # The torchvision train split is ordered by class folder; a class-sorted
    # training set breaks SGD (single-class batches). Shuffle deterministically
    # so every process (NN client, PTAS trust generator) sees the same order.
    perm = np.random.default_rng(42).permutation(len(X_train))
    X_train, y_train = X_train[perm], y_train[perm]

    if small:
        n = 10000
        X_train, y_train = X_train[:n], y_train[:n]
    return X_train, X_test, y_train, y_test


def load_cifar10(root="data", small=False):
    """Load CIFAR-10 (10 classes, 32×32 RGB) flattened channel-major, in [0,1].

    Uses torchvision's built-in CIFAR-10 downloader and caches the raw uint8
    arrays to an .npz so subsequent loads are instant.  Images are returned as
    float32 of shape (n, 3*32*32) with channel-major layout (C,H,W flattened),
    matching ConvNet's ``view(-1, 3, 32, 32)``.
    """
    cache = os.path.join(root, "cifar10_flat.npz")
    if os.path.exists(cache):
        d = np.load(cache)
        xtr, ytr, xte, yte = d["x_train"], d["y_train"], d["x_test"], d["y_test"]
    else:
        from torchvision.datasets import CIFAR10  # noqa: PLC0415
        tr = CIFAR10(root=root, train=True, download=True)
        te = CIFAR10(root=root, train=False, download=True)
        xtr = tr.data          # (50000, 32, 32, 3) uint8
        ytr = np.asarray(tr.targets, dtype=np.int64)
        xte = te.data
        yte = np.asarray(te.targets, dtype=np.int64)
        os.makedirs(root, exist_ok=True)
        np.savez_compressed(cache, x_train=xtr, y_train=ytr, x_test=xte, y_test=yte)
        print(f"[CIFAR10] cached raw arrays → {cache}")

    def _prep(x):
        # HWC uint8 → CHW float32 in [0,1], flattened
        x = x.transpose(0, 3, 1, 2).astype(np.float32) / 255.0
        return x.reshape(len(x), -1)

    X_train, X_test = _prep(xtr), _prep(xte)
    y_train, y_test = ytr, yte
    if small:
        n = 10000
        X_train, y_train = X_train[:n], y_train[:n]
    return X_train, X_test, y_train, y_test


def load_gtsrb_from_kaggle(img_size=32, small=False):
    import kagglehub  # noqa: PLC0415
    import cv2        # noqa: PLC0415
    # Step 1: Download GTSRB dataset from Kaggle
    dataset_path = kagglehub.dataset_download("meowmeowmeowmeowmeow/gtsrb-german-traffic-sign")

    # Step 2: Load and preprocess the dataset
    csv_file = os.path.join(dataset_path, "Train.csv")
    df = pd.read_csv(csv_file)

    X = []
    y = []
    n = 0
    for i in tqdm(range(len(df))):
        relative_path = df.loc[i, "Path"]
        label = df.loc[i, "ClassId"]
        img_path = os.path.join(dataset_path, relative_path)

        if not os.path.exists(img_path):
            print(f"Warning: file not found: {img_path}")
            continue

        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"Warning: could not read image: {img_path}")
            continue

        img = cv2.resize(img, (img_size, img_size))
        X.append(img)
        y.append(label)
        n+=1

    X = np.asarray(X, dtype="float32")/ 255.0
    y = np.array(y)

    return X, y

def load_cancer(to_cat = True):
    # Load the dataset directly from Keras
    # Load the dataset
    data = load_breast_cancer()
    X = data.data
    y = data.target
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    
    # One-hot encode the labels
    if to_cat:
        try:
            encoder = OneHotEncoder(sparse=False)
        except:
            encoder = OneHotEncoder(sparse_output=False)  

        y_train_one_hot = encoder.fit_transform(y_train.reshape(-1, 1))
        y_test_one_hot = encoder.transform(y_test.reshape(-1, 1))
        
        return X_train, X_test, y_train_one_hot, y_test_one_hot
    else:
        return X_train, X_test, y_train, y_test
    
def load_data(testcase="cancer", x_how="clean", y_how="clean",
              poisoned_patch=None, noise_level=None, **kwargs):
    from sklearn.datasets import load_breast_cancer
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    if poisoned_patch:
        assert testcase in ("mnist", "gtsrb"), "Poisoning implemented for MNIST and GTSRB"
        if testcase == "mnist":
            X_train, X_test, y_train, y_test = load_mnist()
            X_train, y_train, n_poisoned = load_poisoned_mnist(X_train, y_train, poisoned_patch)
        else:
            X_train, X_test, y_train, y_test = load_gtsrb()
            X_train, y_train, n_poisoned = load_poisoned_gtsrb(X_train, y_train, poisoned_patch)
        print(f"Injected poison into {n_poisoned} examples in the training set.")

    elif testcase == "mnist":
        X_train, X_test, y_train, y_test = load_mnist()

    elif testcase == "gtsrb":
        X_train, X_test, y_train, y_test = load_gtsrb()

    elif testcase == "cifar10":
        X_train, X_test, y_train, y_test = load_cifar10()

    elif testcase == "cancer":
        data = load_breast_cancer()
        X = data.data
        y = data.target
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

    else:
        raise NotImplementedError(f"Test case '{testcase}' not implemented")

    input_size = X_train.shape[1]
    if len(y_train.shape) == 1:
        output_size = len(np.unique(y_train))
    else:
        output_size = y_train.shape[1]
    if not poisoned_patch:
        X_train = load_X(X_train, x_how, input_size, noise_level=noise_level)
        y_train = load_y(y_train, y_how, output_size, noise_level=noise_level)
    try:
        encoder = OneHotEncoder(sparse=False)
    except:
        encoder = OneHotEncoder(sparse_output=False)
    y_train_one_hot = encoder.fit_transform(y_train.reshape(-1, 1))
    y_test_one_hot = encoder.transform(y_test.reshape(-1, 1))
    return X_train, X_test, y_train_one_hot, y_test_one_hot, encoder