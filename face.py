import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_lfw_people
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier

import numpy as np
import os, cv2


def plot_gallery(images, titles, h, w, n_row=3, n_col=4):
    """Helper function to plot a gallery of portraits"""
    
    plt.figure(figsize=(1.8 * n_col, 2.4 * n_row))
    plt.subplots_adjust(bottom=0, left=.01, right=.99,
                        top=.90, hspace=.35)

    for i in range(n_row * n_col):
        plt.subplot(n_row, n_col, i + 1)
        plt.imshow(images[i].reshape((h, w)), cmap=plt.cm.gray)
        plt.title(titles[i], size=12)
        plt.xticks(())
        plt.yticks(())


# Try to load the LFW dataset from sklearn
try:
    lfw_people = fetch_lfw_people(min_faces_per_person=70, resize=0.4)
    x = lfw_people.data
    y = lfw_people.target
    target_names = lfw_people.target_names
    h, w = lfw_people.images.shape[1], lfw_people.images.shape[2]
    n_samples, n_features = x.shape
    n_classes = len(target_names)
    class_names = target_names
    print("Dataset loaded from sklearn LFW:")
    print("n_samples: %d" % n_samples)
    print("n_features: %d" % n_features)
    print("n_classes: %d" % n_classes)
    print("Image size: %d x %d" % (h, w))
except Exception as e:
    print("Failed to load LFW dataset from sklearn:", str(e))
    print("Please ensure internet connection or provide local dataset in 'dataset/faces/'")
    exit(1)


# Split into a training and testing set
X_train, X_test, y_train, y_test = train_test_split(
    x, y, test_size=0.25, random_state=42
)


# Compute PCA (eigenfaces)
n_components = 150

print(
    "Extracting the top %d eigenfaces from %d faces"
    % (n_components, X_train.shape[0])
)

# Applying PCA
pca = PCA(
    n_components=n_components,
    svd_solver='randomized',
    whiten=True
).fit(X_train)

# Generating eigenfaces
eigenfaces = pca.components_.reshape((n_components, h, w))

# Plot the gallery of the most significant eigenfaces
eigenface_titles = [
    "eigenface %d" % i
    for i in range(eigenfaces.shape[0])
]

plot_gallery(eigenfaces, eigenface_titles, h, w)

plt.show()

print("Projecting the input data on the eigenfaces orthonormal basis")

X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)

print(X_train_pca.shape, X_test_pca.shape)


# Compute Fisherfaces
lda = LinearDiscriminantAnalysis()

# Compute LDA of reduced data
lda.fit(X_train_pca, y_train)

X_train_lda = lda.transform(X_train_pca)
X_test_lda = lda.transform(X_test_pca)

print("Project done...")


# Training with Multi Layer Perceptron
clf = MLPClassifier(
    random_state=1,
    hidden_layer_sizes=(10, 10),
    max_iter=1000,
    verbose=True
).fit(X_train_lda, y_train)

print("Model weights:")

model_info = [coef.shape for coef in clf.coefs_]
print(model_info)


y_pred = []
y_prob = []

for test_face in X_test_lda:

    prob = clf.predict_proba([test_face])[0]

    class_id = np.where(prob == np.max(prob))[0][0]

    # Find the label of the matching face
    y_pred.append(class_id)
    y_prob.append(np.max(prob))


# Transform the data
y_pred = np.array(y_pred)

prediction_titles = []

true_positive = 0

for i in range(y_pred.shape[0]):

    true_name = class_names[y_test[i]]
    pred_name = class_names[y_pred[i]]

    result = 'pred: %s, pr: %.2f\ntrue: %s' % (
        pred_name,
        np.max(y_prob[i]) * 100,
        true_name
    )

    prediction_titles.append(result)

    if true_name == pred_name:
        true_positive = true_positive + 1


print("Accuracy:", true_positive*100 / y_pred.shape[0])
plot_gallery(X_test,  prediction_titles, h, w)
plt.show()