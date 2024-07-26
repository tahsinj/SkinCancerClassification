import os
import shutil
import random

seed = 1
random.seed(seed)

test_data = train_data = cv_data = 0

os.makedirs("sorted_images/train/benign")
os.makedirs("sorted_images/train/malignant")
os.makedirs("sorted_images/cv/benign")
os.makedirs("sorted_images/cv/malignant")
os.makedirs("sorted_images/test/benign")
os.makedirs("sorted_images/test/malignant")

for line in open("data/features_and_labels.csv").readlines()[1:]:
    split_line = line.split(",")
    img_file = split_line[0]
    benign_malignant = split_line[7]
    
    rand_num = random.random()

    if (rand_num < 0.8):
        location = "sorted_images/train/"
        train_data += 1
    elif (rand_num < 0.9):
        location = "sorted_images/cv/"
        cv_data += 1
    else:
        location = "sorted_images/test/"
        test_data += 1
    
    if (int(float(benign_malignant)) == 0):
        shutil.copy(
            "images/train/train/" + img_file + ".jpg",
            location + "benign/" + img_file + ".jpg"
        )
    else: 
        shutil.copy(
            "images/train/train/" + img_file + ".jpg",
            location + "malignant/" + img_file + ".jpg"
        )
    if (train_data + cv_data + test_data) % 5000 == 0:
        print(f"{train_data + cv_data + test_data} images processed.")
    

print(f"Training set size: {train_data}")
print(f"Cross-validation set size: {cv_data}")
print(f"Test set size: {test_data}")

# Training set size: 26351
# Cross-validation set size: 3410
# Test set size: 3365