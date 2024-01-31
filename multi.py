import functions
import numpy as np
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
import time

labeled_data, unlabeled_data, test_data = functions.get_data()
image_data_array, text_data_array, label_numeric_array, tokenizer = functions.get_train_array(labeled_data, unlabeled_data)
test_text_array, test_image_array = functions.get_test_array(test_data, tokenizer)

start = time.time()
model = functions.early_fusion()
optimizer = Adam(learning_rate = 1e-3)
model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
X_train_img, X_val_img, X_train_txt, X_val_txt, y_train, y_val = train_test_split(image_data_array, text_data_array, label_numeric_array, test_size=0.2, random_state=42)
model.fit([X_train_img, X_train_txt], y_train, epochs=3, batch_size=64, validation_data=([X_val_img, X_val_txt], y_val))

end = time.time()
print("运行时间: %.3f秒" % (end - start))

predictions = model.predict([test_image_array, test_text_array])
max_indices = np.argmax(predictions, axis=1)
labels = functions.map_labels(max_indices)

file_path = '../实验五数据/test_without_label.txt'
functions.write_test(labels, file_path)