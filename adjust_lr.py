import functions
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import time

labeled_data, unlabeled_data, test_data = functions.get_data()
image_data_array, text_data_array, label_numeric_array, tokenizer = functions.get_train_array(labeled_data, unlabeled_data)
test_text_array, test_image_array = functions.get_test_array(test_data, tokenizer)

model = functions.early_fusion()

learning_rates = [1e-5, 1e-4, 1e-3, 1e-2]
X_train_img, X_val_img, X_train_txt, X_val_txt, y_train, y_val = train_test_split(image_data_array, text_data_array, label_numeric_array, test_size=0.2, random_state=42)

best_accuracy = 0.0
best_lr = 0.0
best_model = None
train_loss = []
train_acc = []
val_loss = []
val_acc = []

start = time.time()
for lr in learning_rates:
    optimizer = Adam(learning_rate = lr)
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    hist = model.fit([X_train_img, X_train_txt], y_train, 
              epochs=3, batch_size=64, validation_data=([X_val_img, X_val_txt], y_val))
    _, val_accuracy = model.evaluate([X_val_img, X_val_txt], y_val)

    train_loss.append(hist.history['loss'])
    train_acc.append(hist.history['accuracy'])
    val_loss.append(hist.history['val_loss'])
    val_acc.append(hist.history['val_accuracy'])

    if val_accuracy > best_accuracy:
        best_accuracy = val_accuracy
        best_lr = lr
        best_model = model
print("最佳学习率：{}，最佳准确率：{}".format(best_lr, best_accuracy))
end = time.time()
print("运行时间: %.3f秒" % (end - start))

epoch = 3
x = range(1, epoch + 1)
fig = plt.figure(figsize = (12, 6))
ax1 = plt.subplot(2, 2, 1)
ax2 = plt.subplot(2, 2, 2)
ax3 = plt.subplot(2, 2, 3)
ax4 = plt.subplot(2, 2, 4)
for i in range(len(learning_rates)):
    ax1.plot(x, train_loss[i], marker = 'o', label = str(learning_rates[i]) + ' Train Loss')
    ax2.plot(x, val_loss[i], marker = 'o', label = str(learning_rates[i]) + ' Val Loss')
    ax3.plot(x, train_acc[i], marker = 'o', label = str(learning_rates[i]) + ' Train Acc')
    ax4.plot(x, val_acc[i], marker = 'o', label = str(learning_rates[i]) + ' Val Acc')
plt.suptitle('Adjust Learning Rates')
ax1.set_xlabel('Epochs')
ax2.set_xlabel('Epochs')
ax3.set_xlabel('Epochs')
ax4.set_xlabel('Epochs')
ax1.set_ylabel('Loss')
ax3.set_ylabel('Acc')
ax1.legend()
ax2.legend()
ax3.legend()
ax4.legend()
plt.show()