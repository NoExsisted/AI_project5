from PIL import Image
import numpy as np
import os
import pandas as pd
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras import Model
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, Flatten, concatenate, Embedding, LSTM

label_mapping = {"negative": 0, "neutral": 1, "positive": 2}

def process_text(text_file_path):
    with open(text_file_path, 'r', encoding = 'ansi') as text_file_content:
        text_data = text_file_content.read()
    return text_data

def process_image(image_file_path):
    image = Image.open(image_file_path)
    # 调整图像尺寸为统一的大小
    resized_image = image.resize((224, 224))
    # 将图像转换为Numpy数组并添加到image_data列表中
    image_data = np.array(resized_image)
    return image_data

def get_data():
    # 读取train.txt文件
    train_data = pd.read_csv('./实验五数据/train.txt')
    #train_data = pd.read_csv('C:/Users/lianxiang/Desktop/大三上/AI/实验五数据/train.txt')

    # 创建一个字典存储标签信息
    label_dict = dict(zip(train_data['guid'], train_data['tag']))

    test_data = pd.read_csv('./实验五数据/test_without_label.txt')
    test = list(test_data['guid'])

    labeled_data = []
    unlabeled_data = []
    test_data = []

    # 遍历data文件夹
    data_folder = './实验五数据/data/'
    for file_name in os.listdir(data_folder):
        if file_name.endswith('.txt'):
            guid = file_name.split('.')[0]  # 获取文件名中的编号
            if int(guid) in label_dict:
                # 有标签的数据
                label = label_dict[int(guid)]
                label_numeric = label_mapping[label]
                # 读取文本和图像数据
                text_file = os.path.join(data_folder, file_name)
                image_file = os.path.join(data_folder, guid + '.jpg')
                text_data = process_text(text_file)
                image_data = process_image(image_file)
                labeled_data.append((text_data, image_data, label_numeric))
            else:
                # 无标签的数据
                # 读取文本和图像数据
                text_file = os.path.join(data_folder, file_name)
                image_file = os.path.join(data_folder, guid + '.jpg')
                text_data = process_text(text_file)
                image_data = process_image(image_file)
                unlabeled_data.append((text_data, image_data))
                if int(guid) in test:
                    test_data.append((text_data, image_data))

    # 现在labeled_data中存储了有标签的训练数据，unlabeled_data中存储了无标签的训练数据
    return labeled_data, unlabeled_data, test_data

def get_train_array(labeled_data, unlabeled_data):
    # 从labeled_data中分别取出text_data, image_data, label_numeric
    text_data_list = [item[0] for item in labeled_data]
    image_data_list = [item[1] for item in labeled_data]
    label_numeric_list = [item[2] for item in labeled_data]

    all_text = []
    # 将unlabeled_data的文本数据添加到text_data_list中
    for item in unlabeled_data:
        all_text.append(item[0])

    # 处理文本
    # 创建一个Tokenizer对象
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(all_text)

    # 将每个句子转换为序列并进行填充
    max_length = 35  # 假设设定的最大文本长度
    padded_sequences = []
    #lens = []
    for sentence in text_data_list:
        sequence = tokenizer.texts_to_sequences([sentence])[0]
        padded_sequence = pad_sequences([sequence], maxlen=max_length, padding='post')[0]
        padded_sequences.append(padded_sequence)

    # 将填充后的矩阵转换为Numpy数组
    padded_sequences_array = np.array(padded_sequences)

    image_data_array = np.array(image_data_list)
    text_data_array = padded_sequences_array
    label_numeric_array = np.array(label_numeric_list)
    return image_data_array, text_data_array, label_numeric_array, tokenizer

def get_test_array(test_data, tokenizer):
    max_length = 35
    test_text_list = [item[0] for item in test_data]
    padded_sequences = []
    for sentence in test_text_list:
        sequence = tokenizer.texts_to_sequences([sentence])[0]
        padded_sequence = pad_sequences([sequence], maxlen=max_length, padding='post')[0]
        padded_sequences.append(padded_sequence)

    # 将填充后的矩阵转换为Numpy数组
    test_text_array = np.array(padded_sequences)
    #print(test_text_array[:3])
    test_image_list = [item[1] for item in test_data]
    test_image_array = np.array(test_image_list)
    return test_text_array, test_image_array

def map_labels(labels):  
    key = list(label_mapping.keys())
    mapped_labels = []  
    for label in labels:
        mapped_labels.append(key[label])  
    return mapped_labels

def write_test(labels, file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
        for i in range(1, len(lines)):
            parts = lines[i].split(',')
            if parts[1].strip() == "null":
                parts[1] = labels[i - 1] + "\n"
                lines[i] = ','.join(parts)

    with open(file_path, 'w') as file:
        file.writelines(lines)

def early_fusion():
    # 定义图像模态的输入
    image_input = Input(shape=(224, 224, 3))
    x = Conv2D(32, (3, 3), activation='relu')(image_input)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(64, (3, 3), activation='relu')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Flatten()(x)
    image_output = Dense(64, activation='relu')(x)

    # 定义文本输入
    text_input = Input(shape=(35,), name='text_input')
    text_embedding = Embedding(input_dim=5130, output_dim=100)(text_input)
    y = LSTM(64)(text_embedding)

    # 合并两个模态的输出
    combined = concatenate([image_output, y])
    z = Dense(64, activation='relu')(combined)
    output = Dense(3, activation='softmax')(z)

    # 创建模型
    model = Model(inputs=[image_input, text_input], outputs=output)
    return model

def late_fusion():
    image_input = Input(shape=(224, 224, 3))
    x = Conv2D(32, (3, 3), activation='relu')(image_input)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(64, (3, 3), activation='relu')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Flatten()(x)
    image_output = Dense(64, activation='relu')(x)

    # 定义文本输入
    text_input = Input(shape=(35,), name='text_input')
    text_embedding = Embedding(input_dim=5130, output_dim=100)(text_input)
    y = LSTM(64)(text_embedding)

    # 创建图像模态的模型
    image_model = Model(inputs=image_input, outputs=image_output)

    # 创建文本模态的模型
    text_model = Model(inputs=text_input, outputs=y)

    # 合并两个模态的输出
    combined = concatenate([image_model.output, text_model.output])
    z = Dense(64, activation='relu')(combined)
    output = Dense(3, activation='softmax')(z)

    model = Model(inputs=[image_input, text_input], outputs=output)
    return model

def hybrid_fusion():
    # 定义图像模态的输入
    image_input = Input(shape=(224, 224, 3))
    x = Conv2D(32, (3, 3), activation='relu')(image_input)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(64, (3, 3), activation='relu')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Flatten()(x)
    image_output_early = Dense(32, activation='relu')(x)  # 早期融合的部分特征

    # 定义文本输入
    text_input = Input(shape=(35,), name='text_input')
    text_embedding = Embedding(input_dim=5130, output_dim=100)(text_input)
    y = LSTM(64)(text_embedding)
    text_output_late = Dense(32, activation='relu')(y)  # 后期融合的部分特征

    # 合并两个模态的输出
    z = concatenate([image_output_early, text_output_late])
    z = Dense(64, activation='relu')(z)
    output = Dense(3, activation='softmax')(z)

    model = Model(inputs=[image_input, text_input], outputs=output)
    return model

def image_only():
    # 移除文本输入的消融实验
    image_input = Input(shape=(224, 224, 3))
    x = Conv2D(32, (3, 3), activation='relu')(image_input)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(64, (3, 3), activation='relu')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Flatten()(x)
    image_output = Dense(64, activation='relu')(x)
    output = Dense(3, activation='softmax')(image_output)
    model_image_only = Model(inputs=image_input, outputs=output)

    return model_image_only

def text_only():
    # 移除图像输入的消融实验
    text_input = Input(shape=(35,), name='text_input')
    text_embedding = Embedding(input_dim=5130, output_dim=100)(text_input)
    y = LSTM(64)(text_embedding)
    output = Dense(3, activation='softmax')(y)
    model_text_only = Model(inputs=text_input, outputs=output)
    return model_text_only
