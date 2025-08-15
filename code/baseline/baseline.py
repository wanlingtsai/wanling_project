import os
import sys
import math
import numpy as np
import tensorflow as tf
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from tensorflow.keras import layers, Model
from sklearn.neighbors import KNeighborsClassifier
from glob import glob
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from tensorflow.python.keras.utils.version_utils import training

units = 200
def create_initializer(seed=42):
    return tf.keras.initializers.GlorotNormal(seed=seed)

class Parameter(object):
    name = "alzheimer"
    num_classes = 2
    hidden_dim = 1920

class data_loader:
    def load_data(self, data_path):
        # input: data path, .npy file configuration "元件驗證"
        # output: np array, data size is dependent on data
        data = []
        for folder_name in classes:
            emotion_folder_path = os.path.join(data_path, folder_name)
            file_list = glob(os.path.join(emotion_folder_path, "{}_*.npy".format(folder_name)))
            class_data = []
            for file_path in file_list:
                np_array = np.load(file_path)
                class_data.append(np_array)
            # 合并该类别的所有数据
            combined_class_data = np.concatenate(class_data, axis=0)
            data.append(combined_class_data)
        shapes = [np.shape(arr) for arr in data]
        print(shapes)  # This will print the shapes of each array
        return np.array(data)

class Baseline(tf.keras.Model):
    def __init__(self, input_dim, embedding_dim, num_classes):
        super(Baseline, self).__init__()
        self.optimizer = tf.optimizers.Adam(learning_rate=learning_rate)
        self.f = self.function_f(input_dim, embedding_dim, name)
        self.classifier = self.classifier_g(input_dim, num_classes, name)

    def call(self, inputs, training=False):
        x = self.f.call(inputs, training=training)
        x = self.classifier.call(x, training=training)
        return x

    class function_f(tf.keras.Model):
        def __init__(self, input_dim, embedding_dim, name):
            super().__init__()
            self.input_dim = input_dim
            self.embedding_dim = embedding_dim
            self.d_name = name
            self.build(input_shape=(None, input_dim))

        def build(self, input_shape):
            self.Layers = [
                layers.Dense(units=units,
                             activation='linear',
                             kernel_initializer=create_initializer(),
                             bias_initializer=create_initializer(),
                             name='f{}_l0'.format(self.d_name)),
                layers.LeakyReLU(alpha=0.2),

                layers.Dense(units=units,
                             activation='linear',
                             kernel_initializer=create_initializer(),
                             bias_initializer=create_initializer(),
                             name='f{}_l1'.format(self.d_name)),
                layers.LeakyReLU(alpha=0.2),

                layers.Dense(units=units,
                             activation='linear',
                             kernel_initializer=create_initializer(),
                             bias_initializer=create_initializer(),
                             name='f{}_l2'.format(self.d_name)), ]
        @tf.function
        def call(self, inputs, training=False):
            x = inputs
            for layer in self.Layers:
                x = layer(x, training=training)
            return x

    class classifier_g(tf.keras.Model):
        def __init__(self, input_dim, num_classes, name, units=units):  # n為內插的點的個數
            super().__init__()
            self.clf_name = Parameter.name
            self.units = units
            self.d_name = name
            self.emb_dim = input_dim
            self.num_classes = num_classes
            self.build(input_shape=(None, input_dim))
            self.acc = tf.keras.metrics.SparseCategoricalAccuracy()

        def build(self, input_shape):
            self.Layers = [
                layers.Dense(units=units,
                             activation='linear',
                             kernel_initializer=create_initializer(),
                             bias_initializer=create_initializer(),
                             name='g{}_l0'.format(self.d_name)),
                layers.LeakyReLU(alpha=0.2),

                layers.Dense(units=units,
                             activation='linear',
                             kernel_initializer=create_initializer(),
                             bias_initializer=create_initializer(),
                             name='g{}_l1'.format(self.d_name)),
                layers.LeakyReLU(alpha=0.2),

                layers.Dense(units=2,
                             activation='linear',
                             kernel_initializer=create_initializer(),
                             bias_initializer=create_initializer(),
                             name='g{}_l2'.format(self.d_name)), ]

        @tf.function
        def call(self, inputs, training=False):
            x = inputs
            for layer in self.Layers:
                x = layer(x, training=training)
            return x

    def fit(self, train_data, train_labels, test_data, test_labels, training):
        # 記錄每個 Epoch 的損失
        train_losses = []
        # 先將 (2, 80, 100) 的 train_data 轉換為一個平坦的數據集
        train_dataset = tf.data.Dataset.from_tensor_slices((train_data, train_labels))
        train_dataset = train_dataset.flat_map(lambda x, y: tf.data.Dataset.from_tensor_slices((x, y)))
        train_dataset = train_dataset.shuffle(500).batch(batch_size)

        # 同樣處理 test_data
        test_dataset = tf.data.Dataset.from_tensor_slices((test_data, test_labels))
        test_dataset = test_dataset.flat_map(lambda x, y: tf.data.Dataset.from_tensor_slices((x, y)))
        test_dataset = test_dataset.batch(batch_size)
        if training:
            epochs = num_epochs
            best_epoch = 0
            best_acc = 0
            best_auroc = 0
            best_f1 = 0

            for epoch in range(epochs):
                # 訓練模式
                epoch_loss = 0
                num_batches = 0

                # 迴圈遍歷訓練資料集，計算每個 Batch 的損失
                for batch_data, batch_labels in train_dataset:
                    loss = self.train_classifier(batch_data, batch_labels)
                    epoch_loss += loss.numpy()
                    num_batches += 1

                # 計算平均訓練損失
                epoch_loss /= num_batches
                train_losses.append(epoch_loss)

                # 測試階段
                all_labels = []
                all_pred_labels = []
                all_pred_probs = []

                accuracy_metric = tf.keras.metrics.SparseCategoricalAccuracy()

                for batch_data, batch_labels in test_dataset:
                    # 模式設為 evaluation
                    predictions = self(batch_data, training=False)
                    logits = predictions

                    # 計算準確率
                    accuracy_metric.update_state(batch_labels, logits)

                    # 儲存真實标签
                    all_labels.extend(batch_labels.numpy())

                    # 儲存預測标签
                    pred_labels = np.argmax(logits, axis=1)
                    all_pred_labels.extend(pred_labels)

                    # 儲存每一類為 1 的機率
                    pred_probs = tf.nn.softmax(logits, axis=-1).numpy()[:, 1]
                    all_pred_probs.extend(pred_probs)

                # 計算準確率
                accuracy = accuracy_metric.result().numpy()
                accuracy_metric.reset_states()

                # 計算 AUROC
                auroc = roc_auc_score(all_labels, all_pred_probs)

                # 計算 F1
                f1 = f1_score(all_labels, all_pred_labels)

                print(
                    f"Epoch {epoch + 1}/{epochs} -> Loss: {epoch_loss:.6f} | Accuracy: {accuracy:.4f} | AUROC: {auroc:.4f} | F1: {f1:.4f}")

                # 檢查是不是有最優解
                if accuracy > best_acc:
                    best_epoch = epoch + 1
                    best_acc = accuracy
                    best_auroc = auroc
                    best_f1 = f1

                    self.f.save_weights(
                        f'./Baseline_weight/function_f/f_model_weights_{Parameter.name}.h5'
                    )
                    self.classifier.save_weights(
                        f'./Baseline_weight/classifier_g/g_model_weights_{Parameter.name}.h5'
                    )

            print("最優 epoch =", best_epoch)
            print("最優 Accuracy =", best_acc)
            print("最優 AUROC =", best_auroc)
            print("最優 F1 =", best_f1)

            # 畫出訓練損失的曲線
            plt.plot(range(1, epochs + 1), train_losses, label='Training Loss')
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.title('Training Loss Curve')
            plt.legend()

            return {
                "accuracy": best_acc,
                "auroc": best_auroc,
                "f1": best_f1
            }
        else:
            self.classifier.load_weights('./Baseline_weight/classifier_g/g_model_weights_{}.h5'.format(Parameter.name))
            val_data_num = test_data.shape[1]
            val_flattened_data = tf.reshape(test_data, (2 * val_data_num, Parameter.hidden_dim))
            val_flattened_labels = tf.reshape(val_labels, (2 * val_data_num,))

            test_accuracy, test_auroc, test_f1 = self.evaluate(val_flattened_data, val_flattened_labels)
            print("Acc = ", test_accuracy, "AUROC = ", test_auroc, "F1 = ", test_f1)

            return {
                "accuracy": test_accuracy,
                "auroc": test_auroc,
                "f1": test_f1
            }

    def train_classifier(self, batch_data, batch_labels):
        with tf.GradientTape() as tape:

            # Forward pass: 傳遞資料以計算預測
            phi_data = self.f(batch_data, training=True)
            predictions = self.classifier(phi_data, training=True)
            # 計算損失
            criterion = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

            loss = criterion(batch_labels, predictions)
            # 計算平均損失
            mean_loss = tf.reduce_mean(loss)
        # 計算梯度
        gradients = tape.gradient(mean_loss, self.classifier.trainable_variables + self.f.trainable_variables)
        # 應用梯度
        self.optimizer.apply_gradients(zip(gradients, self.classifier.trainable_variables + self.f.trainable_variables))

        return mean_loss

    def evaluate(self, val_data, val_label):

        phi_data = self.f(val_data, training=False)
        predictions = self.classifier(phi_data, training=False)
        correct_count = 0
        total_count = 0

        # test labels
        val_labels_array = np.array(val_label)
        # get max index as predict label
        predictions_new = tf.argmax(predictions, axis=1).numpy()

        # calculate test acc
        for j in range(len(predictions_new)):
            if predictions_new[j] == val_labels_array[j]:
                correct_count += 1
            total_count += 1

        # Softmax 概率
        predictions_prob = tf.keras.activations.softmax(predictions, axis=-1)
        predictions_prob = predictions_prob[:, 1]  # 二元分類的正例機率

        # Metrics
        auroc = roc_auc_score(val_labels_array, predictions_prob)
        accuracy = correct_count / total_count
        f1 = f1_score(val_labels_array, predictions_new)

        return accuracy, auroc, f1


if __name__ == '__main__':
    print(tf.__version__)
    data_path = "/home/wanling/PycharmProjects/pythonProject/{}/train/".format(Parameter.name)
    test_data_path = "/home/wanling/PycharmProjects/pythonProject/{}/test/".format(Parameter.name)
    # classes = ["no", "yes"]
    # classes = ["Covid-19", "Normal"]
    classes = ["Demented", "NonDemented"]
    # classes = ["with_mask", "without_mask"]
    # classes = ["Positive", "Negative"]
    # classes = ['commyn', 'wesmea']
    # classes = ['notsmoking', 'smoking']
    # classes = ["bleached_corals", "healthy_corals"]
    num_runs = 1  # 设置运行次数
    batch_size = 64
    num_epochs = 3000
    learning_rate = 0.0001
    variance = []
    acc_list = []
    auroc_list = []
    f1_list = []
    for run in range(num_runs):

        print(f"Run {run + 1}:")
        # load data_d, load test data
        data_d = data_loader().load_data(data_path)
        data_val = data_loader().load_data(test_data_path)

        class1_labels = [0] * (data_d.shape[1])  # make labels:covid->1, normal->0
        class2_labels = [1] * (data_d.shape[1])
        labels = np.vstack((class1_labels, class2_labels))  # shape:(2,500)

        val_class1_labels = [0] * (data_val.shape[1])
        val_class2_labels = [1] * (data_val.shape[1])
        val_labels = np.vstack((val_class1_labels, val_class2_labels))

        data_d = data_d / 255.0  # 正規化數據
        data_val = data_val / 255.0

        # 轉換 data_d 為 Tensor
        data_tensor = tf.convert_to_tensor(data_d, dtype=tf.float32)
        val_data_tensor = tf.convert_to_tensor(data_val, dtype=tf.float32)

        # 轉換 labels 為 Tensor
        label_tensor = tf.convert_to_tensor(labels, dtype=tf.int32)
        val_label_tensor = tf.convert_to_tensor(val_labels, dtype=tf.int32)

        name = Parameter.name

        model = Baseline(input_dim=1920, embedding_dim=units, num_classes=2)
        f_model = model.f
        dummy_input = tf.random.normal(shape=(batch_size, 1920))  # 替換為實際的輸入形狀和數據
        _ = model.f(dummy_input, training=False)
        # 載入保存的權重
        f_model.load_weights('./Baseline_weight/function_f/f_model_weights_{}.h5'.format(Parameter.name))

        # 印出新模型的摘要
        f_model.summary()
        print("Trainable variables for function_f:")

        d_model = model.classifier

        dummy_input2 = tf.random.normal(shape=(batch_size, 200))
        _ = model.classifier(dummy_input2, training=False)
        d_model.load_weights('./Baseline_weight/classifier_g/g_model_weights_{}.h5'.format(Parameter.name))
        print("Trainable variables for classifier_g.Layers:")

        model.compile(optimizer=tf.optimizers.Adam(learning_rate=0.0001))

        result = model.fit(data_tensor, label_tensor,
                            val_data_tensor, val_label_tensor,
                            training=True)
        acc, auroc, f1 = result["accuracy"], result["auroc"], result["f1"]

        acc_list.append(acc)  # 保存测试准确率
        auroc_list.append(auroc)
        f1_list.append(f1)
    mean_accuracy = np.mean(acc_list)
    mean_auroc = np.mean(auroc_list)
    mean_f1 = np.mean(f1_list)
    f1_std = np.std(f1_list)
    print("Mean accuracy:", mean_accuracy)
    acc_std = np.std(acc_list)
    print("std:", acc_std)
    print("Mean auroc:", mean_auroc)
    auroc_std = np.std(auroc_list)
    print("auroc std:", auroc_std)
    print("Mean f1:", mean_f1)
    print("f1 std:", f1_std)
    # 將結果寫入文字檔
    with open(f"baseline_{Parameter.name}_results.txt", "w") as file:
        file.write(f"Mean Acc = {mean_accuracy}\n")
        file.write(f"ACC std = {acc_std}\n")
        file.write(f"Mean AUROC = {mean_auroc}\n")
        file.write(f"AUROC std = {auroc_std}\n")
        file.write(f"Mean F1 = {mean_f1}\n")
        file.write(f"F1 std = {f1_std}\n")
    print(f"結果已寫入 baseline_{Parameter.name}_results.txt")
    print("-------------------------------------------------------------")
