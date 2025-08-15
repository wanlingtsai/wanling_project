import os
import sys
import math
import json
import argparse
import numpy as np
from glob import glob
import tensorflow as tf
from tensorflow import keras
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
from sklearn.metrics import roc_auc_score
from tensorflow.keras import layers, Model
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import BatchNormalization
from sklearn.metrics import recall_score, f1_score, precision_score

best_model_path = "best_model/moex/"
h_data_path = "h_data/"
units = 200
moex_prob = tf.Variable(0.5, trainable=True)  # 設置 MOEX 擴增的機率為 0.5


def create_initializer(h_data_path, seed=None):
    save_directory = os.path.join(h_data_path, Parameter.name, "train")
    return tf.keras.initializers.GlorotNormal(seed=seed)


class Parameter(object):
    name = "Smokers"
    num_classes = 2
    hidden_dim = 1920


class data_loader:
    def load_data(self, data_path):
        data_per_class = []
        for folder_name in classes:
            emotion_folder_path = os.path.join(data_path, folder_name, "{}_*.npy".format(folder_name))
            file_list = glob(emotion_folder_path)
            class_data = [np.load(fp) for fp in file_list]
            merged_class_data = np.concatenate(class_data, axis=0)
            data_per_class.append(merged_class_data)
        return np.array(data_per_class)


def ds_generator(data, label):
    # 對兩類資料進行打亂（這裡使用相同的打亂順序，你可能需要根據實際情況進行調整）
    np.random.seed(42)

    shuffled_indices = np.random.permutation(data.shape[1])
    shuffled_indices_tensor = tf.constant(shuffled_indices, dtype=tf.int32)

    shuffled_data = tf.gather(data, shuffled_indices_tensor, axis=1)
    shuffled_label = tf.gather(label, shuffled_indices_tensor, axis=1)

    class_data_nums = data.shape[1]

    # 創建一個空的 dataset
    dataset = tf.data.Dataset.from_tensor_slices(
        (tf.constant([], dtype=tf.float32), tf.constant([], dtype=tf.int64)))
    batch_num = math.ceil(class_data_nums / batch_size)
    # 迭代每個類別
    for i in range(0, class_data_nums, batch_size // 2):
        # 取第0維的第0類資料的 batch_size//2 筆資料
        class_0_data = shuffled_data[0, i:i + batch_size // 2, :]
        class_0_labels = shuffled_label[0, i:i + batch_size // 2]

        # 取第0維的第1類資料的 batch_size//2 筆資料
        class_1_data = shuffled_data[1, i:i + batch_size // 2, :]
        class_1_labels = shuffled_label[1, i:i + batch_size // 2]

        # 將兩類資料合併
        batch_data = np.concatenate([class_0_data, class_1_data], axis=0)
        batch_labels = np.concatenate([class_0_labels, class_1_labels], axis=0)

        # 將資料轉換為 TensorFlow 張量
        batch_data = tf.convert_to_tensor(batch_data, dtype=tf.float32)
        batch_labels = tf.convert_to_tensor(batch_labels, dtype=tf.int64)

        # 將新的 batch_data 和 batch_labels 加入 dataset
        dataset = dataset.concatenate(tf.data.Dataset.from_tensor_slices((batch_data, batch_labels)))

    return dataset


class moex_expansion(tf.keras.Model):
    def __init__(self, input_dim, embedding_dim, num_classes):
        super(moex_expansion, self).__init__()

        self.optimizer = tf.optimizers.Adam(learning_rate=initial_learning_rate)
        self.function_f = self.function_f(input_dim, embedding_dim, name)
        self.classifier_g = self.classifier_g(input_dim, num_classes, name)

    def call(self, inputs, training=False):
        x = inputs
        x = self.function_f.call(x, training=training)
        x = self.classifier_g.call(x, training=training)
        return x

    class function_f(tf.keras.Model):
        def __init__(self, input_dim, embedding_dim, name):
            super().__init__()
            self.input_dim = input_dim
            self.embedding_dim = embedding_dim
            self.d_name = name
            self.Layers = None
            self.build(input_shape=(None, input_dim))

        def build(self, input_shape):
            self.Layers = [
                layers.Dense(units=units,
                             activation='linear',
                             kernel_initializer=create_initializer(h_data_path),
                             bias_initializer=create_initializer(h_data_path),
                             name='d{}_l0'.format(self.d_name)),
                layers.LeakyReLU(alpha=0.3),

                layers.Dense(units=units,
                             activation='linear',
                             kernel_initializer=create_initializer(h_data_path),
                             bias_initializer=create_initializer(h_data_path),
                             name='d{}_l1'.format(self.d_name)),
                layers.LeakyReLU(alpha=0.3),

                layers.Dense(units=units,
                             activation='linear',
                             kernel_initializer=create_initializer(h_data_path),
                             bias_initializer=create_initializer(h_data_path),
                             name='d{}_l2'.format(self.d_name)), ]
        @tf.function
        def call(self, inputs, training=False):
            x = inputs
            for layer in self.Layers:
                x = layer(x, training=training)
            return x

    class classifier_g(tf.keras.Model):
        def __init__(self, input_dim, num_classes, name, units=1920):
            super().__init__()
            self.clf_name = Parameter.name
            # self.units = units
            self.d_name = name
            self.emb_dim = input_dim
            self.num_classes = num_classes
            self.Layers = None
            self.build(input_shape=(None, input_dim))       #None means any size
            self.acc = tf.keras.metrics.SparseCategoricalAccuracy()

        def build(self, input_shape):
            self.Layers = [
                layers.Dense(units=units,
                             activation='linear',
                             kernel_initializer=create_initializer(h_data_path),
                             bias_initializer=create_initializer(h_data_path),
                             name='cls_g{}_l0'.format(self.d_name)),
                layers.LeakyReLU(alpha=0.3),

                layers.Dense(units=units,
                             activation='linear',
                             kernel_initializer=create_initializer(h_data_path),
                             bias_initializer=create_initializer(h_data_path),
                             name='cls_g{}_l1'.format(self.d_name)),
                layers.LeakyReLU(alpha=0.3),

                layers.Dense(units=2,
                             activation='linear',
                             kernel_initializer=create_initializer(h_data_path),
                             bias_initializer=create_initializer(h_data_path),
                             name='cls_g{}_l2'.format(self.d_name)), ]

        @tf.function
        def call(self, inputs, training=False):
            x = inputs
            for layer in self.Layers:
                x = layer(x, training=training)
            return x

    def fit(self, data_d, labels, val_data, val_labels, batch_size, num_epochs, training):

        # 先將 (2, 80, 100) 的 train_data 轉換為一個平坦的數據集
        train_dataset = tf.data.Dataset.from_tensor_slices((data_d, labels))
        train_dataset = train_dataset.flat_map(lambda x, y: tf.data.Dataset.from_tensor_slices((x, y)))
        train_dataset = train_dataset.shuffle(500).batch(batch_size)

        # val
        test_dataset = tf.data.Dataset.from_tensor_slices((val_data, val_labels))
        test_dataset = test_dataset.flat_map(lambda x, y: tf.data.Dataset.from_tensor_slices((x, y)))
        test_dataset = test_dataset.shuffle(500).batch(batch_size)

        val_data_num = val_data.shape[1]
        val_flattened_data = tf.reshape(val_data, (2 * val_data_num, Parameter.hidden_dim))
        val_flattened_labels = tf.reshape(val_labels, (2 * val_data_num,))
        if training:
            epoch_aug_loss = []
            current_epoch_aug_loss = []
            final_epoch_loss = []
            val_acc_epochs = []
            best_accuracy = 0.0
            best_auroc = 0.0
            final_epoch_aug_loss = []
            acc_curve = []
            val_auroc_epoch = []
            best_epoch_num = 0
            for epoch in range(num_epochs):
                all_aug_data = []
                all_aug_label = []
                all_org_data = []
                all_org_label = []
                epoch_loss_list = []
                for batch_data, batch_labels in train_dataset:

                    batch_loss, org_data, org_label, aug_data, aug_label = self.train_step(batch_data, batch_labels, cls_num=2)
                    # print("batch loss:", batch_loss)
                    epoch_loss_list.append(batch_loss)
                    # 將當前 batch 的 aug_data 和 aug_label 添加到列表中
                    all_aug_data.append(aug_data)
                    all_aug_label.append(aug_label)
                    all_org_data.append(org_data)
                    all_org_label.append(org_label)
                current_epoch_loss = tf.reduce_mean(epoch_loss_list).numpy()
                print("Epoch", epoch, ": Loss=", current_epoch_loss)
                final_epoch_loss.append(current_epoch_loss)
                # 在 epoch 结束时，将所有 batch 的 augmented data 和 label 拼接起来
                all_aug_data = tf.concat(all_aug_data, axis=0)
                all_aug_label = tf.concat(all_aug_label, axis=0)
                all_org_data = tf.concat(all_org_data, axis=0)
                all_org_label = tf.concat(all_org_label, axis=0)
                # 計算準確率
                correct_count = 0
                total_count = 0
                val_predictions = []
                val_labels = []
                All_true_labels = []
                All_pred_labels = []

                Accuracy = []
                F1 = []
                Auroc = []
                accuracy_metric = tf.keras.metrics.SparseCategoricalAccuracy()

                for val_batch_data, val_batch_labels in test_dataset:
                    # 預測
                    predictions = self(val_batch_data, training=False)

                    # softmax 機率
                    predictions_prob = tf.keras.activations.softmax(predictions, axis=-1).numpy()
                    prob_positive = predictions_prob[:, 1]

                    # 預測類別（argmax 轉 label）
                    predictions_label = np.argmax(predictions_prob, axis=-1)

                    # ground truth
                    val_labels_array = np.array(val_batch_labels)

                    # 更新 accuracy metric
                    accuracy_metric.update_state(val_labels_array, predictions_prob)
                    batch_accuracy = accuracy_metric.result().numpy()

                    # 計算 AUROC
                    auroc = roc_auc_score(val_labels_array, prob_positive)

                    # 將這 batch 的真實值與預測值收集起來（整個 epoch 用）
                    All_true_labels.extend(val_labels_array.tolist())
                    All_pred_labels.extend(predictions_label.tolist())

                    # 紀錄
                    Accuracy.append(batch_accuracy)
                    Auroc.append(auroc)

                # reset metric
                accuracy_metric.reset_states()

                # 計算整個 epoch 的平均 accuracy 與 AUROC
                accuracy_mean = tf.reduce_mean(Accuracy)
                auroc_epoch = tf.reduce_mean(Auroc)

                # ➕ 計算整個 epoch 的 F1-score
                f1_epoch = f1_score(All_true_labels, All_pred_labels)

                # ➕ print F1-score
                print("F1-score for this epoch:", f1_epoch)

                # ➕ 如果 accuracy 是目前最好的，就儲存模型（可選也可加上 f1 條件）
                if accuracy_mean > best_accuracy:
                    best_accuracy = accuracy_mean
                    best_auroc = auroc_epoch
                    best_f1 = f1_epoch
                    best_epoch_num = epoch + 1
                    best_aug_data = all_aug_data
                    best_aug_label = all_aug_label
                    best_org_data = all_org_data
                    best_org_label = all_org_label

                    self.function_f.save_weights(
                        './moex_weight/function_f/f_model_weights_{}.h5'.format(Parameter.name))
                    self.classifier_g.save_weights(
                        './moex_weight/classifier_g/g_model_weights_{}.h5'.format(Parameter.name))

                    np.savez('./Aug_data/moex_aug_data_{}'.format(Parameter.name),
                             all_aug_data=all_aug_data.numpy(), all_aug_label=all_aug_label.numpy())
                    print("Augmented data and labels saved.")

                # 清空 list 準備下一個 epoch
                Accuracy = []
                Auroc = []
                All_true_labels = []
                All_pred_labels = []

                # 儲存歷史紀錄
                val_acc_epochs.append(accuracy_mean)
                val_auroc_epoch.append(auroc_epoch)

            # 最後整個訓練完後列印
            print("mean_acc=", tf.reduce_mean(val_acc_epochs))
            print("best acc=", best_accuracy)
            print("best auroc=", best_auroc)
            print("best f1-score=", best_f1)
            print("best epoch number = ", best_epoch_num)
            # 將 org_data 和 aug_data 合併後進行正規化
            all_data = np.concatenate([best_org_data, best_aug_data], axis=0)
            scaler = StandardScaler()
            all_data_normalized = scaler.fit_transform(all_data)

            # 獲取標籤
            labels = np.concatenate([best_org_label, best_aug_label], axis=0)

            # 執行 t-SNE 降維
            tsne = TSNE(n_components=3, random_state=42)
            tsne_results = tsne.fit_transform(all_data_normalized)

            # t-SNE 結果分開為原始數據和擴增數據
            tsne_original = tsne_results[:len(best_org_data)]
            tsne_aug = tsne_results[len(best_org_data):]

            # 根據標籤繪製圖形
            plt.figure(figsize=(10, 6))

            # 繪製原始數據
            for i, label in enumerate(np.unique(best_org_label)):
                shape = 'o' if label == 0 else '^'  # 類別 1 用圓形，類別 2 用三角形
                plt.scatter(tsne_original[best_org_label == label, 0],
                            tsne_original[best_org_label == label, 1],
                            marker=shape, color='blue', label=f'Original Class {label + 1}')

            # 繪製擴增數據
            for i, label in enumerate(np.unique(best_aug_label)):
                shape = 'o' if label == 0 else '^'  # 類別 1 用圓形，類別 2 用三角形
                plt.scatter(tsne_aug[best_aug_label == label, 0],
                            tsne_aug[best_aug_label == label, 1],
                            marker=shape, color='red', label=f'Augmented Class {label + 1}')

            plt.title("t-SNE of Normalized Original and Augmented Data")
            plt.legend(loc='best')
            plt.grid(True)

            return best_accuracy, best_auroc, best_f1
        else:
            test_accuracy, test_auroc, test_f1 = self.evaluate(val_flattened_data, val_flattened_labels)
            print("Acc = ", test_accuracy, "AUROC = ", test_auroc)

            return test_accuracy, test_auroc, test_f1

    # @tf.function
    def train_step(self, batch_data, batch_labels, cls_num):
        with tf.GradientTape() as tape:

            phi_data = self.function_f(batch_data)

            # Compute the loss
            batch_labels_reshaped = tf.reshape(batch_labels, (-1,))
            criterion = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

            output_aug = []
            moex_out = None
            # Perform moex data augmentation
            moex_prob = 1  # Set your desired moex probability
            if tf.random.uniform([]) < moex_prob:
                # B:batch_size per class
                B = batch_size / Parameter.num_classes
                swap_index = tf.random.shuffle(tf.range(batch_labels.shape[0]))
                # batch_data_combined = tf.concat([emb_cls[0], emb_cls[1]], axis=0)  # Combine the two tensors
                batch_data_input = tf.gather(phi_data, swap_index)
                batch_labels_input = tf.gather(batch_labels_reshaped, swap_index)
                x, _ = self.moex(batch_data_input, batch_labels_input, swap_index)
                moex_out = x
            else:
                batch_data_input = tf.convert_to_tensor(phi_data)
                moex_out = batch_data_input

            output_aug = self.classifier_g.call(moex_out, training=True)
            # Apply classifier_g to the augmented data
            batch_labels_swap = tf.reshape(batch_labels_input, (-1,))
            loss = criterion(batch_labels_reshaped, output_aug)
            loss_b = criterion(batch_labels_swap, output_aug)        #swap
            CE_loss = moex_lambda * loss + (1 - moex_lambda) * loss_b
        moex_label = batch_labels_input
        # Compute gradients
        trainable_variables = self.function_f.trainable_variables + self.classifier_g.trainable_variables
        gradients1 = tape.gradient(CE_loss, trainable_variables)
        # 一起更新梯度
        self.optimizer.apply_gradients(zip(gradients1, trainable_variables))
        return CE_loss, phi_data, batch_labels, moex_out, moex_label

    def moex(self, x, batch, swap_index, norm_type='pono', epsilon=1e-5):
        #swap index is the data's label, for moment exchange,每一筆資料要跟誰做moex取決於swap_index
        B = tf.shape(swap_index)[0]     #number of data for 1 batch
        T = tf.shape(batch)[0] // B     #時間步長:計算數據集中能够组成的完整批次的數量
        C = tf.shape(x)[-1]             #size of dimension

        if norm_type == 'bn':
            norm_dims = [0, 1]
        elif norm_type == 'in':
            norm_dims = [1]
        elif norm_type == 'ln':
            norm_dims = [1, 2]
        elif norm_type == 'pono':
            norm_dims = [1]     #取特徵張量的哪個維度進行norm
        elif norm_type.startswith('gpono'):
            if norm_type.startswith('gpono-d'):
                # gpono-d4 means Group PONO where each group has 4 dims
                G_dim = int(norm_type[len('gpono-d'):])
                G = C // G_dim
            else:
                # gpono4 means Group PONO with 4 groups
                G = int(norm_type[len('gpono'):])
                G_dim = C // G
            assert G * G_dim == C, f'{G} * {G_dim} != {C}'
            x = tf.reshape(x, (B, T, G, G_dim))
            norm_dims = [3]
        elif norm_type.startswith('gn'):
            if norm_type.startswith('gn-d'):
                # gn-d4 means GN where each group has 4 dims
                G_dim = int(norm_type[len('gn-d'):])
                G = C // G_dim
            else:
                # gn4 means GN with 4 groups
                G = int(norm_type[len('gn'):])
                G_dim = C // G
            assert G * G_dim == C, f'{G} * {G_dim} != {C}'
            x = tf.reshape(x, (B, T, G, G_dim))
            norm_dims = [2, 3]
        else:
            raise NotImplementedError(f'norm_type={norm_type}')
        #according to norm_dim ,calculate mean and std
        mean = tf.math.reduce_mean(x, axis=norm_dims, keepdims=True)    #jump to here
        std = tf.math.reduce_std(x, axis=norm_dims, keepdims=True)
        std = tf.sqrt(std + epsilon)
        swap_mean = tf.gather(mean, swap_index)
        swap_std = tf.gather(std, swap_index)
        scale = swap_std / std
        shift = swap_mean - mean * scale
        output = x * scale + shift
        return output, batch

    def evaluate(self, val_data, val_label):
        # 模型預測
        predictions = self(val_data, training=False)

        # 取得預測標籤
        predictions_label = tf.argmax(predictions, axis=1).numpy()

        # 轉換為 numpy array
        val_labels_array = np.array(val_label)

        # 計算準確率
        correct_count = np.sum(predictions_label == val_labels_array)
        total_count = len(val_labels_array)
        accuracy = correct_count / total_count

        # Softmax 機率
        predictions_prob = tf.keras.activations.softmax(predictions, axis=-1).numpy()
        prob_positive_class = predictions_prob[:, 1]  # 只取正類別機率

        # 計算 AUROC
        auroc = roc_auc_score(val_labels_array, prob_positive_class)

        # 計算 F1-score（需為 binary label）
        f1 = f1_score(val_labels_array, predictions_label)

        # 儲存預測結果到檔案
        prediction_output = {
            "true_labels": val_labels_array.tolist(),
            "predicted_labels": predictions_label.tolist(),
            "predicted_probabilities": prob_positive_class.tolist()
        }
        with open(f"moex_prediction_{Parameter.name}.json", "w") as f:
            json.dump(prediction_output, f, indent=2)

        # 儲存評估指標到檔案
        metric_output = {
            "test_accuracy": accuracy,
            "test_auroc": auroc,
            "test_f1_score": f1
        }
        with open(f"moex_test_metric_{Parameter.name}.json", "w") as f:
            json.dump(metric_output, f, indent=2)

        return accuracy, auroc, f1


if __name__ == '__main__':
    print(tf.__version__)
    data_path = "/home/wanling/PycharmProjects/pythonProject/{}/train/".format(Parameter.name)
    test_data_path = "/home/wanling/PycharmProjects/pythonProject/{}/test/".format(Parameter.name)
    # classes = ["no", "yes"]
    # classes = ["Covid-19", "Normal"]
    # classes = ["Demented", "NonDemented"]
    # classes = ["with_mask", "without_mask"]
    # classes = ["Positive", "Negative"]
    # classes = ['commyn', 'wesmea']
    classes = ['notsmoking', 'smoking']
    # classes = ["bleached_corals", "healthy_corals"]
    # h_data_path = "h_data"
    moex_lambda = 0.9
    num_runs = 3  # 设置运行次数
    batch_size = 64
    epoch_num = 500
    initial_learning_rate = 0.0001
    variance = []
    acc_list = []
    auroc_list = []
    f1_list = []
    for run in range(num_runs):
        print(f"Run {run + 1}:")
        data_d = data_loader().load_data(data_path)         #load data_d
        data_val = data_loader().load_data(test_data_path)  #load test data
        print(data_d.shape)

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
        model = moex_expansion(input_dim=1920, embedding_dim=units, num_classes=2)
        # 加載function_f權重
        f_model = model.function_f
        dummy_input = tf.random.normal(shape=(batch_size, 1920))  # 替換為實際的輸入形狀和數據
        _ = model.function_f(dummy_input, training=False)
        # 載入保存的權重
        f_model.load_weights('./moex_weight/function_f/f_model_weights_{}.h5'.format(Parameter.name))

        # 印出新模型的摘要
        f_model.summary()
        print("Trainable variables for function_f:")

        g_model = model.classifier_g

        dummy_input2 = tf.random.normal(shape=(batch_size, units))
        _ = model.classifier_g(dummy_input2, training=False)
        g_model.load_weights('./moex_weight/classifier_g/g_model_weights_{}.h5'.format(Parameter.name))

        acc, auroc, f1 = model.fit(data_tensor, label_tensor, val_data_tensor, val_label_tensor,
                               batch_size=batch_size, num_epochs=epoch_num, training=False)
        acc_list.append(acc)
        auroc_list.append(auroc)
        f1_list.append(f1)
    mean_accuracy = np.mean(acc_list)
    mean_auroc = np.mean(auroc_list)
    mean_f1 = np.mean(f1_list)
    print("Mean accuracy:", mean_accuracy)
    std = np.std(acc_list)
    print("std:", std)
    print("Mean auroc:", mean_auroc)
    std_au = np.std(auroc_list)
    print("std auroc:", std_au)
    std_f1 = np.std(f1_list)
    print("Mean f1 score:", mean_f1)
    print("std f1 score:", std_f1)
    # 將結果寫入文字檔
    with open(f"moex_{Parameter.name}_results.txt", "w") as file:
        file.write(f"All Runs Acc = {mean_accuracy}\n")
        file.write(f"All Runs std = {std}\n")
        file.write(f"All Runs AUROC = {mean_auroc}\n")
        file.write(f"All Runs AUROC std = {std_au}\n")
        file.write(f"All Runs F1 = {mean_f1}\n")
        file.write(f"All Runs F1 std = {std_f1}\n")
    print(f"結果已寫入 moex_{Parameter.name}_results.txt")
    print("-------------------------------------------------------------")

