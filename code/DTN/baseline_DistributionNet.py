import os
import sys
import math
import numpy as np
from glob import glob
import tensorflow as tf
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
from sklearn.metrics import roc_auc_score
from scipy.stats import multivariate_normal
from tensorflow.keras import layers, regularizers
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import BatchNormalization
from sklearn.metrics import recall_score, f1_score, precision_score

h_data_path = "h_data/"
lambda_0 = 0.5
iteration = 1000
num_steps = 40000
units = 200


# initilizers = tf.keras.initializers.GlorotNormal(seed=np.randi())
def create_initializer(h_data_path, seed=None):
    save_directory = os.path.join(h_data_path, Parameter.name, "train")
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


# torch.gather equivalent
def tf_gather(x: tf.Tensor, indices: tf.Tensor, axis: int) -> tf.Tensor:
    complete_indices = np.array(np.where(indices > -1))
    complete_indices[axis] = tf.reshape(indices, [-1])
    flat_ind = np.ravel_multi_index(tuple(complete_indices), np.array(x.shape))
    return tf.reshape(tf.gather(tf.reshape(x, [-1]), flat_ind), indices.shape)


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

class function_f(tf.keras.Model):
    def __init__(self, input_dim, embedding_dim, name):
        super().__init__()
        self.input_dim = input_dim
        self.embedding_dim = Parameter.hidden_dim
        self.d_name = name
        self.Layers = None
        self.mu = None
        self.sigma = None

    def build(self, input_shape):
        self.Layers = [
            layers.Dense(units=units,
                         activation='linear',
                         kernel_initializer=create_initializer(h_data_path),
                         bias_initializer=create_initializer(h_data_path),
                         trainable=True,
                         name='d{}_l0'.format(self.d_name)),
            # BatchNormalization(),
            layers.LeakyReLU(alpha=0.3),

            layers.Dense(units=units,
                         activation='linear',
                         kernel_initializer=create_initializer(h_data_path),
                         bias_initializer=create_initializer(h_data_path),
                         trainable=True,
                         name='d{}_l1'.format(self.d_name)),
            # BatchNormalization(),
            layers.LeakyReLU(alpha=0.3),

            layers.Dense(units=units,
                         activation='linear',
                         kernel_initializer=create_initializer(h_data_path),
                         bias_initializer=create_initializer(h_data_path),
                         trainable=True,
                         name='d{}_l2'.format(self.d_name)),
            # BatchNormalization(),
            layers.LeakyReLU(alpha=0.3),

            layers.Dense(units=units,
                         activation='linear',
                         kernel_initializer=create_initializer(h_data_path),
                         bias_initializer=create_initializer(h_data_path),
                         trainable=True,
                         name='d{}_l3'.format(self.d_name)),
            # BatchNormalization(),
            layers.LeakyReLU(alpha=0.3),
            layers.Dense(units=units,
                         activation='linear',
                         kernel_initializer=create_initializer(h_data_path),
                         bias_initializer=create_initializer(h_data_path),
                         trainable=True,
                         name='d{}_l4'.format(self.d_name)),
        ]

    # @tf.function
    def call(self, inputs, training=True, compute_mu_sig=True):
        x = inputs
        for layer in self.Layers:
            x = layer(x, training=training)

        return x
        # logits: g(f(x))--> output from classifier
        # logits2:# Sampling from sample in the loop to generate logits_tmp


class classifier_g(tf.keras.Model):
    def __init__(self, embedding_dim, num_classes, name, units=units):
        super().__init__()
        self.clf_name = Parameter.name
        self.units = units
        self.d_name = name
        self.emb_dim = embedding_dim
        self.num_classes = num_classes
        self.LayerSigma = None
        self.LayerLogits = None
        self.acc = tf.keras.metrics.SparseCategoricalAccuracy()

    def build(self, input_shape):
        self.LayerSigma = [
            layers.Dense(units=1,
                         activation='linear',
                         # bias_initializer=tf.keras.initializers.Zeros(),
                         bias_initializer=create_initializer(h_data_path),
                         trainable=True,
                         name='d{}_sig'.format(self.d_name)),

        ]
        self.LayerLogits = [
            layers.Dense(units=2,
                         activation='linear',
                         kernel_initializer=create_initializer(h_data_path),
                         bias_initializer=create_initializer(h_data_path),
                         trainable=True,
                         name='d{}_gl0'.format(self.d_name)),
        ]

    # @tf.function
    def call(self, inputs, training=False):
        x = inputs
        sig = []
        mu = tf.reduce_mean(x, axis=1, keepdims=True)  # (32,1)
        for layer in self.LayerSigma:
            sig = layer(x, training=training)  # (32, 1)
        epsilon = 1e-10
        sig = sig + epsilon
        eps = tf.random.normal(
            shape=[tf.shape(sig)[0], 1],
            mean=0.0,
            stddev=1.0,
            dtype=tf.dtypes.float32,
        )
        logits = []
        for layer in self.LayerLogits:
            logits = layer(mu, training=training)

        augmented_data = mu + sig * eps
        logits2 = []
        for layer in self.LayerLogits:
            logits2 = layer(augmented_data, training=training)

        predictions = logits + 0.1 * logits2
        return predictions, mu, sig, augmented_data


class DTN_expansion(tf.keras.Model):
    def __init__(self, input_dim, embedding_dim, num_classes):
        super(DTN_expansion, self).__init__()
        self.data_name = Parameter.name
        self.optimizer = tf.optimizers.Adam(learning_rate=initial_learning_rate)
        self.function_f = function_f(input_dim, embedding_dim, self.data_name)
        self.classifier_g = classifier_g(embedding_dim, num_classes, self.data_name)

    # @tf.function
    def call(self, inputs, training=False):
        predictions = []
        x = inputs
        x = self.function_f.call(x, training=training)
        x, _, _, _ = self.classifier_g.call(x, training=training)
        predictions.append(x)
        return predictions

    def fit(self, data_d, labels, val_data, val_labels, batch_size, num_epochs, class_num, callbacks, training):

        # 先將 (2, 80, 100) 的 train_data 轉換為一個平坦的數據集
        train_dataset = tf.data.Dataset.from_tensor_slices((data_d, labels))
        train_dataset = train_dataset.flat_map(lambda x, y: tf.data.Dataset.from_tensor_slices((x, y)))
        train_dataset = train_dataset.shuffle(500).batch(batch_size)

        #val
        test_dataset = tf.data.Dataset.from_tensor_slices((val_data, val_labels))
        test_dataset = test_dataset.flat_map(lambda x, y: tf.data.Dataset.from_tensor_slices((x, y)))
        test_dataset = test_dataset.shuffle(500).batch(batch_size)

        val_data_num = val_data.shape[1]
        val_flattened_data = tf.reshape(val_data, (2 * val_data_num, Parameter.hidden_dim))
        val_flattened_labels = tf.reshape(val_labels, (2 * val_data_num,))

        if training:
            epoch_loss, final_epoch_loss, val_acc_epochs = [], [], []
            val_auroc_epoch, val_rec_epochs = [], []
            best_accuracy = 0.0
            best_auroc = 0.0
            best_recall = 0.0
            best_precision = 0.0
            c1_h_epoch_list, c2_h_epoch_list = [], []
            best_epoch_num = 0
            acc_curve = []

            for epoch in range(num_epochs):
                correct_count = 0
                total_count = 0

                ratio = lambda_0 * epoch / num_epochs
                for batch_data, batch_labels in train_dataset:
                    batch_loss = self.train_step(batch_data, batch_labels, ratio, cls_num=2)
                    epoch_loss.append(batch_loss)

                current_epoch_loss = tf.reduce_mean(epoch_loss).numpy()

                # epoch_loss = []
                print("Epoch {}: loss={}".format(epoch + 1, current_epoch_loss))
                final_epoch_loss.append(current_epoch_loss)

                # 計算準確率
                val_predictions = []
                val_labels = []
                Accuracy = []   # all batch acc of one epoch
                F1 = []
                Auroc = []  # all batch auroc of one epoch
                accuracy_metric = tf.keras.metrics.SparseCategoricalAccuracy()
                all_preds = []
                all_true_labels = []
                for val_batch_data, val_batch_labels in test_dataset:
                    # 預測測試資料
                    predictions = self(val_batch_data)
                    predictions = predictions[0]
                    predictions_argmax = tf.argmax(predictions, axis=1).numpy()

                    # 更新準確率
                    accuracy_metric.update_state(val_batch_labels, predictions)
                    all_preds.extend(predictions_argmax)
                    all_true_labels.extend(val_batch_labels.numpy())
                    # 計算 softmax 機率
                    predictions_prob = tf.keras.activations.softmax(predictions, axis=-1)

                    # 二元 roc_auc_score 需要取正例標籤的機率
                    predictions_prob = predictions_prob[:, 1]
                    val_labels_array = np.array(val_batch_labels)
                    auroc = roc_auc_score(val_labels_array, predictions_prob)

                    Auroc.append(auroc)

                # 計算該 epoch 的準確率平均值
                accuracy_mean = accuracy_metric.result().numpy()
                accuracy_metric.reset_states()  # 重置 metric 狀態
                print("Metric epoch Accuracy:", accuracy_mean)
                acc_curve.append(accuracy_mean)

                # 計算該 epoch 的平均 AUROC
                auroc_epoch = tf.reduce_mean(Auroc)
                # 計算 F1-score
                f1 = f1_score(all_true_labels, all_preds, average='binary')
                # if the accuracy is higher, then save model
                if accuracy_mean > best_accuracy:
                    best_accuracy = accuracy_mean
                    best_f1 = f1
                    best_auroc = auroc_epoch
                    best_epoch_num = epoch + 1
                    self.function_f.save_weights('./DTN_weight/function_f/f_model_weights_{}.h5'.format(Parameter.name))
                    print("Trainable variables for function_f:")
                    self.classifier_g.save_weights('./DTN_weight/classifier_g/g_model_weights_{}.h5'.format(Parameter.name))

                Accuracy = []
                Auroc = []
                val_acc_epochs.append(accuracy_mean)
                val_auroc_epoch.append(auroc_epoch)
            mean_val_acc = tf.reduce_mean(val_acc_epochs)

            print("mean_acc=", mean_val_acc)
            print("best acc=", best_accuracy)
            print("best auroc=", best_auroc)
            print("best epoch number = ", best_epoch_num)

            plt.plot(range(num_epochs), final_epoch_loss)
            plt.title('loss Curve')
            plt.xlabel('Epoch')
            plt.ylabel('loss')
            # plt.show()
            plt.plot(range(num_epochs), acc_curve)
            plt.title('Accuracy Curve')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
            # plt.show()

            return {
                "acc": best_accuracy,
                "auroc": best_auroc,
                "f1": best_f1
            }

        else:
            test_accuracy, test_auroc, test_f1 = self.evaluate(val_flattened_data, val_flattened_labels)
            print("Acc = ", test_accuracy, "AUROC = ", test_auroc, "F1 = ", test_f1)

            return {"acc": test_accuracy,
                    "auroc": test_auroc,
                    "f1": test_f1
                    }

    # @tf.function
    def train_step(self, batch_data, batch_labels, ratio, cls_num):

        with tf.GradientTape(persistent=True) as tape:
            # phi->psi
            # batch_labels_reshaped = tf.reshape(batch_labels, (-1,))
            phi_data = self.function_f(batch_data)
            logits, mu, sig, aug_data = self.classifier_g(phi_data)

            CE = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
            CE_loss = CE(batch_labels, logits)
            ucloss = self.UCloss(mu, sig)
            # Compute gradients
            loss = CE_loss + ucloss
        gradients1 = tape.gradient(loss,
                                   self.function_f.trainable_variables + self.classifier_g.trainable_variables)
        # 一起更新梯度
        self.optimizer.apply_gradients(zip(gradients1,
                                           self.function_f.trainable_variables + self.classifier_g.trainable_variables))
        return loss

    def UCloss(self, mu, sig, weights=0.001):   #0.001
        sigma_avg = 5
        # threshold = r
        threshold = np.log(sigma_avg) + (1 + np.log(2 * np.pi)) / 2

        # # 添加一个很小的偏移量，以确保所有元素都大于 0
        # epsilon = 1e-6
        # sig += epsilon
        sig = tf.abs(sig)
        normal_distribution = tf.compat.v1.distributions.Normal(loc=mu, scale=sig)
        # losses = tf.reduce_mean(tf.nn.relu(threshold - entropy / 2048))
        entropy = normal_distribution.entropy()
        return tf.reduce_mean(tf.nn.relu(threshold - entropy / batch_size)) * weights
        # return losses * weights

    def evaluate(self, val_data, val_label):
        predictions = self(val_data, training=False)
        # calculate acc
        correct_count = 0
        total_count = 0

        # test labels
        val_labels_array = np.array(val_label)
        # get max index as predict label
        predictions = predictions[0]
        predictions_new = tf.argmax(predictions, axis=1).numpy()

        # calculate test acc
        for j in range(len(predictions_new)):  # len(predictions_new[i])=16
            if np.all(predictions_new[j] == val_label[j]):
                correct_count += 1
            total_count += 1

        # 然后进行 softmax 处理
        predictions_prob = tf.keras.activations.softmax(predictions, axis=-1)
        # 二元 roc_auc_score 需要取正例標籤的機率
        predictions_prob = predictions_prob[:, 1]
        auroc = roc_auc_score(val_labels_array, predictions_prob)
        # one batch accuracy
        accuracy = correct_count / total_count
        f1 = f1_score(val_labels_array, predictions_new)
        return accuracy, auroc, f1


def exp_decay_learning_rate(initial_learning_rate, global_step, decay_steps, decay_factor, staircase=True):
    """
    配置指數衰減的學習率

    Args:
    - initial_learning_rate: 初始學習率
    - global_step: 全局步數，通常是模型訓練的總步數
    - decay_steps: 學習率衰減的步數
    - decay_factor: 學習率衰減的因子
    - staircase: 是否使用梯形函數，預設為 True

    Returns:
    - 學習率的 TensorFlow 張量
    """
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate,
        decay_steps,
        decay_factor,
        staircase=staircase
    )
    learning_rate = lr_schedule(global_step)
    return learning_rate


if __name__ == '__main__':
    print(tf.__version__)
    tf.keras.backend.clear_session()
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

    num_runs = 3  # 设置运行次数
    variance = []
    acc_list = []
    auroc_list = []
    f1_list = []
    # if 0.001, loss will get bigger
    initial_learning_rate = 0.0001   #0.0001
    epochs = 3000   # 訓練週期
    batch_size = 64
    lr_decay_rate = 0.1
    global_step = 100
    decay_steps = 100
    decay_factor = 0.9
    # changing_lr = [150, 225, 300]
    # changing_lr = [80, 120]
    for run in range(num_runs):

        print(f"Run {run + 1}:")

        data_d = data_loader().load_data(data_path)  # load data_d
        data_val = data_loader().load_data(test_data_path)  # load test data

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
        print(data_d.shape)
        # 轉換 labels 為 Tensor
        label_tensor = tf.convert_to_tensor(labels, dtype=tf.int32)
        val_label_tensor = tf.convert_to_tensor(val_labels, dtype=tf.int32)

        name = Parameter.name
        model = DTN_expansion(input_dim=1920, embedding_dim=units, num_classes=2)

        # 加載function_f權重
        f_model = model.function_f
        dummy_input = tf.random.normal(shape=(batch_size, 1920))  # 替換為實際的輸入形狀和數據
        _ = model.function_f(dummy_input, training=False)
        # 載入保存的權重
        f_model.load_weights('./DTN_weight/function_f/f_model_weights_{}.h5'.format(Parameter.name))
        # 印出新模型的摘要
        f_model.summary()
        print("Trainable variables for function_f:")
        for var in model.function_f.trainable_variables:
            print(f"{var.name}: {var.numpy()}, dtype={var.dtype}")

        g_model = model.classifier_g

        dummy_input2 = tf.random.normal(shape=(batch_size, units))
        _ = model.classifier_g(dummy_input2, training=False)
        g_model.load_weights('./DTN_weight/classifier_g/g_model_weights_{}.h5'.format(Parameter.name))
        g_model.summary()
        print("Trainable variables for classifier_g.LayerLogits:")
        for var in model.classifier_g.LayerLogits.trainable_variables:
            print(f"{var.name}: {var.numpy()}, dtype={var.dtype}")

        optimizer = tf.keras.optimizers.Adam(learning_rate=initial_learning_rate)
        lr_scheduler = tf.keras.callbacks.LearningRateScheduler(exp_decay_learning_rate(initial_learning_rate, global_step, decay_steps, decay_factor))

        model.compile(optimizer=optimizer)

        result= model.fit(data_tensor, label_tensor, val_data_tensor, val_label_tensor,
                               batch_size=batch_size, num_epochs=epochs,
                               class_num=Parameter.num_classes, callbacks=[lr_scheduler],
                               training=False)

        acc, auroc, f1 = result["acc"], result["auroc"], result["f1"]

        acc_list.append(acc)  # 保存测试准确率
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
    print("Mean f1:", mean_f1)
    print("std f1:", std_f1)
    # 將結果寫入文字檔
    with open(f"DTN_{Parameter.name}_results.txt", "w") as file:
        file.write(f"Mean Acc = {mean_accuracy}\n")
        file.write(f"ACC std = {std}\n")
        file.write(f"Mean AUROC = {mean_auroc}\n")
        file.write(f"AUROC std = {std_au}\n")
        file.write(f"Mean F1 = {mean_f1}\n")
        file.write(f"F1 std = {std_f1}\n")
    print(f"結果已寫入 DTN_{Parameter.name}_results.txt")
    print("-------------------------------------------------------------")


