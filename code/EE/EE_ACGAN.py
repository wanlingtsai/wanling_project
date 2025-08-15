import tensorflow as tf
# physical_devices = tf.config.list_physical_devices('GPU')
# print("Num GPUs:", len(physical_devices))
import os
import sys
import math
import time
import json
import psutil
import pynvml
import numpy as np
from glob import glob
import tensorflow as tf
from tensorflow import keras
from EE import embedding_expansion
from tensorflow.keras import layers, activations
from baseline_DistributionNet import DTN_expansion
from sklearn.metrics import roc_auc_score, f1_score
from tensorflow.keras.layers import BatchNormalization

h_data_path = "h_data/"
units = 200
_b_acc = None


class Parameter(object):
    name = "alzheimer"
    num_classes = 2
    hidden_dim = 1920

def print_memory_usage():
    process = psutil.Process()
    print(f"Memory Usage: {process.memory_info().rss / 1024 / 1024:.2f} MB")



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


def create_initializer(h_data_path, seed=None):
    save_directory = os.path.join(h_data_path, Parameter.name, "train")
    return tf.keras.initializers.GlorotNormal(seed=seed)


class acgan(tf.keras.Model):
    def __init__(self, latent_dim, label_dim, name):
        super(acgan, self).__init__()
        self.latent_dim = latent_dim
        self.label_dim = label_dim
        self.d_name = name
        self.label_shape = Parameter.num_classes
        self.function_f = embedding_expansion.function_f(input_dim=Parameter.hidden_dim, embedding_dim=latent_dim, name=Parameter.name)
        self.g = self.get_generator(latent_dim, label_dim, name=Parameter.name)
        self.d = self.get_discriminator(latent_dim, name=Parameter.name)
        self.loss_bool = keras.losses.BinaryCrossentropy(from_logits=True, reduction="none")
        self.loss_class = keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction="none")
        self.train_counter = tf.Variable(0, trainable=False, dtype=tf.int32)
        self.opt_d = keras.optimizers.Adam(initial_learning_rate, beta_1=0.5)
        self.opt_g = keras.optimizers.Adam(initial_learning_rate, beta_1=0.5)

    def call(self, target_labels, training=False, mask=None):
        # 產生 noise 向量
        batch_size = tf.shape(target_labels)[0]
        noise = tf.random.normal((batch_size, self.latent_dim))

        # 保險起見：確保 target_labels 是 Tensor 且為 int32
        if isinstance(target_labels, np.ndarray):
            target_labels = tf.convert_to_tensor(target_labels, dtype=tf.int32)
        elif target_labels.dtype != tf.int32:
            target_labels = tf.cast(target_labels, dtype=tf.int32)

        # 修正 shape，保證是 1D (例如 [64] 而非 [64, 1] 或 [64, 2])
        target_labels = tf.reshape(target_labels, [-1])  # 強制壓平成一維

        # 傳入 Generator
        return self.g([noise, target_labels], training=training)
        # [noise, target_labels]: list:2 (64, 1920),(64,)
        # return (64,1920)

    class get_generator(tf.keras.Model):
        def __init__(self, latent_dim, label, name):
            super().__init__()
            self.LayerG = None
            self.label_shape = label    #2
            self.latent_dim = latent_dim    #1920
            self.d_name = name
            self.build(input_shape=(None, (self.label_shape + self.latent_dim)))

        def build(self, input_shape):
            self.LayerG = [
                layers.Dense(units=units,
                             activation='linear',
                             kernel_initializer=create_initializer(h_data_path),
                             bias_initializer=create_initializer(h_data_path),
                             name='g{}_l0'.format(self.d_name)),
                # BatchNormalization(),
                layers.LeakyReLU(alpha=0.2),

                layers.Dense(units=units,
                             activation='linear',
                             kernel_initializer=create_initializer(h_data_path),
                             bias_initializer=create_initializer(h_data_path),
                             name='g{}_l1'.format(self.d_name)),
                # BatchNormalization(),
                layers.LeakyReLU(alpha=0.2),

                layers.Dense(units=units,
                             activation='linear',
                             kernel_initializer=create_initializer(h_data_path),
                             bias_initializer=create_initializer(h_data_path),
                             name='g{}_l2'.format(self.d_name)),
                ]

        # @tf.function
        def call(self, inputs, training=True):
            # inputs are noise and class:1920,2
            noise, label = inputs
            # Concatenate noise and label(1920+2)
            label = tf.cast(label, tf.int32)
            label_onehot = tf.one_hot(label, depth=self.label_shape)
            combined_input = tf.concat((noise, label_onehot), axis=1)
            x = combined_input  #(32,1922)
            for layer in self.LayerG:
                x = layer(x, training=training)

            return x

    class get_discriminator(tf.keras.Model):
        def __init__(self, latent_dim, name):
            super().__init__()
            self.LayerD = None
            self.LayerD_dense = None
            self.classifier_g = embedding_expansion.classifier_g(latent_dim, num_classes=Parameter.num_classes, name=name, units=units)
            self.Layers = self.classifier_g.Layers
            self.d_name = name
            self.latent_dim = latent_dim    #1920
            # self.build(input_shape=(None, latent_dim))

        def build(self, input_shape):
            self.LayerD = [
                layers.Dense(units=units,
                             activation='linear',
                             kernel_initializer=create_initializer(h_data_path),
                             bias_initializer=create_initializer(h_data_path),
                             name='LayerD{}_l0'.format(self.d_name)),
                # BatchNormalization(),
                layers.LeakyReLU(alpha=0.2),

                layers.Dense(units=Parameter.num_classes+1,
                             activation='linear',
                             kernel_initializer=create_initializer(h_data_path),
                             bias_initializer=create_initializer(h_data_path),
                             name='LayerD{}_l1'.format(self.d_name)),
                # BatchNormalization(),
                layers.LeakyReLU(alpha=0.2),]
            #auxilary classifier

            self.LayerD_dense = [
                layers.Dense(units=1,
                             activation='sigmoid',
                             kernel_initializer=create_initializer(h_data_path),
                             bias_initializer=create_initializer(h_data_path),
                             name='LayerD_dense{}_l1_sigmoid'.format(self.d_name)),

            ]


        # @tf.function
        def call(self, inputs, training=True):
            x = inputs
            # inputs only image(vector), no label
            for layer in self.LayerD:
                x = layer(x, training=training)

            # Split x into three parts
            x1 = x[:, 0:1]  # First element

            # o_bool : first element, o_class : second and third elements
            o_bool = self.LayerD_dense[0](x1, training=training)
            x_2 = inputs
            o_class = self.classifier_g.call(x_2, training=training)
            return o_bool, o_class  #(32,1),(32,2)

    def train_g(self, random_img_label):    #random_img_label:(32,)

        # generated_img = self.g(random_img_label, training=True)
        #d_label:(32,1)
        # d_label是一個全是1的向量，它的形狀是(batch_size, 1)，其中
        # batch_size是一批訓練數據的大小。這個向量表示對應著生成器輸出的樣本的標籤，
        # 即「這是真實的樣本」。
        # 在訓練過程中，生成器的目標是迷惑判別器，使其認為生成器生成的樣本是真實的。
        # 因此，在計算生成器的損失（loss）時，你使用了生成器生成的樣本經過判別器的預測結果
        # （pred_bool）和期望的標籤d_label之間的損失。
        d_label = tf.ones((len(random_img_label), 1), tf.float32)
        with tf.GradientTape() as tape:
            g_img = self.call(random_img_label, training=True)
            pred_bool, pred_class = self.d.call(g_img, training=False)  #(64,1)/(64,2)
            loss_bool = self.loss_bool(d_label, pred_bool) #(64,)
            loss_class = self.loss_class(random_img_label, pred_class)  #(64,)
            loss = tf.reduce_mean(loss_bool + loss_class)
        grads = tape.gradient(loss, self.g.trainable_variables)
        self.opt_g.apply_gradients(zip(grads, self.g.trainable_variables))
        return loss, g_img, binary_accuracy(d_label, pred_bool)

    def train_d(self, img, img_label, label):
        with tf.GradientTape(persistent=True) as tape:
            pred_bool, pred_class = self.d.call(img, training=True)
            loss_bool = self.loss_bool(label, pred_bool)
            loss_class = self.loss_class(img_label, pred_class)
            loss = tf.reduce_mean(loss_bool + loss_class)
        grads = tape.gradient(loss, self.d.Layers.trainable_variables)
        self.opt_d.apply_gradients(zip(grads, self.d.trainable_variables))
        return loss, binary_accuracy(label, pred_bool), class_accuracy(img_label, pred_class)

    # @tf.function
    def train_step(self, real_data, real_labels, ep):
        """
        real_img:2(<tf.Tensor: shape=(16, 1920)/real_img_label:2 tf.Tensor([0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.], shape=(16,), dtype=float32)
        real_data (32,1920)->funtion_f(32,200)->ACGAN(64,200)->function_f but input shape doesn't equal to function_f input
        solution2: don't train function_f after loaded
        random_img_label:(32,)
        """
        # phi_data = self.function_f(real_data)
        # 確保模型權重已經被創建
        _ = self.d(real_data)
        dummy_input = [
            tf.random.normal((batch_size, units)),
            tf.random.uniform((batch_size,), minval=0, maxval=2, dtype=tf.int32)
        ]
        _ = self.g(dummy_input)


        if ep == 0:
            g_ratio = 3
            d_ratio = 1
            for i in range(g_ratio):
                random_img_label = tf.convert_to_tensor(np.random.randint(0, 2, len(real_data)), dtype=tf.int32)
                g_loss, g_img, g_bool_loss = self.train_g(random_img_label)

            img = tf.concat((real_data, g_img[:len(g_img) // 2]), axis=0)
            random_img_label = tf.cast(random_img_label[:len(g_img) // 2], dtype=tf.int32)
            img_label = tf.concat((real_labels, random_img_label[:len(g_img) // 2]), axis=0)
            d_label = tf.ones((len(real_labels), 1), tf.float32)
            for j in range(d_ratio):
                d_loss, d_bool_acc, d_class_acc = self.train_d(real_data, real_labels, d_label)

        else:
            g_ratio = 3
            d_ratio = 1
            for i in range(g_ratio):
                random_img_label = tf.convert_to_tensor(np.random.randint(0, 2, len(real_data)), dtype=tf.int32)

                g_loss, g_img, g_bool_loss = self.train_g(random_img_label)
            img = tf.concat((real_data, g_img[:len(g_img) // 2]), axis=0)

            random_img_label = tf.cast(random_img_label[:len(g_img) // 2], dtype=tf.int32)
            img_label = tf.concat((real_labels, random_img_label[:len(g_img) // 2]), axis=0)
            d_label = tf.concat((tf.ones((len(real_labels), 1), tf.float32), tf.zeros((len(g_img) // 2, 1), tf.float32)),
                                axis=0)
            for j in range(d_ratio):
                d_loss, d_bool_acc, d_class_acc = self.train_d(img, img_label, d_label)

        return g_img, d_loss, d_bool_acc, d_class_acc, g_loss, g_bool_loss, random_img_label

    def ganfit(self, data_d, labels, val_data, val_labels, batch_size, num_epochs, training):
        t0 = time.time()
        data_num = data_d.shape[1]
        flattened_data = tf.reshape(data_d, (Parameter.num_classes * data_num, units))
        flattened_labels = tf.reshape(labels, (Parameter.num_classes * data_num,))
        dataset = tf.data.Dataset.from_tensor_slices((flattened_data, flattened_labels))
        buffer_size = 3000
        dataset = dataset.shuffle(buffer_size)
        ds = dataset.batch(batch_size)

        val_data_num = val_data.shape[1]
        val_flattened_data = tf.reshape(val_data, (Parameter.num_classes * val_data_num, units))
        val_flattened_labels = tf.reshape(val_labels, (Parameter.num_classes * val_data_num,))
        val_dataset = tf.data.Dataset.from_tensor_slices((val_flattened_data, val_flattened_labels))
        buffer_size = 3000
        val_dataset = val_dataset.shuffle(buffer_size)
        val_ds = val_dataset.batch(batch_size)

        d_bool_acc, g_bool_acc = 0.0, 0.0
        d_bool_accu, g_bool_accu= [], []
        t0 = time.time()
        best_acc, epoch_acc, batch_acc = 0.0, 0.0, 0.0
        all_acc = []
        batch_acc_list, epoch_acc_list = [], []
        best_epoch, val_best_epoch, gd_best_epoch = 0, 0, 0
        mean_acc, gd_best_acc, val_best_acc = 0.0, 0.0, 0.0
        g_dist, d_dist = 100.0, 100.0
        best_acc_g, best_acc_d = 0.0, 0.0
        val_ep_auroc = 0.0
        val_best_auroc = 0.0
        val_Accuracy, val_Auroc, val_F1 = [], [], []
        if training:
            # self.load_all_weights("./ACGAN_weight")
            for ep in range(num_epochs):
                for batch_data, batch_labels in ds:
                    g_img, d_loss, d_bool_acc, d_class_acc, g_loss, g_bool_loss, g_img_label = self.train_step(batch_data, batch_labels, ep)
                    t1 = time.time()
                    # every epochs print training info
                    print("ep={} | time={:.1f} | d_acc={:.2f} | d_cls_acc={:.2f} | g_acc={:.2f} | d_loss={:.2f} | g_loss={:.2f}".format(
                        ep+1, t1 - t0, d_bool_acc.numpy(), d_class_acc, g_bool_loss.numpy(), d_loss.numpy(), g_loss.numpy(), ))

                    batch_acc_list.append(d_class_acc)
                    g_bool_accu.append(g_bool_loss.numpy())
                    d_bool_accu.append(d_bool_acc)

                epoch_acc = tf.reduce_mean(batch_acc_list)
                g_bool_acc = tf.reduce_mean(g_bool_accu)
                d_bool_acc = tf.reduce_mean(d_bool_accu)
                batch_acc_list = []
                all_acc.append(epoch_acc)
                if g_bool_acc-0.5 < g_dist and d_bool_acc-0.5 < d_dist:
                    g_dist = g_bool_acc-0.5
                    d_dist = d_bool_acc-0.5
                    gd_best_epoch = ep + 1
                    gd_best_acc = epoch_acc

                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_epoch = ep + 1

                for val_batch_data, val_batch_labels in val_ds:
                    bool_pred, class_pred = self.d(val_batch_data)
                    correct_count = 0
                    total_count = 0

                    # 測試标签
                    val_labels_array = np.array(val_batch_labels)
                    # 取得预测标签
                    predictions_new = tf.argmax(class_pred, axis=1).numpy()

                    # 計算準確率
                    for j in range(len(predictions_new)):  # len(predictions_new)==batch size
                        if predictions_new[j] == val_labels_array[j]:
                            correct_count += 1
                        total_count += 1

                    # 計算 F1 Score
                    f1 = f1_score(val_labels_array, predictions_new, average='macro')
                    val_F1.append(f1)

                    # 計算AUROC
                    predictions_prob = tf.keras.activations.softmax(class_pred, axis=-1)
                    predictions_prob = predictions_prob[:, 1]
                    auroc = roc_auc_score(val_labels_array, predictions_prob)
                    val_Auroc.append(auroc)

                    # 計算 Accuracy
                    accuracy = correct_count / total_count
                    val_Accuracy.append(accuracy)

                # 計算 epoch 級別的 metrics
                val_ep_acc = tf.reduce_mean(val_Accuracy)
                val_ep_auroc = tf.reduce_mean(val_Auroc)
                val_ep_f1 = tf.reduce_mean(val_F1)

                if val_ep_acc > val_best_acc:
                    val_best_acc = val_ep_acc
                    val_best_auroc = val_ep_auroc
                    val_best_f1 = val_ep_f1
                    val_best_epoch = ep + 1
                    print("g:", g_bool_acc, "| d:", d_bool_acc)
                    best_acc_g = g_bool_acc
                    best_acc_d = d_bool_acc
                    self.d.save_weights(
                        './ACGAN_weight/EE_d_weight/EE_val_d_weights_{}.h5'.format(Parameter.name))

                    self.g.save_weights(
                        './ACGAN_weight/EE_g_weight/EE_val_g_weights_{}.h5'.format(Parameter.name))

            self.save_all_weights("./ACGAN_weight")
            mean_acc = tf.reduce_mean(all_acc)
            print("train mean acc = ", mean_acc)
            print("train best epoch =", best_epoch)
            print("train best acc =", best_acc)
            print("train gd best epoch =", gd_best_epoch)
            print("train gd best acc =", gd_best_acc)
            print("test best epoch =", val_best_epoch)
            print("test best acc =", val_best_acc)
            print("test best F1 =", val_best_f1)
            print("when acc best, g=", best_acc_g, "d=", best_acc_d)

            return {
                "acc": val_best_acc,
                "auroc": val_best_auroc,
                "f1": val_best_f1
            }
        else:
            test_accuracy, test_auroc, test_f1 = self.evaluate(val_flattened_data, val_flattened_labels)
            print("Acc = ", test_accuracy, "AUROC = ", test_auroc)

            return {
                "acc": test_accuracy,
                "auroc": test_auroc,
                "f1": test_f1
            }


    def evaluate(self, val_data, val_label):
        dummy_input = tf.random.normal([1, units])
        _ = self.d(dummy_input)  # 讓模型build起來

        self.d.load_weights('./ACGAN_weight/EE_d_weight/EE_val_d_weights_{}.h5'.format(Parameter.name))
        # 模型預測
        pre_bool, pre_cls = self.d(val_data, training=False)
        # 取得預測標籤
        predictions_label = tf.argmax(pre_cls, axis=1).numpy()

        # 轉換為 numpy array
        val_labels_array = np.array(val_label)

        # 計算準確率
        correct_count = np.sum(predictions_label == val_labels_array)
        total_count = len(val_labels_array)
        accuracy = correct_count / total_count

        # Softmax 機率
        predictions_prob = tf.keras.activations.softmax(pre_cls, axis=-1).numpy()
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
        with open(f"ISDA_ACGAN_prediction_{Parameter.name}.json", "w") as f:
            json.dump(prediction_output, f, indent=2)

        return accuracy, auroc, f1

    def save_all_weights(self, path):
        """儲存生成器與判別器的權重"""
        # 儲存 Generator 權重
        self.g.save_weights(f"{path}/EE_generator_weights_{Parameter.name}.h5")
        # 儲存 Discriminator 權重
        self.d.save_weights(f"{path}/EE_discriminator_weights_{Parameter.name}.h5")
        print("生成器與判別器的權重已儲存。")

    def load_all_weights(self, path):
        """載入生成器與判別器的權重"""
        # 建立 dummy input 以初始化模型
        dummy_noise = tf.random.normal(shape=(1, self.g.latent_dim))
        dummy_label = tf.constant([0], dtype=tf.int32)  # 假設 label 是單一維度
        _ = self.g([dummy_noise, dummy_label], training=False)

        dummy_input = tf.random.normal(shape=(1, self.d.latent_dim))
        _ = self.d(dummy_input, training=False)

        # 載入權重
        self.g.load_weights(f"{path}/EE_generator_weights_{Parameter.name}.h5")
        self.d.load_weights(f"{path}/EE_discriminator_weights_{Parameter.name}.h5")
        print("生成器與判別器的權重已載入。")


def cosine_decay(epoch):
    global initial_learning_rate
    if epoch in changing_lr:
        initial_learning_rate *= lr_decay_rate
    lr = 0.5 * initial_learning_rate * (1 + math.cos(math.pi * epoch / gan_epochs))
    return lr


def binary_accuracy(label, pred, threshold=0.5):
    global _b_acc
    if _b_acc is None:
        _b_acc = tf.keras.metrics.BinaryAccuracy()
    _b_acc.reset_states()
    # Applying threshold to predictions
    pred_binary = tf.cast(pred > threshold, tf.float32)

    _b_acc.update_state(label, pred_binary)
    return _b_acc.result()



def class_accuracy(label, pred):
    labels = tf.cast(tf.squeeze(label), dtype=tf.int64)
    accuracy_metric = tf.keras.metrics.SparseCategoricalAccuracy()
    accuracy_metric.update_state(labels, pred)
    accuracy = accuracy_metric.result().numpy()
    return accuracy


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
    variance = []
    acc_list = []
    auroc_list = []
    f1_list = []
    initial_learning_rate = 0.0001
    gan_epochs = 200
    batch_size = 64
    lr_decay_rate = 0.1
    changing_lr = [80, 120]
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

        # 轉換 labels 為 Tensor
        label_tensor = tf.convert_to_tensor(labels, dtype=tf.int32)
        val_label_tensor = tf.convert_to_tensor(val_labels, dtype=tf.int32)

        name = Parameter.name

        gan_model = acgan(latent_dim=units, label_dim=Parameter.num_classes, name=Parameter.name)

        #加載function_f權重
        f_model = gan_model.function_f
        dummy_input = tf.random.normal(shape=(batch_size, 1920))  # 替換為實際的輸入形狀和數據
        _ = gan_model.function_f(dummy_input, training=False)
        # 載入保存的權重
        f_model.load_weights('./EE_weight/function_f/f_model_weights_{}.h5'.format(Parameter.name))

        # 印出新模型的摘要
        f_model.summary()
        print("Trainable variables for function_f:")
        for var in gan_model.function_f.trainable_variables:
            print(f"{var.name}: {var.numpy()}, dtype={var.dtype}")

        d_model = gan_model.d.classifier_g

        dummy_input = tf.random.normal(shape=(batch_size, 200))
        _ = gan_model.d.classifier_g(dummy_input, training=False)
        d_model.load_weights('./EE_weight/classifier_g/g_model_weights_{}.h5'.format(Parameter.name))
        print("Trainable variables for classifier_g.Layers:")
        for var in gan_model.d.classifier_g.Layers.trainable_variables:
            print(f"{var.name}: {var.numpy()}, dtype={var.dtype}")

        # 載入保存的數據
        file_path = './Aug_data/EE_aug_data_{}.npz'.format(Parameter.name)  # 記得使用正確的文件名
        data = np.load(file_path)

        # 取得 augmented data 和 labels
        loaded_aug_data = data['all_aug_data']
        loaded_aug_label = data['all_aug_label']

        # 確保 label 為 0 和 1 的資料分別存儲
        class_0_data = loaded_aug_data[loaded_aug_label == 0]
        class_1_data = loaded_aug_data[loaded_aug_label == 1]

        class_0_labels = loaded_aug_label[loaded_aug_label == 0]
        class_1_labels = loaded_aug_label[loaded_aug_label == 1]

        # 將兩類資料進行 stack，形成 (2, 39, 200) 和 (2, 39,)
        stacked_data = np.stack((class_0_data, class_1_data))
        stacked_labels = np.stack((class_0_labels, class_1_labels))

        # 檢查新的形狀
        print("Stacked data shape:", stacked_data.shape)  # 應該是 (2, 39, 200)
        print("Stacked labels shape:", stacked_labels.shape)  # 應該是 (2, 39,)

        lr_scheduler = tf.keras.callbacks.LearningRateScheduler(cosine_decay)

        phi_data = []
        for i in range(Parameter.num_classes):
            phi_data.append(gan_model.function_f(data_tensor[i]))
        # two list two class
        phi_data_tensor = tf.convert_to_tensor(phi_data)
        print(phi_data_tensor.shape)  # (2,78,200)

        test_phi_data = []
        for i in range(Parameter.num_classes):
            test_phi_data.append(gan_model.function_f(val_data_tensor[i]))
        # two list two class
        test_phi_data_tensor = tf.convert_to_tensor(test_phi_data)
        print(test_phi_data_tensor.shape)  # (2,78,200)
        train_data = tf.concat([phi_data_tensor, stacked_data], axis=1)
        train_label = tf.concat([label_tensor, stacked_labels], axis=1)

        result = gan_model.ganfit(train_data, train_label, test_phi_data_tensor, val_label_tensor,
                               batch_size=batch_size, num_epochs=gan_epochs,
                                  training=True)

        acc, auroc, f1 = result["acc"], result["auroc"], result["f1"]

        acc_list.append(acc)  # save test accuracy
        auroc_list.append(auroc)
        f1_list.append(f1)
    mean_accuracy = np.mean(acc_list)
    mean_auroc = np.mean(auroc_list)
    mean_f1 = np.mean(f1_list)
    f1_std = np.std(f1_list)
    print("All runs Mean accuracy:", mean_accuracy)
    std = np.std(acc_list)
    print("std:", std)
    print("Mean auroc:", mean_auroc)
    auroc_std = np.std(auroc_list)
    print("std auroc:", auroc_std)
    print("Mean F1:", mean_f1)
    print("F1 std:", f1_std)
    # 將結果寫入文字檔
    with open(f"EE_ACGAN_{Parameter.name}_results.txt", "w") as file:
        file.write(f"All Runs Acc = {mean_accuracy}\n")
        file.write(f"All Runs std = {std}\n")
        file.write(f"All Runs AUROC = {mean_auroc}\n")
        file.write(f"All Runs AUROC std = {auroc_std}\n")
        file.write(f"All Runs F1 = {mean_f1}\n")
        file.write(f"All Runs F1 std = {f1_std}\n")
    print(f"結果已寫入 EE_ACGAN_{Parameter.name}_results.txt")
    print("-------------------------------------------------------------")


