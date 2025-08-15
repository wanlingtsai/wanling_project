import os
import sys
import math
import time
import numpy as np
from glob import glob
import tensorflow as tf
from tensorflow import keras
from matplotlib import pyplot as plt
from sklearn.metrics import roc_auc_score
from tensorflow.keras import layers, Model
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.optimizers.schedules import CosineDecay
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


def tf_gather(x: tf.Tensor, indices: tf.Tensor, axis: int) -> tf.Tensor:
    complete_indices = np.array(np.where(indices > -1))
    complete_indices[axis] = tf.reshape(indices, [-1])
    flat_ind = np.ravel_multi_index(tuple(complete_indices), np.array(x.shape))
    return tf.reshape(tf.gather(tf.reshape(x, [-1]), flat_ind), indices.shape)


class ISDA_expansion(tf.keras.Model):
    def __init__(self, input_dim, embedding_dim, num_classes):
        super(ISDA_expansion, self).__init__()
        # self.optimizer = tf.optimizers.Adam(learning_rate=0.0001)
        self.function_f = self.function_f(input_dim, embedding_dim, name)
        self.classifier_g = self.classifier_g(embedding_dim, num_classes, name)

    @tf.function
    def call(self, inputs, training=False):
        predictions = []
        x = inputs
        x = self.function_f.call(x, training=training)
        x = self.classifier_g.call(x, training=training)
        predictions.append(x)
        return predictions

    class function_f(tf.keras.Model):
        def __init__(self, input_dim, embedding_dim, name):
            super().__init__()
            self.input_dim = input_dim
            self.embedding_dim = embedding_dim
            self.d_name = name
            self.Layers = None
            self.build(input_shape=(None, input_dim))
            # self.inputs = inputs
        def build(self, input_shape):
            self.Layers = [
                layers.Dense(units=units,
                             activation='linear',
                             kernel_initializer=create_initializer(h_data_path),
                             bias_initializer=create_initializer(h_data_path),
                             name='f{}_l0'.format(self.d_name)),
                # BatchNormalization(),
                layers.LeakyReLU(alpha=0.3),

                layers.Dense(units=units,
                             activation='linear',
                             kernel_initializer=create_initializer(h_data_path),
                             bias_initializer=create_initializer(h_data_path),
                             name='f{}_l1'.format(self.d_name)),
                # BatchNormalization(),
                layers.LeakyReLU(alpha=0.3),

                layers.Dense(units=units,
                             activation='linear',
                             kernel_initializer=create_initializer(h_data_path),
                             bias_initializer=create_initializer(h_data_path),
                             name='f{}_l2'.format(self.d_name)),
                # BatchNormalization(),
                layers.LeakyReLU(alpha=0.3),

                layers.Dense(units=units,
                             activation='linear',
                             kernel_initializer=create_initializer(h_data_path),
                             bias_initializer=create_initializer(h_data_path),
                             name='f{}_l3'.format(self.d_name)),
                # BatchNormalization(),
                layers.LeakyReLU(alpha=0.3),
                #
                layers.Dense(units=units,
                             activation='linear',
                             kernel_initializer=create_initializer(h_data_path),
                             bias_initializer=create_initializer(h_data_path),
                             name='f{}_l4'.format(self.d_name))
            ]
        # @tf.function
        def call(self, inputs, training=False):
            x = inputs
            # print("input:", x)
            for layer in self.Layers:
                x = layer(x, training=training)
            # print("out:", x)
            return x

    class classifier_g(tf.keras.Model):
        def __init__(self, embedding_dim, num_classes, name, units=units):
            super().__init__()
            self.clf_name = Parameter.name
            self.units = units
            self.d_name = name
            self.emb_dim = embedding_dim
            self.num_classes = num_classes
            self.fc = None
            self.build(input_shape=(None, embedding_dim))       #None means any size
            # self.acc = tf.keras.metrics.SparseCategoricalAccuracy()

        def build(self, input_shape):
            self.fc = [
                layers.Dense(units=self.num_classes,
                             activation='linear',  # Change activation function if needed
                             kernel_initializer=create_initializer(h_data_path),
                             bias_initializer=create_initializer(h_data_path),
                             name='d{}_fully_connected'.format(self.d_name))]

        # @tf.function
        def call(self, inputs, training=False):
            # print("call input:", type(inputs))
            x = inputs
            for layer in self.fc:
                x = layer(x, training=training)
            return x

    def fit(self, data_d, labels, val_data, val_labels, batch_size, num_epochs, class_num, callbacks, training):
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

            val_acc_epochs = []
            best_accuracy = 0.0
            best_auroc = 0.0
            isda_criterion = ISDALoss(self.function_f.embedding_dim, class_num)
            fc = self.classifier_g
            best_epoch_num = 0
            for epoch in range(num_epochs):
                epoch_loss = []
                final_epoch_loss = []
                # print("Epoch", epoch + 1)
                t0 = time.time()

                ratio = lambda_0 * (epoch+1) / num_epochs
                for batch_data, batch_labels in train_dataset:

                    batch_loss = self.train_step(batch_data, batch_labels, fc, isda_criterion, ratio, cls_num=2)
                    epoch_loss.append(batch_loss)
                t1 = time.time()
                t2 = t1 - t0
                # print("Epoch", epoch+1, ":", t2, "sec")
                current_epoch_loss = tf.reduce_mean(epoch_loss).numpy()
                print("Epoch", epoch+1, "loss:", current_epoch_loss)

                # print("Epoch {}: loss={}".format(epoch + 1, current_epoch_loss))
                final_epoch_loss.append(current_epoch_loss)

                # 計算準確率
                correct_predictions = 0
                total_predictions = 0

                Auroc = []
                all_preds = []
                all_labels = []
                accuracy_metric = tf.keras.metrics.SparseCategoricalAccuracy()

                for val_batch_data, val_batch_labels in test_dataset:
                    # 預測測試資料
                    predictions = self(val_batch_data)
                    predictions = predictions[0]
                    predictions_argmax = tf.argmax(predictions, axis=1).numpy()
                    # 收集預測與真實標籤（for F1-score）
                    all_preds.extend(predictions_argmax)
                    all_labels.extend(
                        val_batch_labels.numpy() if hasattr(val_batch_labels, 'numpy') else val_batch_labels)

                    correct_predictions += np.sum(np.argmax(predictions, axis=1) == val_batch_labels)
                    # correct_predictions += np.sum(np.argmax(predictions, axis=1).numpy() == val_batch_labels.numpy())

                    total_predictions += len(val_batch_labels)
                    # batch_accuracy = np.mean(predictions_argmax == val_batch_labels.numpy())
                    # 更新準確率
                    accuracy_metric.update_state(val_batch_labels, predictions)

                    # 計算 softmax 機率
                    predictions_prob = tf.keras.activations.softmax(predictions, axis=-1)

                    # 二元 roc_auc_score 需要取正例標籤的機率
                    predictions_prob = predictions_prob[:, 1]
                    val_labels_array = np.array(val_batch_labels)
                    auroc = roc_auc_score(val_labels_array, predictions_prob)

                    Auroc.append(auroc)

                # 計算該 epoch 的準確率平均值
                test_epoch_accuracy = correct_predictions / total_predictions
                accuracy_mean = accuracy_metric.result().numpy()
                # print("Manual epoch Accuracy:", test_epoch_accuracy)
                # print("Metric epoch Accuracy:", accuracy_mean)
                accuracy_metric.reset_states()  # 重置 metric 狀態

                # 計算該 epoch 的平均 AUROC
                auroc_epoch = tf.reduce_mean(Auroc)
                f1 = f1_score(all_labels, all_preds, average='binary')  # 若是 multi-class 可以改成 'macro'

                # accuracy higher then save model
                if accuracy_mean > best_accuracy:
                    best_accuracy = accuracy_mean
                    best_f1 = f1
                    best_auroc = auroc_epoch
                    best_epoch_num = epoch + 1

                    self.function_f.save_weights('./ISDA_weight/function_f/f_model_weights_{}.h5'.format(Parameter.name))

                    self.classifier_g.save_weights(
                        './ISDA_weight/classifier_g/g_model_weights_{}.h5'.format(Parameter.name))

                val_acc_epochs.append(accuracy_mean)
            mean_val_acc = tf.reduce_mean(val_acc_epochs)

            print("best epoch num = ", best_epoch_num)
            print("mean_acc=", mean_val_acc)
            print("best_acc=", best_accuracy)

            plt.plot(range(num_epochs), val_acc_epochs)
            plt.title('Epoch accuracy')
            plt.xlabel('Epoch')
            plt.ylabel('accuracy')
            plt.savefig('ISDA_acc')
            # plt.show()

            return {
                "acc": best_accuracy,
                "auroc": best_auroc,
                "f1": best_f1
            }
        else:
            test_accuracy, test_auroc, test_f1 = self.evaluate(val_flattened_data, val_flattened_labels)
            print("Acc = ", test_accuracy, "AUROC = ", test_auroc)

            return {
                "acc": test_accuracy,
                "auroc": test_auroc,
                "f1": test_f1
            }

    # @tf.function
    def train_step(self, batch_data, batch_labels, fc, isda_criterion, ratio, cls_num):

        with tf.GradientTape() as tape:
            output_aug = []
            #No.4 parameter is target_var,means the true label
            isda_loss, output_aug = isda_criterion(self.function_f, fc, batch_data, batch_labels, ratio)

            CE_loss = isda_loss

        # Compute gradients
        trainable_variables = self.function_f.trainable_variables + self.classifier_g.trainable_variables
        gradients1 = tape.gradient(CE_loss, trainable_variables)
        # print("gradient:", gradients1)
        # 一起更新梯度
        self.optimizer.apply_gradients(zip(gradients1, trainable_variables))
        return CE_loss

    def evaluate(self, val_data, val_label):
        # 初始化 metric
        accuracy_metric = tf.keras.metrics.SparseCategoricalAccuracy()

        # 預測測試資料
        predictions = self(val_data, training=False)
        predictions = predictions[0]  # 取出 logits

        # 更新準確率 metric
        accuracy_metric.update_state(val_label, predictions)

        # 計算 softmax 機率
        predictions_prob = tf.keras.activations.softmax(predictions, axis=-1)

        # 二元 roc_auc_score 需要取正例標籤的機率
        predictions_prob = predictions_prob[:, 1]
        val_labels_array = np.array(val_label)
        auroc = roc_auc_score(val_labels_array, predictions_prob)

        # 從 metric 中提取準確率
        accuracy = accuracy_metric.result().numpy()

        # 重置 metric 狀態（以便下次使用）
        accuracy_metric.reset_states()
        predictions_argmax = tf.argmax(predictions, axis=1).numpy()
        f1 = f1_score(val_labels_array, predictions_argmax, average='binary')  # 若為 multi-class 改 'macro'

        return accuracy, auroc, f1


class EstimatorCV(tf.Module):
    def __init__(self, feature_num, class_num):
        self.class_num = class_num
        self.CoVariance = tf.zeros((class_num, feature_num, feature_num), dtype=tf.float32)
        self.Ave = tf.zeros((class_num, feature_num), dtype=tf.float32)
        self.Amount = tf.zeros(class_num, dtype=tf.float32)

    def update_CV(self, features, labels):
        N = tf.shape(features)[0]   # N is number of data
        C = self.class_num  # c is class num
        A = tf.shape(features)[1]   # A is feature num

        NxCxFeatures = tf.tile(tf.expand_dims(features, 1), (1, C, 1))

        labels = tf.cast(labels, dtype=tf.int32)

        onehot = tf.one_hot(labels, depth=C, dtype=tf.float32)


        NxCxA_onehot = tf.tile(onehot[:, :, tf.newaxis], [1, 1, A])
        features_by_sort = NxCxFeatures * NxCxA_onehot

        Amount_CxA = tf.reduce_sum(NxCxA_onehot, axis=0)
        Amount_CxA = tf.where(Amount_CxA == 0, 1, Amount_CxA)
        expanded_Amount_CxA = tf.expand_dims(Amount_CxA, -1)  # 形狀變成 (2, 1920, 1)
        expanded_Amount_CxA = expanded_Amount_CxA * tf.ones((1, 1, A), dtype=tf.float32)
        # print("expand_amount:", expanded_Amount_CxA)
        ave_CxA = tf.reduce_sum(features_by_sort, axis=0) / Amount_CxA
        # print("ave_CxA:", tf.tile((tf.expand_dims(ave_CxA, 0)), [N, 1, 1]))
        var_temp = features_by_sort - tf.tile((tf.expand_dims(ave_CxA, 0)), [N, 1, 1]) * NxCxA_onehot
        transposed_var_temp_1 = tf.transpose(var_temp, perm=[1, 2, 0])
        transposed_var_temp_2 = tf.transpose(var_temp, perm=[1, 0, 2])
        var_temp = tf.linalg.matmul(transposed_var_temp_1, transposed_var_temp_2) / expanded_Amount_CxA

        sum_weight_CV = tf.reduce_sum(onehot, axis=0, keepdims=True)
        sum_weight_CV = tf.squeeze(sum_weight_CV, axis=0)
        # sum_weight_CV = tf.transpose(sum_weight_CV)  # 将形状从 (1, 2) 调整为 (2, 1)
        expanded_sum_weight_CV = tf.expand_dims(sum_weight_CV, axis=1)  # 在第一个维度上添加新维度

        expanded_sum_weight_CV = tf.expand_dims(expanded_sum_weight_CV, axis=1)
        # print(expanded_sum_weight_CV.shape)
        # expanded_sum_weight_CV = tf.expand_dims(expanded_sum_weight_CV, axis=1)  # 在第二个维度上添加新维度
        expanded_sum_weight_CV = expanded_sum_weight_CV * tf.ones([C, A, A], dtype=tf.float32)  # 扩展为目标形状

        sum_weight_AV = tf.reduce_sum(onehot, axis=0)
        sum_weight_AV = tf.expand_dims(sum_weight_AV, axis=1)  # 在第二个维度上添加新维度，形状变为 (C, 1)
        sum_weight_AV = tf.tile(sum_weight_AV, multiples=[1, A])  # 在第二个维度上进行广播，形状变为 (C, A)

        # sum_weight_AV = tf.tile(sum_weight_AV, [C, 1])  # 广播成 (C, A) 形状
        expanded_amount = tf.expand_dims(tf.expand_dims(self.Amount, -1), -1)
        expanded_amount = tf.tile(expanded_amount, multiples=[1, A, A])

        weight_CV = expanded_sum_weight_CV / (expanded_sum_weight_CV + expanded_amount)
        weight_CV = tf.where(tf.math.is_nan(weight_CV), 0.0, weight_CV)

        weight_AV = sum_weight_AV / (sum_weight_AV + tf.expand_dims(self.Amount, -1))
        weight_AV = tf.where(tf.math.is_nan(weight_AV), 0.0, weight_AV)


        # print("weight_CV:", weight_CV.shape)
        additional_CV = weight_CV * (1 - weight_CV) * tf.linalg.matmul(
            (self.Ave - ave_CxA)[:, :, tf.newaxis],  # 计算差值向量的形状为 (C, A, 1)
            (self.Ave - ave_CxA)[:, tf.newaxis, :]
        )
        self.CoVariance = tf.stop_gradient(self.CoVariance * (1 - weight_CV) + var_temp * weight_CV) + tf.stop_gradient(additional_CV)
        self.Ave = tf.stop_gradient(self.Ave * (1 - weight_AV) + ave_CxA * weight_AV)
        self.Amount += tf.reduce_sum(onehot, axis=0)


class ISDALoss(tf.Module):
    def __init__(self, feature_num, class_num):
        self.estimator = EstimatorCV(feature_num, class_num)
        self.class_num = class_num
        self.cross_entropy = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True)

    def isda_aug(self, fc, features, y, labels, cv_matrix, ratio):
        N = tf.shape(features)[0]
        C = self.class_num
        A = tf.shape(features)[1]

        weight_m = fc.trainable_variables[0]
        weight_m_transposed = tf.transpose(weight_m, perm=[1, 0])
        # print("y:", y)
        NxW_ij = tf.tile(tf.expand_dims(weight_m_transposed, 0), (N, 1, 1))

        labels_kj = tf.expand_dims(tf.expand_dims(labels, -1), -1)
        labels_kj = tf.tile(labels_kj, multiples=[1, C, A])

        # 使用 tf_gather
        NxW_kj = tf_gather(NxW_ij, labels_kj, axis=1)       #shape should be (N,C,A)

        labels = tf.cast(labels, tf.int32)
        CV_temp = tf.gather(cv_matrix, labels)
        # print("CV_temp", CV_temp.shape)
        #\frac{\lambda}{2}\left(\boldsymbol{w}{j}^{T}-\boldsymbol{w}{y_{i}}^{T}\right) \Sigma_{y_{i}}\left(\boldsymbol{w}{j}-\boldsymbol{w}{y_{i}}\right)}}\right)
        # sigma2 = ratio * tf.linalg.matmul(
        #     tf.transpose(NxW_ij - NxW_kj, perm=[0, 2, 1]),
        #     CV_temp
        # ) @ (NxW_ij - NxW_kj)
        # 计算矩阵乘法
        matmul_result = tf.linalg.matmul(NxW_ij - NxW_kj, CV_temp)

        # 计算矩阵的转置
        transposed_NxW_ij_minus_NxW_kj = tf.transpose(NxW_ij - NxW_kj, perm=[0, 2, 1])
        # transposed_NxW_ij_minus_NxW_kj_permuted = tf.transpose(transposed_NxW_ij_minus_NxW_kj, perm=[0, 2, 1])

        # 计算最终的结果
        sigma2 = ratio * tf.linalg.matmul(matmul_result, transposed_NxW_ij_minus_NxW_kj)
        sigma2 = sigma2 * tf.linalg.diag(tf.ones(C))
        #y:predicted value
        aug_result = y + 0.5 * tf.reduce_sum(sigma2, axis=1)

        return aug_result

    def __call__(self, model, fc, x, target_x, ratio):
        features = model(x)
        y = fc(features)
        self.estimator.update_CV(tf.stop_gradient(features), target_x)
        isda_aug_y = self.isda_aug(fc, features, y, target_x, tf.stop_gradient(self.estimator.CoVariance), ratio)
        target_x = tf.cast(target_x, tf.int32)
        loss = self.cross_entropy(target_x, isda_aug_y)
        return loss, y


def cosine_decay(epoch):
    global initial_learning_rate
    if epoch in changing_lr:
        initial_learning_rate *= lr_decay_rate
    lr = 0.5 * initial_learning_rate * (1 + math.cos(math.pi * epoch / epochs))
    return lr


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
    # h_data_path = "h_data"
    num_runs = 3  # 设置运行次数
    variance = []
    acc_list = []
    auroc_list, f1_list = [], []
    initial_learning_rate = 0.001  #0.001
    epochs =  7000  # 訓練週期
    batch_size = 128
    lr_decay_rate = 0.1
    # changing_lr = [150, 225, 300]
    changing_lr = [80, 120]
    for run in range(num_runs):

        print(f"Run {run + 1}:")

        data_d = data_loader().load_data(data_path)         #load data_d
        data_val = data_loader().load_data(test_data_path)  #load test data
        # print(data_d.shape)

        class1_labels = [0] * (data_d.shape[1])  # make labels:covid->1, normal->0
        class2_labels = [1] * (data_d.shape[1])
        labels = np.vstack((class1_labels, class2_labels))  # shape:(2,500)
        # print("label:", labels.shape)

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
        optimizer = tf.keras.optimizers.SGD(learning_rate=initial_learning_rate, momentum=0.9, nesterov=True)
        lr_scheduler = tf.keras.callbacks.LearningRateScheduler(cosine_decay)
        model = ISDA_expansion(input_dim=1920, embedding_dim=units, num_classes=2)
        # 加載function_f權重
        f_model = model.function_f
        dummy_input = tf.random.normal(shape=(batch_size, 1920))  # 替換為實際的輸入形狀和數據
        _ = model.function_f(dummy_input, training=False)
        # 載入保存的權重
        f_model.load_weights('./ISDA_weight/function_f/f_model_weights_{}.h5'.format(Parameter.name))

        # 印出新模型的摘要
        f_model.summary()

        g_model = model.classifier_g

        dummy_input = tf.random.normal(shape=(batch_size, units))
        _ = model.classifier_g(dummy_input, training=False)
        g_model.load_weights('./ISDA_weight/classifier_g/g_model_weights_{}.h5'.format(Parameter.name))
        print("Trainable variables for classifier_g:")
        for var in model.classifier_g.fc.trainable_variables:
            print(f"{var.name}: {var.numpy()}, dtype={var.dtype}")
        model.compile(optimizer=optimizer)

        result = model.fit(data_tensor, label_tensor, val_data_tensor, val_label_tensor,
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
    f1_std = np.std(f1_list)
    print("Mean accuracy:", mean_accuracy)
    std = np.std(acc_list)
    print("std:", std)
    print("Mean auroc:", mean_auroc)
    auroc_std = np.std(auroc_list)
    print("std auroc:", auroc_std)
    print("Mean F1:", mean_f1)
    print("std F1:", f1_std)
    # 將結果寫入文字檔
    with open(f"ISDA_{Parameter.name}_results.txt", "w") as file:
        file.write(f"All Runs Acc = {mean_accuracy}\n")
        file.write(f"All Runs std = {std}\n")
        file.write(f"All Runs AUROC = {mean_auroc}\n")
        file.write(f"All Runs AUROC std = {auroc_std}\n")
        file.write(f"All Runs F1 = {mean_f1}\n")
        file.write(f"All Runs F1 std = {f1_std}\n")
    print(f"結果已寫入 ISDA_{Parameter.name}_results.txt")
    print("-------------------------------------------------------------")


