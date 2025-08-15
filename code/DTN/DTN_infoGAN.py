import os
import psutil
import pynvml
import pickle
import numpy as np
from glob import glob
import tensorflow as tf
from tensorflow import keras
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, f1_score
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras import layers, Model, activations, optimizers
from tensorflow.keras.losses import BinaryCrossentropy, MeanSquaredError
from baseline_DistributionNet import function_f, classifier_g
h_data_path = "h_data/"
units = 200
categorical_dim = 2
continuous_dim = 5
class Parameter(object):
    name = "Corals"
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


def create_initializer(h_data_path, seed=None):
    save_directory = os.path.join(h_data_path, Parameter.name, "train")
    return tf.keras.initializers.GlorotNormal(seed=seed)


class InfoGAN(Model):
    def __init__(self, input_dim, latent_dim, categorical_dim, continuous_dim, info_reg_coeff=1):
        super(InfoGAN, self).__init__()
        self.input_dim = latent_dim
        self.latent_dim = latent_dim
        self.categorical_dim = categorical_dim
        self.continuous_dim = continuous_dim
        self.info_reg_coeff = info_reg_coeff
        self.train_D_times = 1
        self.train_GQ_times = 20
        # Generator model
        self.generator = self.build_generator()

        # Discriminator model
        self.discriminator = self.build_discriminator(input_dim, Parameter.num_classes, Parameter.name)

        # Auxiliary model for mutual information estimation
        self.q_network = self.build_q_network(self.categorical_dim, self.continuous_dim)

        # Optimizers
        self.generator_optimizer = optimizers.Adam(learning_rate=0.0005, beta_1=0.5)
        self.discriminator_optimizer = optimizers.Adam(learning_rate=0.0001, beta_1=0.5)

        # Loss functions
        self.binary_crossentropy = BinaryCrossentropy(from_logits=True)
        self.mse = MeanSquaredError()

    class build_generator(tf.keras.Model):
        def __init__(self):
            super().__init__()

        def build(self, input_shape):
            self.LayerG = [
                layers.Dense(units=units,
                             activation='linear',
                             kernel_initializer=create_initializer(h_data_path),
                             bias_initializer=create_initializer(h_data_path),
                             name='g{}_l0'.format(Parameter.name)),
                # BatchNormalization(),
                layers.LeakyReLU(alpha=0.3),

                layers.Dense(units=units,
                             activation='linear',
                             kernel_initializer=create_initializer(h_data_path),
                             bias_initializer=create_initializer(h_data_path),
                             name='g{}_l1'.format(Parameter.name)),
                # BatchNormalization(),
                layers.LeakyReLU(alpha=0.3),

                layers.Dense(units=units,
                             activation='linear',
                             kernel_initializer=create_initializer(h_data_path),
                             bias_initializer=create_initializer(h_data_path),
                             name='g{}_l2'.format(Parameter.name)),
            ]

        def call(self, inputs, training=True):
            x = inputs
            for layer in self.LayerG:
                x = layer(x, training=training)

            return x

    class build_discriminator(tf.keras.Model):
        def __init__(self, embedding_dim, num_classes, name, units=units):
            super().__init__()
            self.clf_name = Parameter.name
            self.units = units
            self.d_name = name
            self.emb_dim = embedding_dim
            self.num_classes = num_classes
            self.acc = tf.keras.metrics.SparseCategoricalAccuracy()

        def build(self, input_shape):
            self.Layers = [
                layers.Dense(units=units,
                             activation='linear',
                             kernel_initializer=create_initializer(h_data_path),
                             bias_initializer=create_initializer(h_data_path),
                             name='g{}_l0'.format(self.d_name)),

                layers.Dense(units=units,
                             activation='linear',
                             kernel_initializer=create_initializer(h_data_path),
                             bias_initializer=create_initializer(h_data_path),
                             name='g{}_l1'.format(self.d_name)),

                layers.Dense(units=2,
                             activation='linear',
                             kernel_initializer=create_initializer(h_data_path),
                             bias_initializer=create_initializer(h_data_path),
                             name='g{}_l2'.format(self.d_name)), ]

        @tf.function
        def call(self, inputs, training=False):
            # print("call input:", type(inputs))
            x = inputs
            for layer in self.Layers:
                x = layer(x, training=training)
            return x


    class build_q_network(tf.keras.Model):
        def __init__(self, categorical_dim, continuous_dim):
            super().__init__()
            self.units = units
            self.Layers = None
            self.categorical_dim = categorical_dim
            self.continuous_dim = continuous_dim
            self.build(input_shape=(None, units))  # Assuming input shape is (None, units)

        def build(self, input_shape):
            self.Layers = [
                layers.Dense(units=self.units,
                             activation='linear',
                             kernel_initializer=create_initializer(h_data_path),
                             bias_initializer=create_initializer(h_data_path),
                             name='q_net_l0'),
                layers.LeakyReLU(alpha=0.3),
                layers.Dense(units=self.units,
                             activation='linear',
                             kernel_initializer=create_initializer(h_data_path),
                             bias_initializer=create_initializer(h_data_path),
                             name='q_net_l1'),
                layers.LeakyReLU(alpha=0.3),
                layers.Dense(units=self.categorical_dim + self.continuous_dim,
                             activation='linear',
                             kernel_initializer=create_initializer(h_data_path),
                             bias_initializer=create_initializer(h_data_path),
                             name='q_net_l2')
            ]

        @tf.function
        def call(self, inputs, training=False):
            x = inputs
            for layer in self.Layers:
                x = layer(x, training=training)
            return x

    def call(self, inputs, training=None, mask=None):
        return self.generator(inputs)

    def generate_latent_inputs(self, batch_size):
        # c_cat是一个 one-hot 編碼向量，表示離散潛在代碼的類別,用於控制生成數據的離散特徵。例如，在MNIST數據集中，這可以是數字類别（0到9）
        # c_cont是一個連續向量，表示連續潛在代碼的數值,用於控制生成數據的連續特徵。這些特徵是連續變化的，例如圖片的旋轉角度、顏色強度等。
        z = tf.random.normal(shape=(batch_size, self.latent_dim))
        c_cat = tf.one_hot(tf.random.uniform(shape=(batch_size,), minval=0, maxval=self.categorical_dim, dtype=tf.int32), self.categorical_dim)
        c_cont = tf.random.uniform(shape=(batch_size, self.continuous_dim), minval=-1, maxval=1)
        return tf.concat([z, c_cat, c_cont], axis=1)

    def train_step(self, real_features):
        batch_size = tf.shape(real_features)[0]

        for _ in range(self.train_D_times):
            # Generate latent inputs and fake features
            latent_inputs = self.generate_latent_inputs(batch_size)
            fake_features = self.generator(latent_inputs)

            with tf.GradientTape() as disc_tape:
                real_logits = self.discriminator(real_features)
                fake_logits = self.discriminator(fake_features)
                disc_loss = self.binary_crossentropy(tf.ones_like(real_logits), real_logits) + \
                            self.binary_crossentropy(tf.zeros_like(fake_logits), fake_logits)

            disc_grads = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)
            self.discriminator_optimizer.apply_gradients(zip(disc_grads, self.discriminator.trainable_variables))

            # Train generator and Q-network multiple times
        for _ in range(self.train_GQ_times):
            latent_inputs = self.generate_latent_inputs(batch_size)
            with tf.GradientTape() as gen_tape, tf.GradientTape() as q_tape:
                fake_features = self.generator(latent_inputs)
                fake_logits = self.discriminator(fake_features)
                q_logits = self.q_network(fake_features)

                gen_loss = self.binary_crossentropy(tf.ones_like(fake_logits), fake_logits)

                cat_logits, cont_logits = tf.split(q_logits, [self.categorical_dim, self.continuous_dim], axis=1)
                cat_labels, cont_labels = tf.split(latent_inputs[:, self.latent_dim:],
                                                   [self.categorical_dim, self.continuous_dim], axis=1)

                q_cat_loss = self.binary_crossentropy(cat_labels, cat_logits)
                q_cont_loss = self.mse(cont_labels, cont_logits)

                q_loss = q_cat_loss + q_cont_loss
                total_gen_loss = gen_loss + self.info_reg_coeff * q_loss

            gen_grads = gen_tape.gradient(total_gen_loss, self.generator.trainable_variables)
            q_grads = q_tape.gradient(q_loss, self.q_network.trainable_variables)

            self.generator_optimizer.apply_gradients(zip(gen_grads, self.generator.trainable_variables))
            self.generator_optimizer.apply_gradients(zip(q_grads, self.q_network.trainable_variables))

        return {'disc_loss': disc_loss, 'gen_loss': gen_loss, 'q_loss': q_loss}

    def fit(self, dataset, batch_size, epochs ):
        generated_data = None
        train_ds = tf.data.Dataset.from_tensor_slices(dataset)
        train_ds = train_ds.batch(batch_size)
        for epoch in range(epochs):
            for batch_data in train_ds:
                losses = self.train_step(batch_data)
            print(f'Epoch {epoch+1}, Disc Loss: {losses["disc_loss"]}, Gen Loss: {losses["gen_loss"]}, Q Loss: {losses["q_loss"]}')

            if epoch == epochs - 1:
                batch_size = next(iter(dataset)).shape[0]  # Get batch size from dataset
                latent_inputs = self.generate_latent_inputs(batch_size)
                generated_data = self.generator(latent_inputs)

        return generated_data

class TotalModel(tf.keras.Model):
    def __init__(self, input_dim, latent_dim, categorical_dim, continuous_dim):
        super(TotalModel, self).__init__()
        self.infogan1 = InfoGAN(input_dim, latent_dim, categorical_dim, continuous_dim)
        self.infogan2 = InfoGAN(input_dim, latent_dim, categorical_dim, continuous_dim)
        self.function_f = function_f(input_dim=Parameter.hidden_dim, embedding_dim=units, name=Parameter.name)
        self.classifier = classifier_g(embedding_dim=units, num_classes=Parameter.num_classes, name=Parameter.name, units=units)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=initial_learning_rate)

    def call(self, inputs, training=False):
        predictions = []
        x = inputs
        x = self.function_f.call(x, training=training)
        x, _, _ = self.classifier.call(x, training=training)
        predictions.append(x)
        return predictions

    def fit(self, train_data, test_data, test_label, batch_size, num_epochs, cls_epochs, training):
        test_data_c1 = test_data[0]
        test_data_c2 = test_data[1]
        test_label_c1 = test_label[0]
        test_label_c2 = test_label[1]
        train_data_c1 = train_data[0]
        train_data_c2 = train_data[1]
        # test dataset
        test_dataset_c1 = tf.data.Dataset.from_tensor_slices((test_data_c1, test_label_c1))
        test_dataset_c2 = tf.data.Dataset.from_tensor_slices((test_data_c2, test_label_c2))

        # Combine datasets for both classes
        combined_test_dataset = test_dataset_c1.concatenate(test_dataset_c2)
        combined_test_dataset = combined_test_dataset.shuffle(buffer_size=3000)
        combined_test_dataset = combined_test_dataset.batch(batch_size)
        if training:
            final_org_data, final_org_labels = [], []
            final_gen_data, final_gen_labels = [], []
            temp_org_data, temp_org_labels = None, None
            temp_gen_data, temp_gen_labels = None, None
            gan_model1 = self.infogan1
            gan_model2 = self.infogan2

            total_best_acc = 0.0
            total_best_auroc = 0.0
            training_loss, test_epoch_accuracies = [], []
            best_acc, best_auroc, Epoch_loss, test_epoch_accuracy = 0.0, 0.0, 0.0, 0.0
            # train gan total times
            ratio = total_gan_epoch / gan_epochs
            # self.load_all_weights("./infoGAN_weight")
            for i in range(int(ratio)):
                # 分類生成資料
                gen_c1 = gan_model1.fit(train_data_c1, batch_size, num_epochs)
                gen_c2 = gan_model2.fit(train_data_c2, batch_size, num_epochs)

                gen_data_c1 = tf.data.Dataset.from_tensor_slices(gen_c1)
                gen_data_c2 = tf.data.Dataset.from_tensor_slices(gen_c2)

                # 將原始資料轉換為 Dataset
                orig_data_c1 = tf.data.Dataset.from_tensor_slices(train_data_c1)
                orig_data_c2 = tf.data.Dataset.from_tensor_slices(train_data_c2)

                # 創建標籤，假設 gen_data_c1 對應到 0，gen_data_c2 對應到 1
                gen_labels_c1 = tf.data.Dataset.from_tensor_slices(tf.constant([0] * len(gen_c1)))
                gen_labels_c2 = tf.data.Dataset.from_tensor_slices(tf.constant([1] * len(gen_c2)))

                # 將生成的資料和原始資料合併
                combined_data_c1 = orig_data_c1.concatenate(gen_data_c1)
                combined_data_c2 = orig_data_c2.concatenate(gen_data_c2)

                # 原始資料的標籤
                orig_labels_c1 = tf.data.Dataset.from_tensor_slices(tf.constant([0] * len(train_data_c1)))
                orig_labels_c2 = tf.data.Dataset.from_tensor_slices(tf.constant([1] * len(train_data_c2)))

                # 將生成資料的標籤和原始資料的標籤合併
                combined_labels_c1 = orig_labels_c1.concatenate(gen_labels_c1)
                combined_labels_c2 = orig_labels_c2.concatenate(gen_labels_c2)

                # 將資料和標籤合併到一起
                combined_dataset_c1 = tf.data.Dataset.zip((combined_data_c1, combined_labels_c1))
                combined_dataset_c2 = tf.data.Dataset.zip((combined_data_c2, combined_labels_c2))

                # 合併兩個類別的 Dataset
                combined_dataset = combined_dataset_c1.concatenate(combined_dataset_c2)

                # 洗牌資料
                combined_dataset = combined_dataset.shuffle(buffer_size=3000)
                combined_dataset = combined_dataset.batch(batch_size)
                total_training_acc = []
                for epoch in range(cls_epochs):
                    epoch_loss = []
                    epoch_auroc = 0.0

                    correct_predictions, total_predictions = 0.0, 0.0
                    for batch_data, batch_label in combined_dataset:
                        # Train the classifier
                        loss = self.train_classifier(batch_data, batch_label)
                        epoch_loss.append(loss)

                    Epoch_loss = tf.reduce_mean(epoch_loss)
                    training_loss.append(Epoch_loss)
                    correct_predictions = 0
                    total_predictions = 0
                    epoch_auroc = []
                    all_preds = []
                    all_labels = []
                    epoch_auroc = []
                    accuracy_metric = tf.keras.metrics.SparseCategoricalAccuracy()

                    for test_batch_data, test_batch_label in combined_test_dataset:
                        predictions, _, _, _ = self.classifier(test_batch_data)
                        accuracy_metric.update_state(test_batch_label, predictions)
                        # 預測類別
                        pred_labels = np.argmax(predictions, axis=1)

                        # 累計所有預測與標籤，用來算 F1
                        all_preds.extend(pred_labels)
                        all_labels.extend(test_batch_label)
                        predictions_prob = tf.keras.activations.softmax(predictions, axis=-1)
                        # 二元 roc_auc_score 需要取正例標籤的機率
                        predictions_prob = predictions_prob[:, 1]
                        auroc = roc_auc_score(test_batch_label, predictions_prob)
                        epoch_auroc.append(auroc)
                    test_epoch_accuracy = accuracy_metric.result().numpy()
                    accuracy_metric.reset_states()  # 重置 metric 狀態

                    test_epoch_accuracies.append(test_epoch_accuracy)
                    epoch_auroc = tf.reduce_mean(epoch_auroc)
                    epoch_f1 = f1_score(all_labels, all_preds, average='binary')  # 或 'macro' / 'weighted'
                    print(
                        f"Epoch {epoch + 1}/{cls_epochs}, Loss: {Epoch_loss:.4f}, Test Accuracy: {test_epoch_accuracy:.4f}, Test AUROC: {epoch_auroc:.4f}, Test F1: {epoch_f1:.4f}")

                    if test_epoch_accuracy > best_acc:
                        best_acc = test_epoch_accuracy
                        best_epoch = epoch
                        best_auroc = epoch_auroc
                        best_f1 = epoch_f1
                        temp_org_data = tf.concat([train_data_c1, train_data_c2], axis=0)
                        temp_org_labels = tf.constant([0] * len(train_data_c1) + [1] * len(train_data_c2))

                        temp_gen_data = tf.concat([gen_c1, gen_c2], axis=0)
                        temp_gen_labels = tf.constant([0] * len(gen_c1) + [1] * len(gen_c2))

                        self.classifier.save_weights(
                            f'./infoGAN_weight/classifier_g/DTN_temp_best_weights_{Parameter.name}.h5')

                        # self.classifier.save_weights(
                        #     './infoGAN_weight/classifier_g/DTN_g_model_weights_{}.h5'.format(Parameter.name))

                if best_acc > total_best_acc:
                    total_best_acc = best_acc
                    total_best_auroc = best_auroc
                    total_best_f1 = best_f1
                    # ✅ 確認覆蓋時機只有這一輪是真的比以前都好
                    final_org_data = temp_org_data
                    final_org_labels = temp_org_labels
                    final_gen_data = temp_gen_data
                    final_gen_labels = temp_gen_labels
                    self.classifier.load_weights(
                        f'./infoGAN_weight/classifier_g/DTN_temp_best_weights_{Parameter.name}.h5')
                    self.classifier.save_weights(
                        './infoGAN_weight/classifier_g/DTN_FinalBest_g_model_weights_{}.h5'.format(Parameter.name))

                print("best acc = ", best_acc)
                print("best epoch = ", best_epoch+1)
            self.save_all_weights("./infoGAN_weight")
            self.plot_tsne_3d(final_org_data, final_org_labels, final_gen_data, final_gen_labels,
                              tf.concat([test_data[0], test_data[1]], axis=0).numpy(),
                              tf.concat([test_label[0], test_label[1]], axis=0).numpy())

            return {
                "acc": total_best_acc,
                "auroc": total_best_auroc,
                "f1": total_best_f1
            }
        else:
            self.classifier.load_weights(
                './infoGAN_weight/classifier_g/DTN_FinalBest_g_model_weights_{}.h5'.format(Parameter.name))
            correct_predictions = 0
            total_predictions = 0
            eval_acc = 0.0
            eval_auroc = 0.0
            all_auroc = []
            all_preds = []
            all_labels = []
            for test_batch_data, test_batch_label in combined_test_dataset:
                predictions, _, _, _ = self.classifier(test_batch_data)
                pred_labels = np.argmax(predictions, axis=1)
                correct_predictions += np.sum(np.argmax(predictions, axis=1) == test_batch_label)
                total_predictions += len(test_batch_label)
                predictions_prob = tf.keras.activations.softmax(predictions, axis=-1)
                # 二元 roc_auc_score 需要取正例標籤的機率
                predictions_prob = predictions_prob[:, 1]
                # 收集所有 batch 的預測結果與標籤
                all_preds.extend(pred_labels)
                all_labels.extend(test_batch_label)
                auroc = roc_auc_score(test_batch_label, predictions_prob)
                all_auroc.append(auroc)

            eval_acc = correct_predictions / total_predictions
            eval_auroc = tf.reduce_mean(all_auroc)
            eval_f1 = f1_score(all_labels, all_preds)
            return {
                "acc": eval_acc,
                "auroc": eval_auroc,
                "f1": eval_f1
            }

    def save_all_weights(self, path):
        """儲存TotalModel下所有InfoGAN和其他子模型的權重"""
        self.infogan1.generator.save_weights(f"{path}/infogan1_generator_weights_{Parameter.name}.h5")
        self.infogan1.discriminator.save_weights(f"{path}/infogan1_discriminator_weights_{Parameter.name}.h5")
        self.infogan1.q_network.save_weights(f"{path}/infogan1_q_network_weights_{Parameter.name}.h5")

        self.infogan2.generator.save_weights(f"{path}/infogan2_generator_weights_{Parameter.name}.h5")
        self.infogan2.discriminator.save_weights(f"{path}/infogan2_discriminator_weights_{Parameter.name}.h5")
        self.infogan2.q_network.save_weights(f"{path}/infogan2_q_network_weights_{Parameter.name}.h5")
        print("所有模型的權重已儲存。")

    def load_all_weights(self, path):
        """載入TotalModel下所有InfoGAN和其他子模型的權重"""
        dummy_input = tf.random.normal(shape=(batch_size, units))  # 替换为实际的输入形状
        dummy_input2 = tf.random.normal(shape=(batch_size, units + categorical_dim + continuous_dim))
        _ = self.infogan1.generator(dummy_input2, training=False)
        _ = self.infogan1.discriminator(dummy_input, training=False)
        _ = self.infogan1.q_network(dummy_input, training=False)

        self.infogan1.generator.load_weights(f"{path}/infogan1_generator_weights_{Parameter.name}.h5")
        self.infogan1.discriminator.load_weights(f"{path}/infogan1_discriminator_weights_{Parameter.name}.h5")
        self.infogan1.q_network.load_weights(f"{path}/infogan1_q_network_weights_{Parameter.name}.h5")

        _ = self.infogan2.generator(dummy_input2, training=False)
        _ = self.infogan2.discriminator(dummy_input, training=False)
        _ = self.infogan2.q_network(dummy_input, training=False)

        self.infogan2.generator.load_weights(f"{path}/infogan2_generator_weights_{Parameter.name}.h5")
        self.infogan2.discriminator.load_weights(f"{path}/infogan2_discriminator_weights_{Parameter.name}.h5")
        self.infogan2.q_network.load_weights(f"{path}/infogan2_q_network_weights_{Parameter.name}.h5")
        print("所有模型的權重已載入。")

    def train_classifier(self, batch_data, batch_labels):
        with tf.GradientTape() as tape:
            # Forward pass: 傳遞資料以計算預測
            predictions, _, _, _= self.classifier(batch_data, training=True)
            # 計算損失
            criterion = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

            loss = criterion(batch_labels, predictions)
            # 計算平均損失
            mean_loss = tf.reduce_mean(loss)
        # 計算梯度
        gradients = tape.gradient(mean_loss, self.classifier.trainable_variables)
        # 應用梯度
        self.optimizer.apply_gradients(zip(gradients, self.classifier.trainable_variables))

        return mean_loss

    def plot_tsne_3d(self, org_data, org_labels, generated_data, generated_labels, test_data, test_labels):
        # 合併所有數據
        combined_data = np.concatenate([org_data, generated_data, test_data], axis=0)
        combined_labels = np.concatenate([org_labels, generated_labels, test_labels], axis=0)

        # 標準化
        scaler = StandardScaler()
        normalized_data = scaler.fit_transform(combined_data.reshape(-1, combined_data.shape[-1]))

        # T-SNE 降維
        tsne = TSNE(n_components=3, random_state=42)
        tsne_results = tsne.fit_transform(normalized_data)

        # 分離降維結果
        org_tsne = tsne_results[:len(org_data)]
        gen_tsne = tsne_results[len(org_data): len(org_data) + len(generated_data)]
        test_tsne = tsne_results[len(org_data) + len(generated_data):]

        # 3D 繪圖
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        # 繪製原始資料（藍色）
        ax.scatter(org_tsne[org_labels == 0][:, 0], org_tsne[org_labels == 0][:, 1], org_tsne[org_labels == 0][:, 2],
                   color='blue', marker='o', label='Original Class 0')
        ax.scatter(org_tsne[org_labels == 1][:, 0], org_tsne[org_labels == 1][:, 1], org_tsne[org_labels == 1][:, 2],
                   color='blue', marker='^', label='Original Class 1')

        # 繪製生成資料（紅色）
        ax.scatter(gen_tsne[generated_labels == 0][:, 0], gen_tsne[generated_labels == 0][:, 1],
                   gen_tsne[generated_labels == 0][:, 2],
                   color='red', marker='o', label='Generated Class 0')
        ax.scatter(gen_tsne[generated_labels == 1][:, 0], gen_tsne[generated_labels == 1][:, 1],
                   gen_tsne[generated_labels == 1][:, 2],
                   color='red', marker='^', label='Generated Class 1')

        # 繪製測試資料（黃色）
        ax.scatter(test_tsne[test_labels == 0][:, 0], test_tsne[test_labels == 0][:, 1],
                   test_tsne[test_labels == 0][:, 2],
                   color='yellow', marker='o', label='Test Class 0')
        ax.scatter(test_tsne[test_labels == 1][:, 0], test_tsne[test_labels == 1][:, 1],
                   test_tsne[test_labels == 1][:, 2],
                   color='yellow', marker='^', label='Test Class 1')

        # 標題與標籤字體大小
        ax.set_title('3D T-SNE of Normalized Original, Generated, and Test Data', fontsize=16)
        ax.set_xlabel('TSNE Dimension 1', fontsize=14)
        ax.set_ylabel('TSNE Dimension 2', fontsize=14)
        ax.set_zlabel('TSNE Dimension 3', fontsize=14)

        # 圖例字體大小
        ax.legend(fontsize=14)

        # 儲存圖片
        plt.savefig(f'DTN_infoGAN_{Parameter.name}_TSNE.png', dpi=300, bbox_inches='tight')
        np.savez(
            f'DTN_infoGAN_{Parameter.name}_TSNE_data.npz',
            org_tsne=org_tsne,
            gen_tsne=gen_tsne,
            test_tsne=test_tsne,
            org_labels=org_labels,
            generated_labels=generated_labels,
            test_labels=test_labels
        )
        tsne_dict = {
            'org_tsne': org_tsne,
            'gen_tsne': gen_tsne,
            'test_tsne': test_tsne,
            'org_labels': org_labels,
            'generated_labels': generated_labels,
            'test_labels': test_labels
        }
        with open(f'DTN_infoGAN_{Parameter.name}_TSNE_data.pkl', 'wb') as f:
            pickle.dump(tsne_dict, f)
        # plt.show()


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
    # classes = ['notsmoking', 'smoking']
    classes = ["bleached_corals", "healthy_corals"]
    num_runs = 1  # 设置运行次数
    variance = []
    acc_list, auroc_list, f1_list = [], [], []
    initial_learning_rate = 0.0001
    all_runs_acc, all_runs_auroc = 0.0, 0.0
    acc_std, auroc_std = 0.0, 0.0
    gan_epochs = 50
    cls_epoch = 200
    batch_size = 64
    total_gan_epoch = 200


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


        gan_model = TotalModel(Parameter.hidden_dim, units, categorical_dim, continuous_dim)

        # # 加載function_f權重
        f_model = gan_model.function_f
        dummy_input = tf.random.normal(shape=(batch_size, Parameter.hidden_dim))  # 替換為實際的輸入形狀和數據
        _ = gan_model.function_f(dummy_input, training=False)
        # load weight
        f_model.load_weights('./DTN_weight/function_f/f_model_weights_{}.h5'.format(Parameter.name))

        # 印出新模型的摘要
        f_model.summary()
        #
        d_model_1 = gan_model.infogan1.discriminator
        d_model_2 = gan_model.infogan2.discriminator

        classifier = gan_model.classifier
        dummy_input2 = tf.random.normal(shape=(batch_size, units))
        _ = d_model_1(dummy_input2, training=False)
        _ = d_model_2(dummy_input2, training=False)

        _, _, _, _ = classifier(dummy_input2, training=False)
        # d_model_1.load_weights('./DTN_weight/classifier_g/g_model_weights_{}.h5'.format(Parameter.name))
        # d_model_2.load_weights('./DTN_weight/classifier_g/g_model_weights_{}.h5'.format(Parameter.name))
        classifier.load_weights('./DTN_weight/classifier_g/g_model_weights_{}.h5'.format(Parameter.name))
        # print("Trainable variables for classifier_g.Layers:")
        # for var in gan_model.classifier.trainable_variables:
        #     print(f"{var.name}: {var.numpy()}, dtype={var.dtype}")
        # classifier.summary()



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
        result = gan_model.fit(phi_data_tensor, test_phi_data_tensor, val_label_tensor,
                                batch_size=batch_size, num_epochs=gan_epochs,
                                cls_epochs=cls_epoch, training=True)
        acc, auroc, f1 = result["acc"], result["auroc"], result["f1"]
        print("Run", run, "acc :", acc)
        acc_list.append(acc)
        auroc_list.append(auroc)
        f1_list.append(f1)
    all_runs_acc = tf.reduce_mean(acc_list)
    all_runs_auroc = tf.reduce_mean(auroc_list)
    all_runs_f1 = tf.reduce_mean(f1_list)
    acc_std = np.std(acc_list)
    auroc_std = np.std(auroc_list)
    f1_std = np.std(f1_list)
    print("All Runs Acc = ", all_runs_acc)
    print("All Runs std = ", acc_std)
    print("All Runs AUROC = ", all_runs_auroc)
    print("All Runs AUROC std = ", auroc_std)
    print("All Runs F1 = ", all_runs_f1)
    print("All Runs F1 std = ", f1_std)
    # 將結果寫入文字檔
    with open(f"DTN_infoGAN_{Parameter.name}_results.txt", "w") as file:
        file.write(f"All Runs Acc = {all_runs_acc}\n")
        file.write(f"All Runs std = {acc_std}\n")
        file.write(f"All Runs AUROC = {all_runs_auroc}\n")
        file.write(f"All Runs AUROC std = {auroc_std}\n")
        file.write(f"All Runs F1 = {all_runs_f1}\n")
        file.write(f"All Runs F1 std = {f1_std}\n")
    print(f"結果已寫入 DTN_infoGAN_{Parameter.name}_results.txt")
