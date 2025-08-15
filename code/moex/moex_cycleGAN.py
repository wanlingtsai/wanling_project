import os
import math
import psutil
import pynvml
import pickle
import numpy as np
from glob import glob
import tensorflow as tf
from tensorflow import keras
from moex import moex_expansion
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, f1_score
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras import layers, Model, activations


h_data_path = "h_data/"
units = 200
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


def create_initializer(h_data_path, seed=None):
    save_directory = os.path.join(h_data_path, Parameter.name, "train")
    return tf.keras.initializers.GlorotNormal(seed=seed)


class Generator(tf.keras.Model):
    def __init__(self):
        super(Generator, self).__init__()

    def build(self, input_shape):
        self.LayerG = [
            layers.Dense(units=units,
                         activation='linear',
                         kernel_initializer=create_initializer(h_data_path),
                         bias_initializer=create_initializer(h_data_path),
                         name='g{}_l0'.format(Parameter.name)),
            BatchNormalization(),
            layers.LeakyReLU(alpha=0.2),

            layers.Dense(units=units,
                         activation='linear',
                         kernel_initializer=create_initializer(h_data_path),
                         bias_initializer=create_initializer(h_data_path),
                         name='g{}_l1'.format(Parameter.name)),
            BatchNormalization(),
            layers.LeakyReLU(alpha=0.2),

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


class Discriminator(tf.keras.Model):
    def __init__(self):
        super(Discriminator, self).__init__()

    def build(self, input_shape):
        self.LayerD = [
            layers.Dense(units=units,
                         activation='linear',
                         kernel_initializer=create_initializer(h_data_path),
                         bias_initializer=create_initializer(h_data_path),
                         name='d{}_l0'.format(Parameter.name)),
            BatchNormalization(),
            layers.LeakyReLU(alpha=0.2),

            layers.Dense(units=units,
                         activation='linear',
                         kernel_initializer=create_initializer(h_data_path),
                         bias_initializer=create_initializer(h_data_path),
                         name='d{}_l1'.format(Parameter.name)),
            BatchNormalization(),
            layers.LeakyReLU(alpha=0.2), ]

    def call(self, inputs, training=True):
        x = inputs
        for layer in self.LayerD:
            x = layer(x, training=training)
        predictions = x
        probabilities = tf.nn.softmax(predictions[:, 0])
        return probabilities

class CycleGAN(tf.keras.Model):
    def __init__(self, gen_G_optimizer, gen_F_optimizer, disc_X_optimizer, disc_Y_optimizer):
        super(CycleGAN, self).__init__()
        self.generator_G = Generator()
        self.generator_F = Generator()
        self.discriminator_X = Discriminator()
        self.discriminator_Y = Discriminator()
        self.gen_G_optimizer = gen_G_optimizer
        self.gen_F_optimizer = gen_F_optimizer
        self.disc_X_optimizer = disc_X_optimizer
        self.disc_Y_optimizer = disc_Y_optimizer

    def train_gen(self, batch_data):
        with tf.GradientTape(persistent=True) as tape:
            real_x, real_y = batch_data

            # Generator G translates X -> Y
            fake_y = self.generator_G(real_x, training=True)
            # Generator F translates Y -> X
            fake_x = self.generator_F(real_y, training=True)

            # Cycle consistency
            cycle_x = self.generator_F(fake_y, training=True)
            cycle_y = self.generator_G(fake_x, training=True)
            # Discriminator X evaluates X
            disc_real_x = self.discriminator_X(real_x, training=False)
            disc_fake_x = self.discriminator_X(fake_x, training=False)

            # Discriminator Y evaluates Y
            disc_real_y = self.discriminator_Y(real_y, training=False)
            disc_fake_y = self.discriminator_Y(fake_y, training=False)
            gen_G_loss = tf.reduce_mean((disc_fake_y - 1) ** 2)
            gen_F_loss = tf.reduce_mean((disc_fake_x - 1) ** 2)

            cycle_loss_x = tf.reduce_mean(tf.abs(real_x - cycle_x))
            cycle_loss_y = tf.reduce_mean(tf.abs(real_y - cycle_y))

            lambda_cycle = 0.1
            # Compute total losses
            total_gen_G_loss = gen_G_loss + lambda_cycle * cycle_loss_x
            total_gen_F_loss = gen_F_loss + lambda_cycle * cycle_loss_y

        # Calculate gradients for generators
        gen_G_gradients = tape.gradient(total_gen_G_loss, self.generator_G.trainable_variables)
        gen_F_gradients = tape.gradient(total_gen_F_loss, self.generator_F.trainable_variables)

        # Apply gradients for generators
        gen_G_optimizer.apply_gradients(zip(gen_G_gradients, self.generator_G.trainable_variables))
        gen_F_optimizer.apply_gradients(zip(gen_F_gradients, self.generator_F.trainable_variables))
        return total_gen_G_loss, total_gen_F_loss, fake_x

    def train_dic(self, batch_data):
        with tf.GradientTape(persistent=True) as tape:
            real_x, real_y = batch_data

            # Generator G translates X -> Y
            fake_y = self.generator_G(real_x, training=True)
            # Generator F translates Y -> X
            fake_x = self.generator_F(real_y, training=True)
            # Discriminator X evaluates X
            disc_real_x = self.discriminator_X(real_x, training=True)
            disc_fake_x = self.discriminator_X(fake_x, training=True)

            # Discriminator Y evaluates Y
            disc_real_y = self.discriminator_Y(real_y, training=True)
            disc_fake_y = self.discriminator_Y(fake_y, training=True)
            disc_X_loss = tf.reduce_mean((disc_real_x - 1) ** 2)
            disc_Y_loss = tf.reduce_mean((disc_real_y - 1) ** 2)
            total_disc_X_loss = disc_X_loss
            total_disc_Y_loss = disc_Y_loss
        # Calculate gradients for discriminators
        disc_X_gradients = tape.gradient(total_disc_X_loss, self.discriminator_X.trainable_variables)
        disc_Y_gradients = tape.gradient(total_disc_Y_loss, self.discriminator_Y.trainable_variables)

        # Apply gradients for discriminators
        disc_X_optimizer.apply_gradients(zip(disc_X_gradients, self.discriminator_X.trainable_variables))
        disc_Y_optimizer.apply_gradients(zip(disc_Y_gradients, self.discriminator_Y.trainable_variables))
        return total_disc_X_loss, total_disc_Y_loss

    def fit(self, X_data, batch_size, epoch_num):

        train_ds = tf.data.Dataset.from_tensor_slices(X_data)
        train_ds = train_ds.batch(batch_size)
        disc_X_losses_epoch = []
        gen_data = []
        for epoch in range(epoch_num):
            gen_G_losses = []
            gen_F_losses = []
            disc_X_losses = []
            disc_Y_losses = []

            for batch_data in train_ds:
                # Generate fake data for domain Y
                real_Y = np.random.normal(0, 1, size=batch_data.shape)

                # Train the CycleGAN model
                losses = self.train_step((batch_data, real_Y))

                # Collect losses
                gen_G_losses.append(losses["gen_G_loss"])
                gen_F_losses.append(losses["gen_F_loss"])
                disc_X_losses.append(losses["disc_X_loss"])
                disc_Y_losses.append(losses["disc_Y_loss"])
                if epoch == epoch_num - 1:
                    gen_data.append(losses["gen_data"])
            # Calculate average losses for the epoch
            avg_gen_G_loss = np.mean(gen_G_losses)
            avg_gen_F_loss = np.mean(gen_F_losses)
            avg_disc_X_loss = np.mean(disc_X_losses)
            avg_disc_Y_loss = np.mean(disc_Y_losses)

            # Collect discriminator X loss for plotting
            disc_X_losses_epoch.append(avg_disc_X_loss)

            # Print or log the average losses for the epoch
            print(f"Epoch {epoch + 1}:")
            print(f"  Generator G Loss: {avg_gen_G_loss}")
            print(f"  Generator F Loss: {avg_gen_F_loss}")
            print(f"  Discriminator X Loss: {avg_disc_X_loss}")
            print(f"  Discriminator Y Loss: {avg_disc_Y_loss}")

        # Plot the discriminator X loss over epochs
        plt.plot(range(epoch_num), disc_X_losses_epoch, label='Discriminator X Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Discriminator X Loss over Epochs')
        plt.legend()
        plt.savefig('disc_X_loss')
        plt.clf()
        # plt.show()
        merged_gen_data = tf.concat(gen_data, axis=0)
        return merged_gen_data

    def train_step(self, batch_data):
        total_gen_G_loss, total_gen_F_loss = 0.0, 0.0
        fake_x = None
        num_generator_updates = 10
        for _ in range(num_generator_updates):
            total_gen_G_loss, total_gen_F_loss, fake_x = self.train_gen(batch_data)

        disc_X_loss, disc_Y_loss = self.train_dic(batch_data)

        return {
            "gen_G_loss": total_gen_G_loss,
            "gen_F_loss": total_gen_F_loss,
            "disc_X_loss": disc_X_loss,
            "disc_Y_loss": disc_Y_loss,
            "gen_data": fake_x
        }


class TotalModel(tf.keras.Model):
    def __init__(self, gen_G_optimizer, gen_F_optimizer, disc_X_optimizer, disc_Y_optimizer):
        super(TotalModel, self).__init__()
        self.cyclegan1 = CycleGAN(gen_G_optimizer, gen_F_optimizer, disc_X_optimizer, disc_Y_optimizer)
        self.cyclegan2 = CycleGAN(gen_G_optimizer, gen_F_optimizer, disc_X_optimizer, disc_Y_optimizer)
        self.function_f = moex_expansion.function_f(input_dim=Parameter.hidden_dim, embedding_dim=units, name=Parameter.name)
        self.classifier = moex_expansion.classifier_g(input_dim=units, num_classes=Parameter.num_classes, name=Parameter.name, units=units)

    def call(self, inputs, training=False):
        predictions = []
        x = inputs
        x = self.function_f.call(x, training=training)
        x = self.classifier.call(x, training=training)
        predictions.append(x)
        return predictions

    def fit(self, train_data, test_data, test_label, batch_size, num_epochs, cls_epochs, training):
        train_data_c1 = train_data[0]
        train_data_c2 = train_data[1]

        test_data_c1 = test_data[0]
        test_data_c2 = test_data[1]

        test_label_c1 = test_label[0]
        test_label_c2 = test_label[1]

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
            gan_model1 = self.cyclegan1
            gan_model2 = self.cyclegan2

            total_best_acc = 0.0
            total_best_auroc = 0.0
            training_loss, test_epoch_accuracies = [], []
            best_acc, best_auroc, Epoch_loss, test_epoch_accuracy = 0.0, 0.0, 0.0, 0.0

            ratio = total_gan_epoch / gan_epochs
            # self.load_all_weights("./CycleGAN_weight")
            for i in range(int(ratio)):
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
                        # 使用模型进行预测
                        predictions = self.classifier(batch_data)

                        # 将模型的输出转换为类别
                        predicted_labels = tf.argmax(predictions, axis=1)

                        # 计算准确率
                        correct_predictions += np.sum(predicted_labels == batch_label.numpy())
                        total_predictions += len(batch_label)

                    training_accuracy = correct_predictions / total_predictions
                    Epoch_loss = tf.reduce_mean(epoch_loss)
                    training_loss.append(Epoch_loss)
                    total_training_acc.append(training_accuracy)
                    correct_predictions = 0
                    total_predictions = 0
                    epoch_auroc = []
                    all_true_labels = []
                    all_pred_labels = []
                    accuracy_metric = tf.keras.metrics.SparseCategoricalAccuracy()
                    for test_batch_data, test_batch_label in combined_test_dataset:
                        predictions = self.classifier(test_batch_data)
                        accuracy_metric.update_state(test_batch_label, predictions)
                        predicted_labels = np.argmax(predictions, axis=1)
                        correct_predictions += np.sum(np.argmax(predictions, axis=1) == test_batch_label)
                        total_predictions += len(test_batch_label)

                        # ← 累積真實標籤與預測標籤
                        all_true_labels.extend(test_batch_label.numpy())
                        all_pred_labels.extend(predicted_labels)

                        predictions_prob = tf.keras.activations.softmax(predictions, axis=-1)
                        # 二元 roc_auc_score 需要取正例標籤的機率
                        predictions_prob = predictions_prob[:, 1]
                        auroc = roc_auc_score(test_batch_label, predictions_prob)
                        epoch_auroc.append(auroc)
                    test_epoch_accuracy = accuracy_metric.result().numpy()
                    accuracy_metric.reset_states()  # 重置 metric 狀態
                    epoch_auroc = tf.reduce_mean(epoch_auroc)
                    # ← 計算 F1 分數（macro）
                    epoch_f1 = f1_score(all_true_labels, all_pred_labels, average='macro')

                    print(
                        f"Epoch {epoch + 1}/{cls_epochs}, Loss: {Epoch_loss:.4f}, Test Accuracy: {test_epoch_accuracy:.4f}, Test AUROC: {epoch_auroc:.4f}")
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
                            f'./CycleGAN_weight/classifier_g/moex_temp_best_weights_{Parameter.name}.h5')


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
                        f'./CycleGAN_weight/classifier_g/moex_temp_best_weights_{Parameter.name}.h5')
                    self.classifier.save_weights(
                        './CycleGAN_weight/classifier_g/moex_FinalBest_g_model_weights_{}.h5'.format(Parameter.name))
                print("best epoch = ", best_epoch)
                print("best accuracy = ", best_acc)
            print("Total best accuracy = ", total_best_acc)
            self.save_all_weights("./CycleGAN_weight")

            # 呼叫 t-SNE 畫圖
            self.plot_tsne_3d(
                org_data=final_org_data,
                org_labels=final_org_labels,
                generated_data=final_gen_data,
                generated_labels=final_gen_labels,
                test_data=tf.concat([test_data[0], test_data[1]], axis=0).numpy(),
                test_labels=tf.concat([test_label[0], test_label[1]], axis=0).numpy()
            )

            return {
                "acc": total_best_acc,
                "auroc": total_best_auroc,
                "f1": total_best_f1
            }
        else:
            self.classifier.load_weights(
                './CycleGAN_weight/classifier_g/moex_FinalBest_g_model_weights_{}.h5'.format(Parameter.name))

            correct_predictions = 0
            total_predictions = 0

            all_auroc = []
            # ← 初始化 F1-score 累積用列表
            all_true_labels = []
            all_pred_labels = []
            for test_batch_data, test_batch_label in combined_test_dataset:
                predictions = self.classifier(test_batch_data)
                pred_labels = np.argmax(predictions, axis=1)
                correct_predictions += np.sum(np.argmax(predictions, axis=1) == test_batch_label)
                total_predictions += len(test_batch_label)

                # ← 累積真實標籤與預測標籤
                all_true_labels.extend(test_batch_label.numpy())
                all_pred_labels.extend(pred_labels)

                predictions_prob = tf.keras.activations.softmax(predictions, axis=-1)
                # 二元 roc_auc_score 需要取正例標籤的機率
                predictions_prob = predictions_prob[:, 1]
                auroc = roc_auc_score(test_batch_label, predictions_prob)
                all_auroc.append(auroc)

            eval_acc = correct_predictions / total_predictions
            eval_auroc = tf.reduce_mean(all_auroc)
            # ← 計算 F1-score（macro）
            eval_f1 = f1_score(all_true_labels, all_pred_labels, average='macro')
            return {
                "acc": eval_acc,
                "auroc": eval_auroc,
                "f1": eval_f1
            }


    def train_classifier(self, batch_data, batch_labels):
        with tf.GradientTape() as tape:
            # Forward pass: 傳遞資料以計算預測
            predictions = self.classifier(batch_data, training=True)
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

    def save_all_weights(self, path):
        """儲存 TotalModel 下所有 CycleGAN 和其他子模型的權重"""
        # 保存 cyclegan1 的模型權重
        self.cyclegan1.generator_G.save_weights(f"{path}/moex_cyclegan1_generator_G_weights_{Parameter.name}.h5")
        self.cyclegan1.generator_F.save_weights(f"{path}/moex_cyclegan1_generator_F_weights_{Parameter.name}.h5")
        self.cyclegan1.discriminator_X.save_weights(f"{path}/moex_cyclegan1_discriminator_X_weights_{Parameter.name}.h5")
        self.cyclegan1.discriminator_Y.save_weights(f"{path}/moex_cyclegan1_discriminator_Y_weights_{Parameter.name}.h5")

        # 保存 cyclegan2 的模型權重
        self.cyclegan2.generator_G.save_weights(f"{path}/moex_cyclegan2_generator_G_weights_{Parameter.name}.h5")
        self.cyclegan2.generator_F.save_weights(f"{path}/moex_cyclegan2_generator_F_weights_{Parameter.name}.h5")
        self.cyclegan2.discriminator_X.save_weights(f"{path}/moex_cyclegan2_discriminator_X_weights_{Parameter.name}.h5")
        self.cyclegan2.discriminator_Y.save_weights(f"{path}/moex_cyclegan2_discriminator_Y_weights_{Parameter.name}.h5")

        print("所有模型的權重已儲存。")

    def load_all_weights(self, path):
        """載入 TotalModel 下所有 CycleGAN 和其他子模型的權重"""
        # 載入 cyclegan1 的模型權重
        dummy_input = tf.random.normal(shape=(batch_size, units))  # 替换为实际的输入形状
        _ = self.cyclegan1.generator_G(dummy_input, training=False)
        _ = self.cyclegan1.generator_F(dummy_input, training=False)
        _ = self.cyclegan1.discriminator_X(dummy_input, training=False)
        _ = self.cyclegan1.discriminator_Y(dummy_input, training=False)
        self.cyclegan1.generator_G.load_weights(f"{path}/moex_cyclegan1_generator_G_weights_{Parameter.name}.h5")
        self.cyclegan1.generator_F.load_weights(f"{path}/moex_cyclegan1_generator_F_weights_{Parameter.name}.h5")
        self.cyclegan1.discriminator_X.load_weights(f"{path}/moex_cyclegan1_discriminator_X_weights_{Parameter.name}.h5")
        self.cyclegan1.discriminator_Y.load_weights(f"{path}/moex_cyclegan1_discriminator_Y_weights_{Parameter.name}.h5")

        # 載入 cyclegan2 的模型權重
        _ = self.cyclegan2.generator_G(dummy_input, training=False)
        _ = self.cyclegan2.generator_F(dummy_input, training=False)
        _ = self.cyclegan2.discriminator_X(dummy_input, training=False)
        _ = self.cyclegan2.discriminator_Y(dummy_input, training=False)
        self.cyclegan2.generator_G.load_weights(f"{path}/moex_cyclegan2_generator_G_weights_{Parameter.name}.h5")
        self.cyclegan2.generator_F.load_weights(f"{path}/moex_cyclegan2_generator_F_weights_{Parameter.name}.h5")
        self.cyclegan2.discriminator_X.load_weights(f"{path}/moex_cyclegan2_discriminator_X_weights_{Parameter.name}.h5")
        self.cyclegan2.discriminator_Y.load_weights(f"{path}/moex_cyclegan2_discriminator_Y_weights_{Parameter.name}.h5")

        print("所有模型的權重已載入。")

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

        # 圖片細節
        ax.set_title('3D T-SNE of Normalized Original, Generated, and Test Data', fontsize=14)
        ax.set_xlabel('TSNE Dimension 1', fontsize=12)
        ax.set_ylabel('TSNE Dimension 2', fontsize=12)
        ax.set_zlabel('TSNE Dimension 3', fontsize=12)
        ax.legend(fontsize=14)
        # 儲存圖像
        save_path = f"moex_cycleGAN_{Parameter.name}_TSNE.png"  # 你可以改成其他格式，例如 "ours_moex_fig.pdf"
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"圖像已儲存至 {save_path}")
        np.savez(
            f'moex_cycleGAN_{Parameter.name}_TSNE_data.npz',
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
        with open(f'moex_cycleGAN_{Parameter.name}_TSNE_data.pkl', 'wb') as f:
            pickle.dump(tsne_dict, f)
        plt.show()

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
    acc_list, auroc_list, f1_list = [], [], []
    initial_learning_rate = 0.0002
    all_runs_acc, all_runs_auroc = 0.0, 0.0
    acc_std, auroc_std = 0.0, 0.0
    gan_epochs = 50
    cls_epoch = 100
    batch_size = 128
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
        gen_G_optimizer = tf.keras.optimizers.Adam(beta_1=0.9, beta_2=0.999)
        gen_F_optimizer = tf.keras.optimizers.Adam(beta_1=0.9, beta_2=0.999)
        disc_X_optimizer = tf.keras.optimizers.Adam(beta_1=0.9, beta_2=0.999)
        disc_Y_optimizer = tf.keras.optimizers.Adam(beta_1=0.9, beta_2=0.999)

        gan_model = TotalModel(gen_G_optimizer, gen_F_optimizer, disc_X_optimizer, disc_Y_optimizer)

        # 加載function_f權重
        f_model = gan_model.function_f
        dummy_input = tf.random.normal(shape=(batch_size, 1920))  # 替換為實際的輸入形狀和數據
        _ = gan_model.function_f(dummy_input, training=False)
        # load weight
        f_model.load_weights('./moex_weight/function_f/f_model_weights_{}.h5'.format(Parameter.name))

        # 印出新模型的摘要
        f_model.summary()


        classifier = gan_model.classifier
        dummy_input2 = tf.random.normal(shape=(batch_size, units))
        _ = classifier(dummy_input2, training=False)

        classifier.load_weights('./moex_weight/classifier_g/g_model_weights_{}.h5'.format(Parameter.name))
        print("Trainable variables for classifier_g.Layers:")
        for var in gan_model.classifier.trainable_variables:
            print(f"{var.name}: {var.numpy()}, dtype={var.dtype}")
        classifier.summary()
        # 載入保存的數據
        file_path = './Aug_data/moex_aug_data_{}.npz'.format(Parameter.name)  # 記得使用正確的文件名
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
        optimizer = tf.keras.optimizers.Adam(learning_rate=initial_learning_rate)

        # cannot delete
        gan_model.compile(optimizer=optimizer)
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

        result = gan_model.fit(train_data, test_phi_data_tensor, val_label_tensor,
                                   batch_size=batch_size, num_epochs=gan_epochs, cls_epochs=cls_epoch,
                                   training=True)
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
    with open(f"moex_cycleGAN_{Parameter.name}_results.txt", "w") as file:
        file.write(f"All Runs Acc = {all_runs_acc}\n")
        file.write(f"All Runs std = {acc_std}\n")
        file.write(f"All Runs AUROC = {all_runs_auroc}\n")
        file.write(f"All Runs AUROC std = {auroc_std}\n")
        file.write(f"All Runs F1 = {all_runs_f1}\n")
        file.write(f"All Runs F1 std = {f1_std}\n")
    print(f"結果已寫入 moex_cycleGAN_{Parameter.name}_results.txt")
