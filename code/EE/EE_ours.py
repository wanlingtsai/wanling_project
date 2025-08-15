import os
import time
import pickle
import numpy as np
from glob import glob
import tensorflow as tf
from sklearn.manifold import TSNE
from EE import embedding_expansion
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, f1_score
from tensorflow.keras import layers, Model, activations, optimizers

units = 200


class Parameter(object):
    name = "alzheimer"
    num_classes = 2
    hidden_dim = 1920


def create_initializer(seed=None):
    return tf.keras.initializers.GlorotNormal(seed=seed)


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


def generate_noise(batch_size, dim):
    # 生成形狀為 (batch_size, dim) 的兩組正態分佈噪聲
    noise1 = tf.random.normal(shape=(batch_size, dim), mean=0.0, stddev=1.0)
    noise2 = tf.random.normal(shape=(batch_size, dim), mean=0.0, stddev=1.0)
    return [noise1, noise2]


class Generator(tf.keras.Model):
    def __init__(self):
        super(Generator, self).__init__()

    def build(self, input_shape):
        self.LayerG = [
            layers.Dense(units=units,
                         activation='linear',
                         kernel_initializer=create_initializer(),
                         bias_initializer=create_initializer(),
                         name='g{}_l0'.format(Parameter.name)),
            layers.BatchNormalization(),
            layers.LeakyReLU(alpha=0.2),

            layers.Dense(units=units,
                         activation='linear',
                         kernel_initializer=create_initializer(),
                         bias_initializer=create_initializer(),
                         name='g{}_l1'.format(Parameter.name)),
            layers.BatchNormalization(),
            layers.LeakyReLU(alpha=0.2),

            layers.Dense(units=units,
                         activation='linear',
                         kernel_initializer=create_initializer(),
                         bias_initializer=create_initializer(),
                         name='g{}_l2'.format(Parameter.name)),
        ]

    def call(self, inputs, training=True):

        x = inputs
        for layer in self.LayerG:
            x = layer(x, training=training)

        return x


class Discriminator(tf.keras.Model):
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
        self.class_Layers = [
            layers.Dense(units=units,
                         activation='linear',
                         kernel_initializer=create_initializer(),
                         bias_initializer=create_initializer(),
                         name='d{}_l0'.format(self.d_name)),
            # layers.BatchNormalization(),
            layers.LeakyReLU(alpha=0.3),

            layers.Dense(units=units,
                         activation='linear',
                         kernel_initializer=create_initializer(),
                         bias_initializer=create_initializer(),
                         name='d{}_l1'.format(self.d_name)),
            # layers.BatchNormalization(),
            layers.LeakyReLU(alpha=0.3),

            layers.Dense(units=2,
                         activation='linear',
                         kernel_initializer=create_initializer(),
                         bias_initializer=create_initializer(),
                         name='d{}_l2'.format(self.d_name)), ]


    @tf.function
    def call(self, inputs, training=False):
        # print("call input:", type(inputs))
        x = inputs
        for layer in self.class_Layers:
            x = layer(x, training=training)
        return x


class Discriminator_rf(tf.keras.Model):
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
        # 用於真假判別的網路
        self.real_fake_layers = [
            layers.Dense(units=self.units,
                         activation='linear',
                         kernel_initializer=create_initializer(),
                         bias_initializer=create_initializer(),
                         name='{}_rf_l0'.format(self.d_name)),
            layers.BatchNormalization(),
            layers.LeakyReLU(alpha=0.2),
            layers.Dense(units=self.units,
                         activation='linear',
                         kernel_initializer=create_initializer(),
                         bias_initializer=create_initializer(),
                         name='{}_rf_l1'.format(self.d_name)),
            layers.BatchNormalization(),
            layers.LeakyReLU(alpha=0.2),
            layers.Dense(units=1,  # 判別真假，輸出為1維（真或假）
                         activation='sigmoid',  # 使用 sigmoid 作為激活函數
                         kernel_initializer=create_initializer(),
                         bias_initializer=create_initializer(),
                         name='{}_rf_output'.format(self.d_name)),
        ]

    @tf.function
    def call(self, inputs, training=False):
        # 真假判別網路
        x_rf = inputs
        for layer in self.real_fake_layers:
            x_rf = layer(x_rf, training=training)

        return x_rf


class Jammer(tf.keras.Model):
    def __init__(self):
        super(Jammer, self).__init__()

    def build(self, input_shape):
        self.query_layer = layers.Dense(units=units,
                                        activation='linear',
                                        kernel_initializer=create_initializer(),
                                        bias_initializer=create_initializer(),
                                        name='query'
                                        )

        self.key_layer = layers.Dense(units=units,
                                      activation='linear',
                                      kernel_initializer=create_initializer(),
                                      bias_initializer=create_initializer(),
                                      name='key'
                                      )

        self.value_layer = layers.Dense(units=units,
                                        activation='linear',
                                        kernel_initializer=create_initializer(),
                                        bias_initializer=create_initializer(),
                                        name='value'
                                        )

    @tf.function
    def call(self, generated, real):
        query = self.query_layer(generated)
        key = self.key_layer(real)
        value = self.value_layer(real)

        attention = tf.matmul(query, key, transpose_b=True)
        # 將內積值轉為機率分布，每個值代表生成數據中不同部分的重要性或權重，簡而言之，att_prob 決定了在生成數據中每個部分應該受到多大程度的關注或影響。
        att_prob = tf.nn.softmax(attention / tf.math.sqrt(tf.cast(units, tf.float32)), axis=-1)
        # 根據 att_prob 的權重分佈，對生成數據 generated 進行了一種調整或干擾。這個操作使得 j_output 更接近於真實數據 real 的特徵
        j_output = tf.matmul(att_prob, value)

        return j_output  # A fake instance but is like real one

class Jam_GAN(tf.keras.Model):
    def __init__(self, input_dim):
        super(Jam_GAN, self).__init__()
        self.input_dim = input_dim
        self.function_f = embedding_expansion.function_f(input_dim=Parameter.hidden_dim, embedding_dim=units,
                        name=Parameter.name)
        self.classifier = embedding_expansion.classifier_g(input_dim=units, num_classes=Parameter.num_classes,
                        name=Parameter.name)
        self.generator1 = Generator()
        self.generator2 = Generator()
        self.Generator_list = [self.generator1, self.generator2]
        self.discriminator1 = Discriminator(input_dim, Parameter.num_classes, Parameter.name)
        self.discriminator2 = Discriminator(input_dim, Parameter.num_classes, Parameter.name)
        self.Discriminator_list = [self.discriminator1, self.discriminator2]
        self.jammer = Jammer()
        self.disc_rf = Discriminator_rf(input_dim, Parameter.num_classes, Parameter.name)
        self.generator_optimizer = optimizers.Adam(learning_rate=0.0005, beta_1=0.5)
        self.discriminator_optimizer = optimizers.Adam(learning_rate=0.0001, beta_1=0.5)
        self.discriminator_rf_optimizer = optimizers.Adam(learning_rate=0.0001, beta_1=0.5)
        self.jammer_optimizer = optimizers.Adam(learning_rate=0.0001, beta_1=0.5)
        self.acc = tf.keras.metrics.SparseCategoricalAccuracy()
        self.optimizer = optimizer

    def call(self, inputs, training=False):
        class1_data = inputs[0]
        class2_data = inputs[1]
        batch_size = tf.shape(class1_data)[0]
        noise1 = tf.random.normal([batch_size, self.input_dim])
        noise2 = tf.random.normal([batch_size, self.input_dim])
        generated_1 = self.generator1(noise1, training=training)
        generated_2 = self.generator2(noise2, training=training)
        jammed_1 = self.jammer(generated_1, class2_data, training=training)
        jammed_2 = self.jammer(generated_2, class1_data, training=training)
        CE_loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        BCE_loss = tf.keras.losses.BinaryCrossentropy(from_logits=False)
        output_class_logits, output_real_fake_logits = [], []

        D_real_loss_class, D_real_loss_rf = [], []
        for i in range(Parameter.num_classes):
            # inputs[i]: 對應每個類別的真實數據
            # 判別器輸出分類和真假判別的 logits
            class_logits = self.Discriminator_list[i](inputs[i], training=training)
            real_fake_logits = self.disc_rf(inputs[i], training=training)
            batch_size = tf.shape(class_logits)[0]
            # 保存 logits
            output_class_logits.append(class_logits)
            output_real_fake_logits.append(real_fake_logits)

            # 計算分類損失
            # 是這類
            class_labels = tf.fill([tf.shape(class_logits)[0]], 1)  # 創建分類標籤
            pos_class_loss = CE_loss(class_labels, class_logits)

            # 不是這類
            neg_labels = tf.fill([tf.shape(class_logits)[0]], 0)
            neg_class_loss = CE_loss(neg_labels, class_logits)
            real_loss = pos_class_loss + neg_class_loss
            D_real_loss_class.append(real_loss)
            # 計算真假判別損失
            real_labels = tf.ones((batch_size, 1))  # 真實數據標籤為 1
            rf_loss = BCE_loss(real_labels, real_fake_logits)
            D_real_loss_rf.append(rf_loss)
        # 到時候 D_real_loss_class 的[0]和[1]要分別更新D1 D2
        D_real_loss_rf = tf.reduce_sum(D_real_loss_rf)
        # 合併生成資料
        generated = [generated_1, generated_2]

        output_class_logits, output_real_fake_logits, \
            output_label_class_D, output_label_class_G, \
            output_rf_label_D, output_rf_label_G = self.take_generated(generated, training=training)

        # print("again:", output_real_fake_logits)# 計算損失 D+_cls 3/ G1/ D_rf 3/ G4
        D_gen_loss_class, G_gen_loss_class,\
            D_gen_loss_real_fake,\
            G_gen_loss_real_fake = self.calculate_gen_losses(output_class_logits,
                                                        output_real_fake_logits,
                                                        output_label_class_D,output_label_class_G,
                                                        output_rf_label_D,output_rf_label_G)
        jammer = [jammed_1, jammed_2]

        output_jammer = self.take_jammer(jammer, training=training)

        JD_loss_class_list, JG_loss_class_list, JJ_loss_class_list,\
            JD_loss_real_fake_list, JG_loss_real_fake_list,\
            JJ_loss_real_fake_list = self.calculate_jam_losses(output_jammer)

        return D_real_loss_class, D_real_loss_rf, D_gen_loss_class, G_gen_loss_class,\
            D_gen_loss_real_fake,G_gen_loss_real_fake, JD_loss_class_list, JG_loss_class_list, JJ_loss_class_list,\
            JD_loss_real_fake_list, JG_loss_real_fake_list,\
            JJ_loss_real_fake_list

    def fit(self, train_data, train_label, test_data, test_label, batch_size, num_epochs, cls_epochs, mode, training=True):
        self.load_all_weights('./ours_weight')
        ratio = total_training_epoch // num_epochs

        total_best_acc = 0.0
        total_best_auroc = 0.0
        total_best_f1 = 0.0

        train_data_c1 = train_data[0]
        train_data_c2 = train_data[1]
        train_label_c1 = train_label[0]
        train_label_c2 = train_label[1]

        train_dataset_c1 = tf.data.Dataset.from_tensor_slices((train_data_c1, train_label_c1)).batch(batch_size)
        train_dataset_c2 = tf.data.Dataset.from_tensor_slices((train_data_c2, train_label_c2)).batch(batch_size)
        train_dataset = tf.data.Dataset.zip((train_dataset_c1, train_dataset_c2))

        if training:
            final_org_data, final_org_labels = [], []
            final_gen_data, final_gen_labels = [], []

            for step in range(ratio):
                start_time = time.time()
                best_acc = 0.0
                best_auroc = 0.0
                best_f1 = 0.0

                # --- 1. Train GAN ---
                for epoch in range(gan_epochs):
                    print("Epoch", (gan_epochs * step + epoch + 1), ":")
                    self.acc.reset_states()
                    if mode == 'gd':
                        self.d_ratio, self.g_ratio, self.j_ratio = 15, 20, 0
                    elif mode == 'gdj':
                        self.d_ratio, self.g_ratio, self.j_ratio = 3, 5, 3

                    all_class1_data, all_class1_label = [], []
                    all_class2_data, all_class2_label = [], []

                    for batch in train_dataset:
                        (data_c1, label_c1), (data_c2, label_c2) = batch
                        batch_data = tf.stack([data_c1, data_c2], axis=0)

                        all_class1_data.append(data_c1)
                        all_class1_label.append(label_c1)
                        all_class2_data.append(data_c2)
                        all_class2_label.append(label_c2)

                        for num in range(self.d_ratio):
                            D_loss, D_rf_loss = self.train_step(batch_data, mode="train_D", num=num)
                            print(
                                f"D1 loss : {D_loss[0]:>8.6f}   D2 loss : {D_loss[1]:>10.6f}   Drf loss : {D_rf_loss:>8.6f}")

                        for _ in range(self.g_ratio):
                            G_loss = self.train_step(batch_data, mode="train_G")
                            print(f"G1 loss : {G_loss[0]:>8.6f}   G2 loss : {G_loss[1]:>10.6f}")

                        if mode == 'gdj':
                            for _ in range(self.j_ratio):
                                J_loss = self.train_step(batch_data, mode="train_J")
                                print(f"J loss : {J_loss:>8.6f}")

                print(f"GAN Training Time: {(time.time() - start_time) / 60.0:.3f} minutes")

                # --- 2. Collect Sr and generate Sg ---
                org_data_class1 = tf.concat(all_class1_data, axis=0)
                org_label_class1 = tf.concat(all_class1_label, axis=0)
                org_data_class2 = tf.concat(all_class2_data, axis=0)
                org_label_class2 = tf.concat(all_class2_label, axis=0)

                org_data = tf.concat([org_data_class1, org_data_class2], axis=0)
                org_labels = tf.concat([org_label_class1, org_label_class2], axis=0)

                gen_c0, gen_c1 = [], []
                for _ in range(5):  # 產生固定 M 批
                    noise = generate_noise(batch_size, units)
                    fake = [G(noise[i], training=False) for i, G in enumerate(self.Generator_list)]
                    gen_c0.append(fake[0])
                    gen_c1.append(fake[1])

                generated_data_0 = tf.concat(gen_c0, axis=0)
                generated_data_1 = tf.concat(gen_c1, axis=0)
                generated_data = tf.concat([generated_data_0, generated_data_1], axis=0)
                generated_labels = tf.concat(
                    [tf.zeros_like(generated_data_0[:, 0], dtype=tf.int32),
                     tf.ones_like(generated_data_1[:, 0], dtype=tf.int32)], axis=0)

                # --- 3. Train Classifier on Sw = Sr ∪ Sg ---
                Sw_data = tf.concat([org_data, generated_data], axis=0)
                Sw_labels = tf.concat([org_labels, generated_labels], axis=0)
                train_ds = tf.data.Dataset.from_tensor_slices((Sw_data, Sw_labels)).shuffle(1000).batch(batch_size)

                test_data_flat = tf.concat([test_data[0], test_data[1]], axis=0)
                test_label_flat = tf.concat([test_label[0], test_label[1]], axis=0)
                test_ds = tf.data.Dataset.from_tensor_slices((test_data_flat, test_label_flat)).batch(batch_size)

                # self.classifier.load_weights(
                #     './ours_weight/classifier_g/EE_FinalBest_g_model_weights_{}.h5'.format(Parameter.name))

                for epoch in range(cls_epoch):
                    epoch_loss, num_batches = 0.0, 0
                    for batch_x, batch_y in train_ds:
                        loss = self.train_classifier(batch_x, batch_y)
                        epoch_loss += loss.numpy()
                        num_batches += 1
                    epoch_loss /= num_batches

                    if (epoch + 1) % 1 == 0:
                        all_labels, all_preds = [], []
                        acc_metric = tf.keras.metrics.SparseCategoricalAccuracy()
                        for batch_x, batch_y in test_ds:
                            preds = self.classifier(batch_x, training=False)
                            acc_metric.update_state(batch_y, preds)
                            all_labels.extend(batch_y.numpy())
                            all_preds.extend(tf.nn.softmax(preds, axis=-1).numpy()[:, 1])
                        acc = acc_metric.result().numpy()
                        auroc = roc_auc_score(all_labels, all_preds)
                        pred_labels = [1 if p >= 0.5 else 0 for p in all_preds]
                        f1 = f1_score(all_labels, pred_labels)

                        print(
                            f"[Classifier Epoch {epoch + 1}] Loss: {epoch_loss:.4f}, Acc: {acc:.4f}, AUROC: {auroc:.4f}, F1: {f1:.4f}")

                        if acc > best_acc:
                            best_acc, best_auroc, best_f1 = acc, auroc, f1
                            temp_org_data = tf.concat([train_data_c1, train_data_c2], axis=0)
                            temp_org_labels = tf.constant([0] * len(train_data_c1) + [1] * len(train_data_c2))

                            temp_gen_data = tf.concat([generated_data_0, generated_data_1], axis=0)
                            temp_gen_labels = tf.constant([0] * len(generated_data_0) + [1] * len(generated_data_1))

                            # 儲存暫時最佳分類器權重
                            self.classifier.save_weights(f'./ours_weight/classifier_g/EE_temp_best_weights_{Parameter.name}.h5')

                if best_acc > total_best_acc:
                    total_best_acc, total_best_auroc, total_best_f1 = best_acc, best_auroc, best_f1
                    final_org_data = org_data.numpy()
                    final_org_labels = org_labels.numpy()
                    final_gen_data = generated_data.numpy()
                    final_gen_labels = generated_labels.numpy()
                    self.classifier.load_weights(f'./ours_weight/classifier_g/EE_temp_best_weights_{Parameter.name}.h5')
                    self.classifier.save_weights(
                        './ours_weight/classifier_g/EE_FinalBest_g_model_weights_{}.h5'.format(Parameter.name))

            self.save_all_weights('./ours_weight')
            self.plot_tsne_3d(final_org_data, final_org_labels, final_gen_data, final_gen_labels,
                              tf.concat([test_data[0], test_data[1]], axis=0).numpy(),
                              tf.concat([test_label[0], test_label[1]], axis=0).numpy())

            print("total best acc = ", total_best_acc)
            return {
                'acc': total_best_acc,
                'auroc': total_best_auroc,
                'f1': total_best_f1
            }
        else:
            self.classifier.load_weights(
                './ours_weight/classifier_g/EE_FinalBest_g_model_weights_{}.h5'.format(Parameter.name))
            # 假如 test_data 本身就是 Tensor，你可以轉換為 ndarray
            val_data = test_data.numpy().reshape(-1, 200)
            val_labels = test_label.numpy().reshape(-1)

            # 產生隨機索引用以shuffle
            indices = np.arange(len(val_data))
            np.random.shuffle(indices)

            val_data_shuffled = val_data[indices]
            val_labels_shuffled = val_labels[indices]

            test_acc, test_auroc, test_f1 = self.evaluate(val_data_shuffled, val_labels_shuffled)
            return {
                "acc": test_acc,
                "auroc": test_auroc,
                "f1": test_f1
            }

    def train_step(self, inputs, mode=None, num=None):
        with tf.GradientTape(persistent=True) as gen_tape, tf.GradientTape(persistent=True) as disc_tape,tf.GradientTape(persistent=True) as disc_rf_tape, tf.GradientTape(persistent=True) as jam_tape:

            D_real_loss_class, D_real_loss_rf, D_gen_loss_class, G_gen_loss_class, \
                D_gen_loss_real_fake, G_gen_loss_real_fake, JD_loss_class_list, JG_loss_class_list, JJ_loss_class_list, \
                JD_loss_real_fake_list, JG_loss_real_fake_list, \
                JJ_loss_real_fake_list = self(inputs, training=True)
            D_gen_loss_real_fake = tf.reduce_sum(D_gen_loss_real_fake)
            JD_loss_real_fake = tf.reduce_sum(JD_loss_real_fake_list)
            JG_loss_real_fake = tf.reduce_sum(JG_loss_real_fake_list)
            JJ_loss_class = tf.reduce_sum(JJ_loss_class_list)
            JJ_loss_real_fake = tf.reduce_sum(JJ_loss_real_fake_list)
            D_loss = [(i + j + k) for i, j, k in
                      zip(D_real_loss_class, D_gen_loss_class, JD_loss_class_list)]

            D_rf_loss = D_real_loss_rf + D_gen_loss_real_fake + JD_loss_real_fake

            G_loss = []
            for cls in range(Parameter.num_classes):
                start = cls * 2
                end = start + 2

                G_class_loss = G_gen_loss_class[cls] if cls < len(G_gen_loss_class) else 0.0

                G_rf_loss = G_gen_loss_real_fake[cls] if cls < len(G_gen_loss_real_fake) else 0.0
                JG_class_loss = tf.constant(0.0, dtype=tf.float32)
                if JG_loss_class_list[start:end]:
                    JG_class_loss += tf.add_n(JG_loss_class_list[start:end])

                JG_rf_loss = JG_loss_real_fake_list[cls] if cls < len(JG_loss_real_fake_list) else 0.0
                # 最終 G_loss（單一數值）
                G_total_loss = G_class_loss + G_rf_loss + JG_class_loss + JG_rf_loss
                G_loss.append(G_total_loss)

            if mode == "train_J":
                J_loss = (
                        tf.add_n(JD_loss_real_fake_list) +
                        tf.add_n(JG_loss_real_fake_list) +
                        tf.add_n(JJ_loss_class_list) +
                        tf.add_n(JJ_loss_real_fake_list)
                )

        if mode == "train_D":
            for i, discriminator in enumerate(self.Discriminator_list):
                # 獲取對應的 D 的損失
                disc_loss = D_loss[i]

                # 計算梯度
                grad_D = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
                # 更新對應的 D
                self.discriminator_optimizer.apply_gradients(zip(grad_D, discriminator.trainable_variables))

            disc_rf_loss = D_rf_loss
            grad_D_rf = disc_rf_tape.gradient(disc_rf_loss, self.disc_rf.trainable_variables)
            self.discriminator_rf_optimizer.apply_gradients(zip(grad_D_rf, self.disc_rf.trainable_variables))

            return D_loss, D_rf_loss

        elif mode == "train_G":
            for i, generator in enumerate(self.Generator_list):
                # 獲取對應的 D 的損失
                gen_loss = G_loss[i]
                # 計算梯度
                grad_G = gen_tape.gradient(gen_loss, generator.trainable_variables)
                # 更新對應的 G
                self.generator_optimizer.apply_gradients(zip(grad_G, generator.trainable_variables))
            return  G_loss

        elif mode == "train_J":
            # 獲取對應的 D 的損失
            jam_loss = J_loss
            # 計算梯度
            grad_J = jam_tape.gradient(jam_loss, self.jammer.trainable_variables)
            # 更新對應的 G
            self.jammer_optimizer.apply_gradients(zip(grad_J, self.jammer.trainable_variables))
            return  J_loss

        else:
            raise ValueError(f"Unknown training mode: {mode}")

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

    def take_generated(self, generated, training=False):
        """
        Args:
            generated: List of generated data for each class, [generated_1, generated_2].
            training: Boolean, indicating if the discriminator is in training mode.

        Returns:
            output_class_logits: List of logits from each Discriminator for the generated data (分類 logits)。
            output_real_fake_logits: List of real/fake logits from each Discriminator for the generated data。
            output_label_class_D: Labels for updating the Discriminator's classification task (屬於該類的標籤，用於更新D)。
            output_label_class_G: Labels for updating the Generator's classification task (屬於該類的標籤，用於更新G)。
            output_label_D: Labels for updating the Discriminator's real/fake task (真/假標籤，用於更新D)。
            output_label_G: Labels for updating the Generator's real/fake task (真/假標籤，用於更新G)。
        """
        output_class_logits = []
        output_real_fake_logits = []
        output_label_class_D = []  # 用於更新 Discriminator (分類標籤)
        output_label_class_G = []  # 用於更新 Generator (分類標籤)
        output_rf_label_D = []  # 用於更新 Discriminator (真假標籤)
        output_rf_label_G = []  # 用於更新 Generator (真假標籤)

        for i in range(Parameter.num_classes):
            # 取得對應類別的生成數據
            generated_data = generated[i]

            # 判別器對生成數據的輸出 (分類 logits 和真假 logits)
            class_logits = self.Discriminator_list[i](generated_data, training=training)
            real_fake_logits = self.disc_rf(generated_data, training=training)

            # 保存 logits
            output_class_logits.append(class_logits)
            output_real_fake_logits.append(real_fake_logits)

            # === 建立分類標籤 ===
            # 用於更新 D: 判別器希望將生成數據判為「非該類」
            batch_size = tf.shape(class_logits)[0]  # 獲取 batch_size
            class_labels_D = tf.zeros((batch_size,), dtype=tf.int32)  # 0 表示生成數據不屬於該類別
            output_label_class_D.append(class_labels_D)

            # 用於更新 G: 生成器希望生成數據被判為「屬於該類」
            class_labels_G = tf.ones((batch_size,), dtype=tf.int32)  # 1 表示生成數據希望被認為屬於該類別
            output_label_class_G.append(class_labels_G)

            # === 建立真假標籤 ===
            # 用於判別器 (D_rf): 0 表示生成數據 (假數據)
            d_labels = tf.zeros((batch_size, 1), dtype=tf.float32)
            # d_labels = tf.zeros_like(real_fake_logits)  # 假數據標籤為 0
            output_rf_label_D.append(d_labels)

            # 用於生成器 (G): 1 表示希望判別器認為生成數據是 "真實" 的
            g_labels = tf.ones((batch_size, 1), dtype=tf.float32)
            # g_labels = tf.ones_like(real_fake_logits)  # 希望被判為真實
            output_rf_label_G.append(g_labels)

        # 最後將各 list 疊起來
        output_class_logits = tf.stack(output_class_logits, axis=0)
        output_real_fake_logits = tf.stack(output_real_fake_logits, axis=0)
        output_label_class_D = tf.stack(output_label_class_D, axis=0)
        output_label_class_G = tf.stack(output_label_class_G, axis=0)
        output_rf_label_D = tf.stack(output_rf_label_D, axis=0)
        output_rf_label_G = tf.stack(output_rf_label_G, axis=0)

        return output_class_logits, output_real_fake_logits, output_label_class_D, output_label_class_G, output_rf_label_D, output_rf_label_G

    def calculate_jam_losses(self, output_jammer):
        """
        Args:
            output_class_logits: List of classification logits for generated data (G(z)).
            output_real_fake_logits: List of real/fake logits for generated data (G(z)).
            output_label_class_D: Labels for Discriminator's classification task.
            output_label_class_G: Labels for Generator's classification task.
            output_rf_label_D: Labels for Discriminator's real/fake task.
            output_rf_label_G: Labels for Generator's real/fake task.

        Returns:
            D_loss_class_list: List of classification losses for updating D_class.
            G_loss_class_list: List of classification losses for updating G.
            D_loss_real_fake_list: List of real/fake losses for updating D_rf.
            G_loss_real_fake_list: List of real/fake losses for updating G.
        """
        # 分類損失 (SparseCategoricalCrossentropy)
        CE_loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        # 真/假損失 (BinaryCrossentropy)
        BCE_loss = tf.keras.losses.BinaryCrossentropy(from_logits=False)

        # 初始化損失值
        D_loss_class_list, G_loss_class_list, J_loss_class_list = [], [], []
        D_loss_real_fake_list, G_loss_real_fake_list, J_loss_real_fake_list = [], [], []
        # class
        D_pos_cls_logit = [output_jammer['D1_G1_cls_logits'], output_jammer['D2_G2_cls_logits']]
        D_pos_cls_label = [output_jammer['D1_G1_cls_label'], output_jammer['D2_G2_cls_label']]
        D_neg_cls_logit = [output_jammer['D1_G2_cls_logits'], output_jammer['D2_G1_cls_logits']]
        D_neg_cls_label = [output_jammer['D1_G2_cls_label'], output_jammer['D2_G1_cls_label']]
        D_neg_cls_logit_swapped = [output_jammer['D2_G1_cls_logits'], output_jammer['D1_G2_cls_logits']]
        G_pos_cls_label = [output_jammer['G1_G1_cls_label'], output_jammer['G2_G2_cls_label']]
        G_neg_cls_label = [output_jammer['G1_D2_G1_cls_label'], output_jammer['G2_D1_G2_cls_label']]
        J_pos_cls_label = [output_jammer['J_D1_G1_cls_label'], output_jammer['J_D2_G2_cls_label']]
        J_neg_cls_label = [output_jammer['J_D1_G2_cls_label'], output_jammer['J_D2_G1_cls_label']]
        # real_fake
        D_rf_output = [output_jammer['D_G1_rf_output'], output_jammer['D_G2_rf_output']]
        D_rf_label = [output_jammer['D_G1_rf_label'], output_jammer['D_G2_rf_label']]
        G_rf_label = [output_jammer['G1_G1_rf_label'], output_jammer['G2_G2_rf_label']]
        J_rf_label = [output_jammer['J_G1_rf_label'], output_jammer['J_G2_rf_label']]
        # 遍歷每個類別計算損失
        for i in range(Parameter.num_classes):
            # --- 判別器的分類損失 ---
            D_pos_loss_class = CE_loss(D_pos_cls_label[i], D_pos_cls_logit[i])
            D_loss_class_list.append(D_pos_loss_class)
            D_neg_loss_class = CE_loss(D_neg_cls_label[i], D_neg_cls_logit[i])
            D_loss_class_list.append(D_neg_loss_class)

            # --- 生成器的分類損失 ---
            G_pos_loss_class = CE_loss(G_pos_cls_label[i], D_pos_cls_logit[i])
            G_loss_class_list.append(G_pos_loss_class)
            G_neg_loss_class = CE_loss(G_neg_cls_label[i], D_neg_cls_logit_swapped[i])
            G_loss_class_list.append(G_neg_loss_class)

            # --- 干擾器的分類損失 ---
            J_pos_loss_class = CE_loss(J_pos_cls_label[i], D_pos_cls_logit[i])
            J_loss_class_list.append(J_pos_loss_class)
            J_neg_loss_class = CE_loss(J_neg_cls_label[i], D_neg_cls_logit[i])
            J_loss_class_list.append(J_neg_loss_class)
            # --- 判別器的真假損失 ---
            D_loss_real_fake = BCE_loss(D_rf_label[i], D_rf_output[i])
            D_loss_real_fake_list.append(D_loss_real_fake)

            # --- 生成器的真假損失 ---
            G_loss_real_fake = BCE_loss(G_rf_label[i], D_rf_output[i])
            G_loss_real_fake_list.append(G_loss_real_fake)
            # --- 干擾器的真假損失 ---
            J_loss_real_fake = BCE_loss(J_rf_label[i], D_rf_output[i])
            J_loss_real_fake_list.append(J_loss_real_fake)

        return D_loss_class_list, G_loss_class_list, J_loss_class_list, D_loss_real_fake_list, G_loss_real_fake_list, J_loss_real_fake_list

    def calculate_gen_losses(self, output_class_logits, output_real_fake_logits,
                             output_label_class_D, output_label_class_G,
                             output_rf_label_D, output_rf_label_G):
        """
        Args:
            output_class_logits: List of classification logits for generated data (G(z)).
            output_real_fake_logits: List of real/fake logits for generated data (G(z)).
            output_label_class_D: Labels for Discriminator's classification task.
            output_label_class_G: Labels for Generator's classification task.
            output_rf_label_D: Labels for Discriminator's real/fake task.
            output_rf_label_G: Labels for Generator's real/fake task.

        Returns:
            D_loss_class_list: List of classification losses for updating D_class.
            G_loss_class_list: List of classification losses for updating G.
            D_loss_real_fake_list: List of real/fake losses for updating D_rf.
            G_loss_real_fake_list: List of real/fake losses for updating G.
        """
        # 分類損失 (SparseCategoricalCrossentropy)
        CE_loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        # 真/假損失 (BinaryCrossentropy)
        BCE_loss = tf.keras.losses.BinaryCrossentropy(from_logits=False)

        # 初始化損失值
        D_loss_class_list, G_loss_class_list = [], []
        D_loss_real_fake_list, G_loss_real_fake_list = [], []

        # 遍歷每個類別計算損失
        for i in range(Parameter.num_classes):
            # --- 判別器的分類損失 ---
            D_loss_class = CE_loss(output_label_class_D[i], output_class_logits[i])
            D_loss_class_list.append(D_loss_class)

            # --- 生成器的分類損失 ---
            G_loss_class = CE_loss(output_label_class_G[i], output_class_logits[i])
            G_loss_class_list.append(G_loss_class)

            # --- 判別器的真假損失 ---
            D_loss_real_fake = BCE_loss(output_rf_label_D[i], output_real_fake_logits[i])
            D_loss_real_fake_list.append(D_loss_real_fake)

            # --- 生成器的真假損失 ---
            G_loss_real_fake = BCE_loss(output_rf_label_G[i], output_real_fake_logits[i])
            G_loss_real_fake_list.append(G_loss_real_fake)

        return D_loss_class_list, G_loss_class_list, D_loss_real_fake_list, G_loss_real_fake_list

    def take_jammer(self, jammer, training=False):
        """
        Args:
            jammer: A list containing jammer outputs in the form of [(64, 200), (64, 200)]
                    e.g., jammer = [jammed_1, jammed_2]
                    where jammed_1 = J(G1, R2) and jammed_2 = J(G2, R1)

        Returns:
            output_jammer: Dictionary containing logits and labels required for loss calculation.
        """
        # Initialize dictionary to store the final outputs
        output_jammer = {}

        # Process jammer pairs (G1, R2) and (G2, R1)
        for i in range(2):  # Iterate through jammer pairs

            if i == 0:  # jammer(G1, R2)
                jammed_output = jammer[i]

                # D1 outputs (G+x-)
                logits_D1 = self.Discriminator_list[0](jammed_output, training=training)
                pred_D1_rf = self.disc_rf(jammed_output, training=training)
                batch_size = tf.shape(logits_D1)[0]
                output_jammer[f'D1_G{i + 1}_cls_logits'] = logits_D1
                output_jammer[f'D_G{i + 1}_rf_output'] = pred_D1_rf
                output_jammer[f'D1_G{i + 1}_cls_label'] = tf.ones((batch_size,))  # Class 1   D_cls+ 5
                output_jammer[f'D_G{i + 1}_rf_label'] = tf.zeros((batch_size, 1))  # Real      D_rf 5

                # D2 outputs (G+x-)
                logits_D2 = self.Discriminator_list[1](jammed_output, training=training)
                # pred_D2_rf = self.disc_rf(jammed_output, training=training)
                output_jammer[f'D2_G{i + 1}_cls_logits'] = logits_D2
                output_jammer[f'D2_G{i + 1}_cls_label'] = tf.zeros((batch_size,))  # Class 1    D_cls- 6
                # J
                output_jammer[f'J_D2_G{i + 1}_cls_label'] = tf.ones((batch_size,))  # Class 1  J3
                # G
                output_jammer[f'G1_D2_G{i + 1}_cls_label'] = tf.zeros((batch_size,))  # Class 1  G+3    改
                # G1 labels (G+x-)
                output_jammer[f'G1_G{i + 1}_cls_label'] = tf.zeros((batch_size,))  # Not class 1  G+2
                output_jammer[f'G1_G{i + 1}_rf_label'] = tf.ones((batch_size, 1))  # Fake (generated) G+5

                # J labels (G+x-)   第一個從1改0
                output_jammer[f'J_D1_G{i + 1}_cls_label'] = tf.zeros((batch_size,))  # Class 1  J1
                output_jammer[f'J_G{i + 1}_rf_label'] = tf.ones((batch_size, 1))  # Real  J5

            elif i == 1:  # jammer(G2, R1)
                jammed_output = jammer[i]
                # D2 outputs (G-x+)
                logits_D2 = self.Discriminator_list[1](jammed_output, training=training)
                batch_size = tf.shape(logits_D2)[0]
                pred_D2_rf = self.disc_rf(jammed_output, training=training)
                output_jammer[f'D2_G{i + 1}_cls_logits'] = logits_D2
                output_jammer[f'D_G{i + 1}_rf_output'] = pred_D2_rf
                output_jammer[f'D2_G{i + 1}_cls_label'] = tf.ones((batch_size,))  # Class 1  D_cls- 5
                output_jammer[f'D_G{i + 1}_rf_label'] = tf.zeros((batch_size, 1))  # Real     D_rf 6

                # D1 outputs (G-x+)
                logits_D1 = self.Discriminator_list[0](jammed_output, training=training)
                # pred_D1_rf = self.disc_rf(jammed_output, training=training)
                # 更新誰_(G+X- or G-X+)
                output_jammer[f'D1_G{i + 1}_cls_logits'] = logits_D1
                output_jammer[f'D1_G{i + 1}_cls_label'] = tf.zeros((batch_size,))  # Class 1  D_cls+ 6
                # D1 J
                output_jammer[f'J_D1_G{i + 1}_cls_label'] = tf.zeros((batch_size,))  # Not class 1  J2
                # G  更新誰_哪個D_(G+X- or G-X+)
                output_jammer[f'G2_D1_G{i + 1}_cls_label'] = tf.zeros((batch_size,))  # Class 1  G-3   改
                # G2 labels (G-x+)
                output_jammer[f'G2_G{i + 1}_cls_label'] = tf.zeros((batch_size,))  # Not class 1  G-2
                output_jammer[f'G2_G{i + 1}_rf_label'] = tf.ones((batch_size, 1))  # Fake (generated)  G-5

                # J2 labels (G-x+)  第一個從0改1
                output_jammer[f'J_D2_G{i + 1}_cls_label'] = tf.ones((batch_size,))  # Not class 1  J4
                output_jammer[f'J_G{i + 1}_rf_label'] = tf.ones((batch_size, 1))  # Real  J6

        return output_jammer

    def evaluate(self, val_data, val_label):

        predictions = self.classifier(val_data, training=False)
        # calculate acc
        correct_count = 0
        total_count = 0

        # test labels
        val_labels_array = np.array(val_label)
        # get max index as predict label
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

    def compute_accuracy(self, org_data, generated_data):
        D_accuracies = []
        G_accuracies = []

        for c in range(Parameter.num_classes):
            real_data = org_data[c]  # 取出第 i 類的真實數據
            fake_data = generated_data[c]  # 取出第 i 類的生成數據

            # D 判斷真實數據
            real_pred = self.Discriminator_list[c](real_data)  # (batch_size, num_classes)
            real_labels = tf.cast(tf.fill([tf.shape(real_pred)[0]], 1), dtype=tf.int64) # 真實標籤 (全為 i)
            real_acc = tf.reduce_mean(tf.cast(tf.sigmoid(real_pred) > 0.5, tf.float32))

            # D 判斷生成數據
            fake_pred = self.Discriminator_list[c](fake_data)
            # fake_labels = tf.cast(tf.fill([tf.shape(fake_pred)[0]], 0), dtype=tf.int64)
            fake_acc = tf.reduce_mean(tf.cast(tf.sigmoid(fake_pred) < 0.5, tf.float32))  # 錯誤分類的比例

            fake_g_labels = tf.cast(tf.fill([tf.shape(fake_pred)[0]], 1), dtype=tf.int64)
            # 計算 D 和 G 的準確率
            D_accuracy = (real_acc + fake_acc) / 2

            G_accuracy = tf.reduce_mean(
                tf.cast(tf.sigmoid(fake_pred) > 0.5, tf.float32)  # 超過 0.5 當作 1
            )
            D_accuracies.append(D_accuracy)
            G_accuracies.append(G_accuracy)

            print(f"Class {c} - D Accuracy: {D_accuracy.numpy():.4f}, G Accuracy: {G_accuracy.numpy():.4f}")

        return D_accuracies, G_accuracies

    def save_all_weights(self, save_dir):
        os.makedirs(save_dir, exist_ok=True)

        # 保存 Generators 權重
        for i, generator in enumerate(self.Generator_list):
            dir_path = f"{save_dir}/generator/generator_{i}"
            file_path = f"{dir_path}/EE_{Parameter.name}_G_weights"

            if not os.path.exists(dir_path):
                os.makedirs(dir_path)  # 创建目录
            if os.path.exists(file_path):
                os.remove(file_path)  # 删除已经存在的文件，确保不会重复创建组
            generator.save_weights(file_path, save_format='tf')
            print(f"Saved weights for generator {i} to {file_path}")

        # 保存 Discriminators 權重
        for i, discriminator in enumerate(self.Discriminator_list):
            dir_path = f"{save_dir}/discriminator/discriminator_{i}"
            file_path = f"{dir_path}/EE_{Parameter.name}_D_weights"
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)  # 创建目录
            if os.path.exists(file_path):
                os.remove(file_path)  # 删除已经存在的文件，确保不会重复创建组
            discriminator.save_weights(file_path, save_format='tf')
            print(f"Saved weights for discriminator {i} to {file_path}")

            # 保存 Drf 權重
            dir_path = f"{save_dir}/Discriminator_rf/Drf"
            file_path = f"{dir_path}/EE_{Parameter.name}_Drf_weights"
            # 確保目錄存在
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)
            # 如果檔案已經存在，先刪除確保不重複
            if os.path.exists(file_path):
                os.remove(file_path)

            # 儲存 `Drf` 的權重
            self.disc_rf.save_weights(file_path, save_format='tf')
            print(f"Saved weights for Drf {i} to {file_path}")

            print(f"All weights saved to {save_dir}")

        # 保存 Jammers 權重
        dir_path = f"{save_dir}/jammer/jammer"
        file_path = f"{dir_path}/EE_{Parameter.name}_J_weights"
        # 確保目錄存在
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        # 如果檔案已經存在，先刪除確保不重複
        if os.path.exists(file_path):
            os.remove(file_path)

        # 儲存 `Jammer` 的權重
        self.jammer.save_weights(file_path, save_format='tf')
        print(f"Saved weights for jammer to {file_path}")

        print(f"All weights saved to {save_dir}")

    def load_all_weights(self, save_dir):
        for i, generator in enumerate(self.Generator_list):
            dummy_input = tf.random.normal(shape=(batch_size, units))  # 替换为实际的输入形状
            _ = generator(dummy_input, training=False)
            file_path = f"{save_dir}/generator/generator_{i}/EE_{Parameter.name}_G_weights"
            print(i)
            generator.load_weights(file_path)
            print(f"Loaded weights for generator {i} from {file_path}")

        for i, discriminator in enumerate(self.Discriminator_list):
            dummy_input = tf.random.normal(shape=(batch_size, units))  # 替换为实际的输入形状
            _ = discriminator(dummy_input, training=False)
            file_path = f"{save_dir}/discriminator/discriminator_{i}/EE_{Parameter.name}_D_weights"
            discriminator.load_weights(file_path)
            print(f"Loaded weights for discriminator {i} from {file_path}")

        # load Drf weights
        dummy_input = tf.random.normal(shape=(batch_size, units))  # 替换为实际的输入形状
        # 使用 `dummy_input` 作為輸入進行一次前向傳播，以構建 Jammer 模型
        _ = self.disc_rf(dummy_input, training=False)
        # 設定要載入的權重路徑
        file_path = f"{save_dir}/Discriminator_rf/Drf/EE_{Parameter.name}_Drf_weights"

        # 打印當前加載進度
        print(f"Loading weights for Drf {i} from {file_path}")
        # 載入權重
        self.disc_rf.load_weights(file_path)

        dummy_generated = tf.random.normal(shape=(batch_size, units))  # 生成的假數據輸入
        dummy_real = tf.random.normal(shape=(batch_size, units))  # 真實的數據輸入
        # 使用 `dummy_generated` 和 `dummy_real` 作為輸入進行一次前向傳播，以構建 Jammer 模型
        _ = self.jammer(dummy_generated, dummy_real, training=False)
        # 設定要載入的權重路徑
        file_path = f"{save_dir}/jammer/jammer/EE_{Parameter.name}_J_weights"

        # 打印當前加載進度
        print(f"Loading weights for jammer {i} from {file_path}")
        # 載入權重
        self.jammer.load_weights(file_path)

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
        plt.savefig(f'EE_ours_{Parameter.name}_TSNE.png', dpi=300, bbox_inches='tight')
        np.savez(
            f'EE_ours_{Parameter.name}_TSNE_data.npz',
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
        with open(f'EE_ours_{Parameter.name}_TSNE_data.pkl', 'wb') as f:
            pickle.dump(tsne_dict, f)
        # plt.show()


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
    initial_learning_rate = 0.0001
    all_runs_acc, all_runs_auroc = 0.0, 0.0
    acc_std, auroc_std = 0.0, 0.0
    cls_epoch = 200
    batch_size = 128
    gan_epochs = 50
    total_training_epoch = 500

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


        gan_model = Jam_GAN(units)

        # # 加載function_f權重
        f_model = gan_model.function_f
        dummy_input = tf.random.normal(shape=(batch_size, Parameter.hidden_dim))  # 替換為實際的輸入形狀和數據
        _ = gan_model.function_f(dummy_input, training=False)
        # load weight
        f_model.load_weights('./EE_weight/function_f/f_model_weights_{}.h5'.format(Parameter.name))

        # 印出新模型的摘要
        f_model.summary()

        classifier = gan_model.classifier
        dummy_input2 = tf.random.normal(shape=(batch_size, units))
        _ = classifier(dummy_input2, training=False)
        classifier.load_weights('./EE_weight/classifier_g/g_model_weights_{}.h5'.format(Parameter.name))

        optimizer = tf.keras.optimizers.Adam(learning_rate=initial_learning_rate)

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

        # Let data go through function_f
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
        history = gan_model.fit(train_data, train_label, test_phi_data_tensor, val_label_tensor,
                                      batch_size=batch_size, num_epochs=gan_epochs, cls_epochs=cls_epoch,
                                        mode='gdj', training=True)

        acc, auroc, f1 = history['acc'], history['auroc'], history['f1']
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
    print("All Runs f1 = ", all_runs_f1)
    print("All Runs f1 std = ", f1_std)
    # 將結果寫入文字檔
    with open(f"EE_ours_{Parameter.name}_results.txt", "w") as file:
        file.write(f"All Runs Acc = {all_runs_acc}\n")
        file.write(f"All Runs std = {acc_std}\n")
        file.write(f"All Runs AUROC = {all_runs_auroc}\n")
        file.write(f"All Runs AUROC std = {auroc_std}\n")
        file.write(f"All Runs F1 = {all_runs_f1}\n")
        file.write(f"All Runs F1 std = {f1_std}\n")
    print(f"結果已寫入 EE_ours_{Parameter.name}_results.txt")





