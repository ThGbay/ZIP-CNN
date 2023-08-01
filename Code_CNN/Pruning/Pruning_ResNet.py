#!/usr/bin/env python
# coding: utf-8

# ## Bibiliothèque




import numpy as np
import tensorflow as tf;
import matplotlib.pyplot as plt
from resnet import resnet_v2
import time

seed = tf.random.set_seed(10)




def plot_hist(figname):
    plt.figure(figsize=(15,7))
    plt.subplot(121)
    plt.plot(accuracy, label = "train accuracy")
    plt.plot(val_accuracy, label = "validation accuracy")
    plt.title("Accuracy")
    plt.grid()
    plt.legend()

    plt.subplot(122)
    plt.plot(l, label = "train loss")
    plt.plot(val_l, label = "validation loss")
    plt.title("Loss")
    plt.grid()
    plt.legend()

    plt.savefig(figname)
    plt.show()
    
def inference_time():
    scratch = []
    pruned = []
    for i in range(10):
        t1 = time.time()
        pred1 = scratch_model(x_test)
        t2 = time.time()
        scratch.append(t2-t1)

        # Pruned model
        t3 = time.time()
        pred2 = P.model(x_test)
        t4 = time.time()
        pruned.append(t4-t3)

    # display
    print("Scratch inference time : ", np.mean(scratch), " s")
    print("Pruned inference time : ", np.mean(pruned), " s")
    return np.mean(pruned)


def count_parameters(model):
    somme = 0
    for l in model.trainable_variables:
        somme += np.count_nonzero(l)
    return somme


def scratch_hist():   
    loss = dico["scratch_hist"][0].history["loss"]
    val_loss = dico["scratch_hist"][0].history["val_loss"]
    accuracy = dico["scratch_hist"][0].history["sparse_categorical_accuracy"]
    val_accuracy =  dico["scratch_hist"][0].history["val_sparse_categorical_accuracy"]

    for i in range(len( dico["scratch_hist"])):
        if i !=0:
            loss = np.append(loss, dico["scratch_hist"][i].history["loss"])
            val_loss = np.append(val_loss, dico["scratch_hist"][i].history["val_loss"])
            accuracy = np.append(accuracy, dico["scratch_hist"][i].history["sparse_categorical_accuracy"])
            val_accuracy =  np.append(val_accuracy, dico["scratch_hist"][i].history["val_sparse_categorical_accuracy"])

    dico["scratch_hist"] = (accuracy, val_accuracy, loss, val_loss)


# ## Loading cifar10 Dataset


print("================ Data Loading ================")
(x_train, y_train),(x_test, y_test) = tf.keras.datasets.cifar10.load_data()

# Normalize data.
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

# Data shapes
print("x_train shape:", x_train.shape)
print("x_test shape:", x_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)
print("")


# ## Building Resnet8 model 



model = resnet_v2((32, 32, 3), depth = 8)

model.summary();


# ## Scratch Training



# disctionnaire pour enregistrer les infos pertinentes
dico = {}
scratch_model = tf.keras.models.clone_model(model)



BATCH_SIZE = 32
EPOCHS = 100
lr = 1


dico["scratch_hist"] = []
for EPOCHS in [75,15,10]:
    lr /= 10
    scratch_model.compile(
            optimizer = tf.keras.optimizers.SGD(learning_rate=lr),
            metrics = [tf.keras.metrics.SparseCategoricalAccuracy()],
            loss= tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
            )

        # Train and evaluate on data.
    hist = scratch_model.fit(x_train, y_train, 
          batch_size = BATCH_SIZE,
          epochs=EPOCHS,
          steps_per_epoch = len(x_train)/BATCH_SIZE,
          validation_data =(x_test, y_test),
          workers =40,
          use_multiprocessing= True,
          )

    scratch_model.evaluate(x_test, y_test)
    dico["scratch_hist"].append(hist)
scratch_hist()



# ## Pruning class


class Pruning:
    def __init__(self, model, pruning_factor = 0.5):
        
        # attributs liés au model
        self.model = model
        self.pruning_factor = pruning_factor
    
    # Tensorflow utils setting
    def compile(self,optimizer, loss_fn, metric):
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.acc_metric = metric
    
    # Pruning Function
    def pruning(self, P_factor = 0.5):
        if P_factor >=1 or P_factor <= 0:
            raise ValueError ("Pruning factor value Error : Pruning factor value should be ]0 ;1[")
        for layer in self.model.layers:
            if "conv" in layer.name:
                
                # Récuper les kernels
                w = layer.get_weights()[0]
                b = layer.get_weights()[1]
                
                # Calcul du filtre contenant la median
                tab = []
                for i in range(w.shape[-1]):
                    somme = 0
                    for j in range(w.shape[-1]):
                        if i !=j:
                            somme += np.linalg.norm(w[:,:,:,i] - w[:,:,:,j])
                    tab.append(somme)
                    
                # calcul du nombre de filtrer a annuler selon le facteur de pruning
                nb_pruned_filters = int(w.shape[-1]*P_factor)
                
                for i in range(nb_pruned_filters):
                    # récupérer l'indice du minimum
                    ind_min = np.argmin(tab)
                    
                    #anuuler le filtre qui minimise la formule précedente
                    w[:, :, :, ind_min] = np.zeros(w[:, :, :, ind_min].shape)
                    
                    # astuce pour déplacer le minimum lorsque il faut annuler plusieurs filtres
                    tab[ind_min] = 1e10
                
                layer.set_weights([w, b])


                
                
    #Training algorithm
    def train(self,x_train, y_train, val_data, val_labels, epochs = 100, batch_size= 32):
        self.epochs = epochs
        self.batch_size = batch_size
        
        # training history storage
        accur = []
        L = []
        
        # validation history storage
        v_accur = []
        v_loss = []

        if x_train.shape[0] % batch_size == 0:
            nb_train_steps = x_train.shape[0] // batch_size
        else:
            nb_train_steps = (x_train.shape[0] // batch_size) + 1
        # Training Loop
        for epoch in range(epochs):
            print(f"Epoch ({epoch +1 }/{epochs})")
            for i in range(nb_train_steps):
                # Batching data
                x = x_train[i*batch_size:(i+1)*batch_size]
                y = y_train[i*batch_size:(i+1)*batch_size]
                
                x = tf.constant(x)
                y = tf.constant(y)
                
                with tf.GradientTape() as tape:
                    # Forward pass
                    predictions = self.model(x, training=True)
                    # calcul de la loss
                    loss = self.loss_fn(y, predictions)
                    
                # Calcul du gradient
                grads = tape.gradient(loss, self.model.trainable_weights)
                
                # Decente de gradient
                self.optimizer.apply_gradients(zip(grads, self.model.trainable_weights))
                
                #Pruning step
                self.pruning(P_factor = self.pruning_factor)
                
                # Update training metric.
                self.acc_metric.update_state(y, predictions)
                train_acc = self.acc_metric.result()
                
                print("Accuracy: {:.4f} ; loss: {:.4f}".format(float(train_acc),  loss), end='\r')
            print("\nValidation Step :")
            
            # Validation step
            val_accur, val_loss = self.test(val_data, val_labels)
                
            accur.append(float(train_acc))
            L.append(loss)
            
            v_accur.append(val_accur)
            v_loss.append(val_loss)
            print("")
        return (accur, L, v_accur, v_loss)  

    
    # Test Step 
    def test(self,data, labels):
        accur = []
        l = []
        if data.shape[0] % self.batch_size == 0:
            nb_test_steps = data.shape[0] // self.batch_size
        else:
            nb_test_steps = (data.shape[0] // self.batch_size) + 1
            
        for i in range(nb_test_steps):
            # Batching data
            x = data[i*self.batch_size:(i+1)*self.batch_size]
            y = labels[i*self.batch_size:(i+1)*self.batch_size]
            
            x = tf.constant(x)
            y = tf.constant(y)
            
            # Forward pass
            predictions = self.model(x)

            # calcul de la loss
            loss = self.loss_fn(y, predictions)
            # calcul de l'accuracy
            self.acc_metric.update_state(y, predictions)
            test_acc = self.acc_metric.result()
            print("Accuracy: {:.4f} ; loss: {:.4f}".format(float(test_acc),  loss), end='\r')
                
            accur.append(float(test_acc))
            l.append(float(loss))
        print("")        
        print("Accuracy Moy : {:.4f} ; loss Moy: {:.4f}" .format(np.mean(accur), np.mean(l) ))
        
        return (np.mean(accur), np.mean(l))


# ## Training Network


for p in [0.5, 0.6, 0.7, 0.8, 0.9]:
    accuracy = np.array([])
    val_accuracy = np.array([])

    l = np.array([])
    val_l = np.array([])


    P = Pruning(tf.keras.models.clone_model(model), 
                pruning_factor =p)
    lr = 1
    for epoch in [75, 15, 10]:
        # Paramètre d'entrainement
        lr /= 10
        P.compile(optimizer = tf.keras.optimizers.SGD(learning_rate=lr),
                 loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                 metric = tf.keras.metrics.SparseCategoricalAccuracy(),
        )

        # Entrainement         
        accur, loss, val_accur, val_loss = P.train(x_train, y_train, 
                                                   x_test, y_test, 
                                                   epochs = epoch, 
                                                   batch_size= 32 )  


        # plot hist
        accuracy = np.append(accuracy, accur)
        val_accuracy = np.append(val_accuracy, val_accur)
        l = np.append(l, loss)
        val_l = np.append(val_l,val_loss)

    
    # Afficher les courbes d'entrainement
    #plot_hist(f"Lenet5_P_factor_{p}.png")
    
    # inference time
    pruned_inf_time = inference_time()
    
    # Enregister l'historique
    dico[f"P_factor_{p}_hist"] = (accuracy, val_accuracy, l, val_l)
    
    # calcul du temps d'inférence
    dico[f"P_factor_{p}_inf_time"] = pruned_inf_time

    
    # memory used
    dico[f"nb_params_p_factor_{p}"] = count_parameters(P.model)
    
    # sauvegarder les poids
    P.model.save_weights(f"w_Resnet8_p_{p}.h5")
    
    # Sauvegarder les données du dictionnaire
    np.save("summary_resnet8.npy", dico)
        


# ## Evaluation des performances


def eval_plot(dic, figname, scratch = False):
    plt.figure(figsize=(15,15))
    for p in [0.5, 0.6, 0.7, 0.8, 0.9]:
        # Train accuracy
        plt.subplot(221)
        plt.plot(dic[f"P_factor_{p}_hist"][0], label = f"{p}")

        plt.title("Train accuracy")
        plt.grid()
        plt.legend()

        # Validation accuracy
        plt.subplot(222)
        plt.plot(dico[f"P_factor_{p}_hist"][1], label = f"{p}")

        plt.title("Validation accuracy")
        plt.grid()
        plt.legend()

        # train loss
        plt.subplot(223)
        plt.plot(dic[f"P_factor_{p}_hist"][2], label = f"{p}")

        plt.title("Train loss")
        plt.grid()
        plt.legend()

        # validation loss
        plt.subplot(224)
        plt.plot(dic[f"P_factor_{p}_hist"][3], label = f"{p}")

        plt.title("Validation loss")
        plt.grid()
        plt.legend()
        
    if scratch == True: 
        # Courbe scratch
        plt.subplot(221)
        plt.plot(dic["scratch_hist"][0], label = "Scratch Train accur")
        plt.legend()
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy Value")

        plt.subplot(222)
        plt.plot(dic["scratch_hist"][1], label = "Scratch Val accur")
        plt.legend()
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy Value")

        plt.subplot(223)
        plt.plot(dic["scratch_hist"][2], label = "Scratch Train loss")
        plt.legend()
        plt.xlabel("Epochs")
        plt.ylabel("Loss Value")


        plt.subplot(224)
        plt.plot(dic["scratch_hist"][3], label = "Scratch Val loss")
        plt.legend()
        plt.xlabel("Epochs")
        plt.ylabel("Loss Value")
    
    
    plt.savefig(figname)
    plt.show()


figname= f"resnet8.png"
eval_plot(dico, figname, scratch = True)

