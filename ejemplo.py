import mnist_loader
import network
import pickle

training_data, validation_data , test_data = mnist_loader.load_data_wrapper()

training_data = list(training_data)
test_data = list(test_data)

#net=network.Network([784,30,10])

#net.SGD( training_data, 15, 10, 0.1, test_data=test_data)

#archivo = open("red_prueba1.pkl",'wb')
#pickle.dump(net,archivo)
#archivo.close()
#exit()
#leer el archivo

archivo_lectura = open("red_prueba.pkl",'rb')
net = pickle.load(archivo_lectura)
archivo_lectura.close()

# Debido a que hemos actualizado la función de costo, al llamar el código
# network, este código no está reconociendo a la función de costo nueva
# Cross Entropy, por lo que debemos asegurarnos de que la red contenga
# a la función, así pues la importamos de nuestro otro código
from network import CrossEntropyCost
if not hasattr(net, "cost"):   # Pues el error que nos sale anteriormente
    # es que no tiene el atributo cost
    net.cost = CrossEntropyCost # Así llamamos a nuestra nueva función como
    # la anterior para no alterar más el código

#Ahora tenemos que entrenar la red de nuevo
net.SGD( training_data, 15, 50, 0.5, test_data=test_data)

archivo = open("red_prueba.pkl",'wb')
pickle.dump(net,archivo)
archivo.close()
exit()

#esquema de como usar la red :
imagen = leer_imagen("disco.jpg")
print(net.feedforward(imagen))