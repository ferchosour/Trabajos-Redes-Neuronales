# %load network.py

"""
network.py
~~~~~~~~~~
IT WORKS

A module to implement the stochastic gradient descent learning
algorithm for a feedforward neural network.  Gradients are calculated
using backpropagation.  Note that I have focused on making the code
simple, easily readable, and easily modifiable.  It is not optimized,
and omits many desirable features.
"""

#### Libraries
# Standard library
import random

# Third-party libraries
import numpy as np

### Primero implementaremos nuestra nueva función de costo Cross Entropy:
class CrossEntropyCost(object):
    @staticmethod #Esta línea es un método dentro de la clase que no necesita
    # la instancia de la clase, ni la clase misma, sirve para
    # organizar el código y así llamarlos directamente desde esta clase
    def fn(a, y):
        #Regresa  el costo asociado con la salida 'a' y la salida deseada 'y'
        epsilon = 1e-12     # El epsilon recomendado para esta parte
        a = np.clip(a, epsilon, 1. - epsilon)  # evitar log(0)
        return np.sum(np.nan_to_num(-y*np.log(a) - (1-y)*np.log(1-a)))

    @staticmethod
    def delta(z, a, y):
        # Regresa el error delta de la capa de salida
        return a - y

class Network(object):

    # Aquí añadimos los valores recomendados de Adam para beta1, beta2 y epsilon
    def __init__(self, sizes, cost=CrossEntropyCost,
                 beta1=0.9, beta2=0.99, epsilon=1e-8):
        """The list ``sizes`` contains the number of neurons in the
        respective layers of the network.  For example, if the list
        was [2, 3, 1] then it would be a three-layer network, with the
        first layer containing 2 neurons, the second layer 3 neurons,
        and the third layer 1 neuron.  The biases and weights for the
        network are initialized randomly, using a Gaussian
        distribution with mean 0, and variance 1.  Note that the first
        layer is assumed to be an input layer, and by convention we
        won't set any biases for those neurons, since biases are only
        ever used in computing the outputs from later layers."""
        self.num_layers = len(sizes)
        self.sizes = sizes
        #self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        #self.weights = [np.random.randn(y, x)
                       # for x, y in zip(sizes[:-1], sizes[1:])]
        # Para una mejor inicialización de pesos, vamos a usar la inicialización
        # de Xavier, pues ajusta la varianza de los pesos según el tamaño de la capa
        #self.biases = [np.zeros((y, 1)) for y in sizes[1:]]  # Iniciamos en ceros para
        # tener una mejor estabilidad de la función sigmoide y evitar ruido
        # innecesario al principio
        #self.weights = [np.random.randn(y, x) * np.sqrt(1 / x)
                        #for x, y in zip(sizes[:-1], sizes[1:])]  # La mejora de inicialización Xavier
        # Añadimos esta función para que nuestra clase "Network" sepa qué función
        # de costo tiene que usar (Cross Entropy)
        self.cost = cost
        #Añadimos nuevos valores a nuestra clase Network para usar más adelante
        self.default_weight_initializer()   # Añadimos esto para reemplazar la antigua inicialización
        # y usar la inicialización Xavier
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.t = 0  #Pasos recomendados para Adam

    def default_weight_initializer(self):
        #Inicialización Xavier de una manera más eficiente
        self.biases = [np.random.randn(y, 1) for y in self.sizes[1:]]
        self.weights = [np.random.randn(y, x) / np.sqrt(x)
                        for x, y in zip(self.sizes[:-1], self.sizes[1:])]

    def feedforward(self, a):
        """Return the output of the network if ``a`` is input."""
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a)+b)
        return a

    def SGD(self, training_data, epochs, mini_batch_size, eta,
            test_data=None):
        """Train the neural network using mini-batch stochastic
        gradient descent.  The ``training_data`` is a list of tuples
        ``(x, y)`` representing the training inputs and the desired
        outputs.  The other non-optional parameters are
        self-explanatory.  If ``test_data`` is provided then the
        network will be evaluated against the test data after each
        epoch, and partial progress printed out.  This is useful for
        tracking progress, but slows things down substantially."""

        training_data = list(training_data)
        n = len(training_data)

        if test_data:
            test_data = list(test_data)
            n_test = len(test_data)

        for j in range(epochs):
            random.shuffle(training_data)
            mini_batches = [
                training_data[k:k+mini_batch_size]
                for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                # Acá estábamos usando un optimizador SGD simple, por lo que lo reemplazaremos
                # por parámetros que ayudarán a nuestro nuevo optimizador Adam
                    self.t += 1
                    self.update_mini_batch_adam(mini_batch, eta,
                                                self.beta1, self.beta2,
                                                self.epsilon, self.t)
            if test_data:
                print("Epoch {} : {} / {}".format(j,self.evaluate(test_data),n_test))
            else:
                print("Epoch {} complete".format(j))


# Esta parte no la vamos a usar ya, pues al implementar el nuevo optimizador, estaremos
# usando otros parámetros
    #def update_mini_batch(self, mini_batch, eta):
        #"""Update the network's weights and biases by applying
        #gradient descent using backpropagation to a single mini batch.
        #The ``mini_batch`` is a list of tuples ``(x, y)``, and ``eta``
        #is the learning rate."""
        #nabla_b = [np.zeros(b.shape) for b in self.biases]
        #nabla_w = [np.zeros(w.shape) for w in self.weights]
        #for x, y in mini_batch:
            #delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            #nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            #nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        #self.weights = [w-(eta/len(mini_batch))*nw
                        #for w, nw in zip(self.weights, nabla_w)]
        #self.biases = [b-(eta/len(mini_batch))*nb
                       #for b, nb in zip(self.biases, nabla_b)]

# Optimzador Adam
    def update_mini_batch_adam(self, mini_batch, eta, beta1, beta2, epsilon, t):

        # Vamos a inicializar momentos si es que no existen
        if not hasattr(self, "m_w"): # No lo mencioné anteriormente cuando renombré los parámetros
            # en ejemplo.py pero hasttr nos devuelve un true o un false, así que si no existen los
            # momentos entrarán en juego estos nuevos, que es lo que pasará en esta ocasión
            self.m_w = [np.zeros(w.shape) for w in self.weights]
            self.v_w = [np.zeros(w.shape) for w in self.weights]
            self.m_b = [np.zeros(b.shape) for b in self.biases]
            self.v_b = [np.zeros(b.shape) for b in self.biases]

        # Estos son los gradientes acumulados para los mini-batches
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        # Acá estamos realizando un algoritmo Backpropagation para cada muestra de mis mini-batches
        # y así obtener  un entrenamiento más estable y eficiente
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]

        # Aquí vamos a obtener el promedio del gradiente en el mini-batch, pues los
        # usaremos más adelante
        nabla_w = [nw / len(mini_batch) for nw in nabla_w]
        nabla_b = [nb / len(mini_batch) for nb in nabla_b]

        # Aquí iremos actualizando los parámetros y los momentos de Adam, estas son las fórmulas
        # que vimos en clase
        for i in range(len(self.weights)):
            # Estas dos son la primera media (momento) para los gradientes que acelera
            # la convergencia, tanto para los pesos como para los biases
            self.m_w[i] = beta1 * self.m_w[i] + (1 - beta1) * nabla_w[i]
            self.m_b[i] = beta1 * self.m_b[i] + (1 - beta1) * nabla_b[i]

            # Estas otras dos son la segunda media (RMSProp) para los gradientes al cuadrado
            # y así ajustar la tasa de aprendizaje en base a estos
            self.v_w[i] = beta2 * self.v_w[i] + (1 - beta2) * (nabla_w[i] ** 2)
            self.v_b[i] = beta2 * self.v_b[i] + (1 - beta2) * (nabla_b[i] ** 2)

            # Aquí realizamos una corrección de sesgo (así evitamos que m y v estén
            # "cerca de cero" en los primeros pasos, tal y como vimos en clase)
            m_hat_w = self.m_w[i] / (1 - beta1 ** t)
            v_hat_w = self.v_w[i] / (1 - beta2 ** t)
            m_hat_b = self.m_b[i] / (1 - beta1 ** t)
            v_hat_b = self.v_b[i] / (1 - beta2 ** t)

            # En esta parte actualizamos parámetros para cada paso que realizamos con los mini-batches
            self.weights[i] -= eta * m_hat_w / (np.sqrt(v_hat_w) + epsilon)
            self.biases[i]  -= eta * m_hat_b / (np.sqrt(v_hat_b) + epsilon)

    def backprop(self, x, y):
        """Return a tuple ``(nabla_b, nabla_w)`` representing the
        gradient for the cost function C_x.  ``nabla_b`` and
        ``nabla_w`` are layer-by-layer lists of numpy arrays, similar
        to ``self.biases`` and ``self.weights``."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # feedforward
        activation = x
        activations = [x] # list to store all the activations, layer by layer
        zs = [] # list to store all the z vectors, layer by layer
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation)+b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        # backward pass
        #delta = self.cost_derivative(activations[-1], y) * \
            #sigmoid_prime(zs[-1])
        #Ya no estamos usando la función de costo derivación autómatica,
        # ahora lo hacemos con la función de costo Cross Entropy
        delta = self.cost.delta(zs[-1], activations[-1], y)
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        # Note that the variable l in the loop below is used a little
        # differently to the notation in Chapter 2 of the book.  Here,
        # l = 1 means the last layer of neurons, l = 2 is the
        # second-last layer, and so on.  It's a renumbering of the
        # scheme in the book, used here to take advantage of the fact
        # that Python can use negative indices in lists.
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w)

    def evaluate(self, test_data):
        """Return the number of test inputs for which the neural
        network outputs the correct result. Note that the neural
        network's output is assumed to be the index of whichever
        neuron in the final layer has the highest activation."""
        test_results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    #def cost_derivative(self, output_activations, y):
        r"""Return the vector of partial derivatives \partial C_x /
        \partial a for the output activations."""
        return (output_activations - y)

#### Miscellaneous functions
def sigmoid(z):
    #Acá obtenemos una sigmoide numéricamente estable, pues me ha marcado un error que
    # indicaba que la función sigmoide era demasiado grande
    #return np.where(z >= 0,
                    #1.0 / (1.0 + np.exp(-z)),
                    #np.exp(z) / (1.0 + np.exp(z)))  # Aquí le indicamos a la máquina que
    # para valores positivos usamos una forma estándar de la sigmoide y para valores
    # negativos usamos una forma alternativa
    # La función anterior nos mostró todavía un overflow, así que limitaremos z a un rango,
# el cuál será de [-500,500]
    #Sigmoide clipada para evitar overflow
    z = np.clip(z, -500, 500)
    return 1.0 / (1.0 + np.exp(-z))

def sigmoid_prime(z):
    """Derivada de la sigmoide estable."""
    s = sigmoid(z)
    return s * (1 - s)