import nn

def weight(input, output):
    return nn.Parameter(input, output)

def bias(output):
    return nn.Parameter(1, output)

class PerceptronModel(object):
    def __init__(self, dimensions):
        """
        Initialize a new Perceptron instance.

        A perceptron classifies data points as either belonging to a particular
        class (+1) or not (-1). `dimensions` is the dimensionality of the data.
        For example, dimensions=2 would mean that the perceptron must classify
        2D points.
        """
        self.w = nn.Parameter(1, dimensions)

    def get_weights(self):
        """
        Return a Parameter instance with the current weights of the perceptron.
        """
        return self.w

    def run(self, x):
        """
        Calculates the score assigned by the perceptron to a data point x.

        Inputs:
            x: a node with shape (1 x dimensions)
        Returns: a node containing a single number (the score)
        """
        "*** YOUR CODE HERE ***"
        return nn.DotProduct(self.get_weights(), x)

    def get_prediction(self, x):
        """
        Calculates the predicted class for a single data point `x`.

        Returns: 1 or -1
        """
        "*** YOUR CODE HERE ***"
        if nn.as_scalar(self.run(x))>=0:
            return 1
        return -1

    def train(self, dataset):
        """
        Train the perceptron until convergence.
        """
        "*** YOUR CODE HERE ***"
        accuracy = 0
        while accuracy < 1:
            trial = 0
            correct = 0
            for x, y in dataset.iterate_once(1):
                trial+=1
                if self.get_prediction(x) == nn.as_scalar(y):
                    correct+=1
                else:
                    self.get_weights().update(x, nn.as_scalar(y))
            accuracy=correct/trial

class RegressionModel(object):
    """
    A neural network model for approximating a function that maps from real
    numbers to real numbers. The network should be sufficiently large to be able
    to approximate sin(x) on the interval [-2pi, 2pi] to reasonable precision.
    """

    def __init__(self):
        # Initialize your model parameters here
        "*** YOUR CODE HERE ***"
        self.learningRate = -0.005
        self.batchSize = 5
        self.weight30 = weight(1,30)
        self.bias30 = bias(30)
        self.weight1 = weight(30,1)
        self.bias1 = bias(1)
        self.parameters = [self.weight30, self.bias30, self.weight1, self.bias1]

    def run(self, x):
        """
        Runs the model for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
        Returns:
            A node with shape (batch_size x 1) containing predicted y-values
        """
        "*** YOUR CODE HERE ***"
        # layer1 = nn.Linear(x, self.weight30)        # Shape: batch x 30
        # layer2 = nn.AddBias(layer1, self.bias30)    # Shape: batch x 30
        # layer3 = nn.ReLU(layer2)                    # Shape: batch x 30
        # layer4 = nn.Linear(layer3, self.weight1)    # Shape: batch x 1
        # finalLayer = nn.AddBias(layer4, self.bias1) # Shape: batch x 1
        layer1 = nn.ReLU(nn.AddBias(nn.Linear(x, self.weight30), self.bias30)) # Shape: inputSize x 30
        finalLayer = nn.AddBias(nn.Linear(layer1, self.weight1), self.bias1) # Shape: inputSize x 1
        return finalLayer

    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
            y: a node with shape (batch_size x 1), containing the true y-values
                to be used for training
        Returns: a loss node
        """
        "*** YOUR CODE HERE ***"
        loss = nn.SquareLoss(self.run(x), y)
        return loss

    def train(self, dataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"
        lossTotal=trial=0
        finalLoss = 1
        while finalLoss > 0.02:
            for x, y in dataset.iterate_once(self.batchSize):
                trial+=1
                loss = self.get_loss(x,y)
                grads = nn.gradients(loss, self.parameters)
                for param in self.parameters:
                    param.update(grads[self.parameters.index(param)], self.learningRate)
                lossTotal += nn.as_scalar(loss)
                finalLoss = lossTotal/trial
class DigitClassificationModel(object):
    """
    A model for handwritten digit classification using the MNIST dataset.

    Each handwritten digit is a 28x28 pixel grayscale image, which is flattened
    into a 784-dimensional vector for the purposes of this model. Each entry in
    the vector is a floating point number between 0 and 1.

    The goal is to sort each digit into one of 10 classes (number 0 through 9).

    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """

    def __init__(self):
        # Initialize your model parameters here
        "*** YOUR CODE HERE ***"
        self.learningRate = -0.015
        self.batchSize = 10

        # self.weight10 = nn.Parameter(784,10)
        # self.bias10 = nn.Parameter(1,10)
        # self.parameters = [self.weight10, self.bias10]

        # self.weight40 = nn.Parameter(784,40)
        # self.bias40 = nn.Parameter(1,40)
        # self.weight10 = nn.Parameter(40,10)
        # self.bias10 = nn.Parameter(1,10)
        # self.parameters = [self.weight40, self.bias40, self.weight10, self.bias10]

        self.weight54 = weight(784,256)
        self.bias54 = bias(256)
        self.weight20 = weight(256,40)
        self.bias20 = bias(40)
        self.weight10 = weight(40,10)
        self.bias10 = bias(10)
        self.parameters = [self.weight54, self.bias54, self.weight20, self.bias20, self.weight10, self.bias10]

    def run(self, x):
        """
        Runs the model for a batch of examples.

        Your model should predict a node with shape (batch_size x 10),
        containing scores. Higher scores correspond to greater probability of
        the image belonging to a particular class.

        Inputs:
            x: a node with shape (batch_size x 784)
        Output:
            A node with shape (batch_size x 10) containing predicted scores
                (also called logits)
        """
        "*** YOUR CODE HERE ***"

        # finalLayer = nn.AddBias(nn.Linear(x, self.weight10), self.bias10)

        # layer1 = nn.Linear(x, self.weight40)         # Shape: batch x 40
        # layer2 = nn.AddBias(layer1, self.bias40)     # Shape: batch x 40
        # layer3 = nn.ReLU(layer2)                     # Shape: batch x 40
        # layer4 = nn.Linear(layer3, self.weight10)    # Shape: batch x 10
        # finalLayer = nn.AddBias(layer4, self.bias10) # Shape: batch x 10

        layer1 = nn.ReLU(nn.AddBias(nn.Linear(x, self.weight54),
                                        self.bias54)) # Shape: inputSize x 256
        layer2 = nn.ReLU(nn.AddBias(nn.Linear(layer1, self.weight20),
                                        self.bias20)) # Shape: inputSize x 40
        finalLayer = nn.AddBias(nn.Linear(layer2, self.weight10),
                                        self.bias10) # Shape: inputSize x 10
        return finalLayer



    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        The correct labels `y` are represented as a node with shape
        (batch_size x 10). Each row is a one-hot vector encoding the correct
        digit class (0-9).

        Inputs:
            x: a node with shape (batch_size x 784)
            y: a node with shape (batch_size x 10)
        Returns: a loss node
        """
        "*** YOUR CODE HERE ***"
        return nn.SoftmaxLoss(self.run(x), y)

    def train(self, dataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"
        while dataset.get_validation_accuracy() < 0.975:
            for x, y in dataset.iterate_once(self.batchSize):
                loss = self.get_loss(x,y)
                grads = nn.gradients(loss, self.parameters)
                for param in self.parameters:
                    param.update(grads[self.parameters.index(param)], self.learningRate)
                # print(dataset.get_validation_accuracy())

class LanguageIDModel(object):
    """
    A model for language identification at a single-word granularity.

    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """

    def __init__(self):
        # Our dataset contains words from five different languages, and the
        # combined alphabets of the five languages contain a total of 47 unique
        # characters.
        # You can refer to self.num_chars or len(self.languages) in your code
        self.num_chars = 47
        self.languages = ["English", "Spanish", "Finnish", "Dutch", "Polish"]

        # Initialize your model parameters here
        "*** YOUR CODE HERE ***"
        self.learningRate = -0.01
        self.batchSize = 5
        self.hiddenDim = 200
        self.w_init = weight(self.num_chars, self.hiddenDim)
        self.b_init = bias(self.hiddenDim)
        self.w_loop = weight(self.hiddenDim, self.hiddenDim)
        self.b_loop = bias(self.hiddenDim)
        self.w_final = weight(self.hiddenDim, len(self.languages))
        self.b_final = bias(len(self.languages))
        self.parameters = [self.w_init, self.b_init, self.w_loop, self.b_loop, self.w_final, self.b_final]

    def run(self, xs):
        """
        Runs the model for a batch of examples.

        Although words have different lengths, our data processing guarantees
        that within a single batch, all words will be of the same length (L).

        Here `xs` will be a list of length L. Each element of `xs` will be a
        node with shape (batch_size x self.num_chars), where every row in the
        array is a one-hot vector encoding of a character. For example, if we
        have a batch of 8 three-letter words where the last word is "cat", then
        xs[1] will be a node that contains a 1 at position (7, 0). Here the
        index 7 reflects the fact that "cat" is the last word in the batch, and
        the index 0 reflects the fact that the letter "a" is the inital (0th)
        letter of our combined alphabet for this task.

        Your model should use a Recurrent Neural Network to summarize the list
        `xs` into a single node of shape (batch_size x hidden_size), for your
        choice of hidden_size. It should then calculate a node of shape
        (batch_size x 5) containing scores, where higher scores correspond to
        greater probability of the word originating from a particular language.

        Inputs:
            xs: a list with L elements (one per character), where each element
                is a node with shape (batch_size x self.num_chars)
        Returns:
            A node with shape (batch_size x 5) containing predicted scores
                (also called logits)
        """
        "*** YOUR CODE HERE ***"
        layer = nn.ReLU(nn.AddBias(nn.Linear(xs[0], self.w_init),self.b_init))
        for x in xs[1:]:
            layer = nn.ReLU(nn.Add(nn.AddBias(nn.Linear(x, self.w_init), self.b_init),
                                   nn.AddBias(nn.Linear(layer, self.w_loop), self.b_loop)))
        finalLayer = nn.AddBias(nn.Linear(layer, self.w_final), self.b_final)
        return finalLayer

    def get_loss(self, xs, y):
        """
        Computes the loss for a batch of examples.

        The correct labels `y` are represented as a node with shape
        (batch_size x 5). Each row is a one-hot vector encoding the correct
        language.

        Inputs:
            xs: a list with L elements (one per character), where each element
                is a node with shape (batch_size x self.num_chars)
            y: a node with shape (batch_size x 5)
        Returns: a loss node
        """
        "*** YOUR CODE HERE ***"
        return nn.SoftmaxLoss(self.run(xs), y)

    def train(self, dataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"
        while dataset.get_validation_accuracy() < 0.85:
            for x, y in dataset.iterate_once(self.batchSize):
                loss = self.get_loss(x,y)
                grads = nn.gradients(loss, self.parameters)
                for param in self.parameters:
                    param.update(grads[self.parameters.index(param)], self.learningRate)
