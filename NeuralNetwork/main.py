from number_network import NumberNetwork
from canvas import Canvas

if __name__ == "__main__":
    network = NumberNetwork()
    # acc = 0
    # while True:
    #     network.train_with_dataset()
    #     new_acc = network.test_with_dataset()
    #     if new_acc <= acc:
    #         break
    #     acc = new_acc
    #     print(network.lr, acc)
    #     network.lr *= 0.90
    # network.save_weights_biases()

    network.load_weights_biases()
    print(network.test_with_dataset())


    c = Canvas()
    c.draw()
    ip, label = c.get_input()
    network.test(ip, label)
    # print(network.test(ip, label))