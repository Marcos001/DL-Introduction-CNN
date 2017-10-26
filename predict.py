
from data import load_test_data

def get_data():

    X_test, Y_test = load_test_data()

    print(X_test.shape, len(X_test)) # matrix das imagens
    print(Y_test.shape, len(Y_test)) # label certo das imagens
    print(Y_test)





if __name__ == '__main__':
    """"""
    # aquisição de imagen
    get_data()

    # carregar o modelo

    # fazer a predição