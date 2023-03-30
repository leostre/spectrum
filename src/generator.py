from spectrum import Spectrum
from sklearn.decomposition import PCA
from matrix import Matrix
import matplotlib.pyplot as plt

class Generator:
    def __init__(self, matrix):
        self.mtr = matrix


if __name__ == '__main__':
    print('GENERATOR')
    from scan import get_spectra_list
    import pandas as pd
    spa = get_spectra_list(path=r'C:\Users\user\PycharmProjects\spectrum\new_data', recursive=True, classify=True)
    healthy = Matrix.create_matrix(spa, {}, lambda x: 'heal' in x.clss)
    healthy_2d = healthy.as_2d_array()
    pca = PCA(24)
    pca_healthy = pd.DataFrame(pca.fit_transform(healthy_2d), columns=['pca_' + str(i) for i in range(24)])
    plt.show()



