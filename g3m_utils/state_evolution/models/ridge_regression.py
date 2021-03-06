import numpy as np
from .base_model import Model


class RidgeRegression(Model):
    '''
    Implements updates for ridge regression task.
    See base_model for details on modules.
    '''
    def __init__(self, *, sample_complexity, regularisation, data_model):
        self.alpha = sample_complexity
        self.lamb = regularisation

        self.data_model = data_model

    def get_info(self):
        info = {
            'model': 'ridge_regression',
            'sample_complexity': self.alpha,
            'lambda': self.lamb,
        }
        return info

    def _update_overlaps(self, Vhat, qhat, mhat):
        if self.lamb>1e-6:
            #print("rescaled")
            V = np.mean(self.data_model.spec_Omega/(self.lamb + Vhat * self.data_model.spec_Omega))

            if self.data_model.commute:
                q = np.mean((self.data_model.spec_Omega**2 * qhat +
                            mhat**2 * self.data_model.spec_Omega * self.data_model.spec_PhiPhit) /
                            (self.lamb + Vhat*self.data_model.spec_Omega)**2)

                m = mhat/np.sqrt(self.data_model.gamma) * np.mean(self.data_model.spec_PhiPhit /
                                                        (self.lamb + Vhat*self.data_model.spec_Omega))

            else:
                q = qhat * np.mean(self.data_model.spec_Omega**2 / (self.lamb + Vhat*self.data_model.spec_Omega)**2)
                q += mhat**2 * np.mean(self.data_model._UTPhiPhiTU *
                                       self.data_model.spec_Omega /
                                       (self.lamb + Vhat * self.data_model.spec_Omega)**2)

                m = mhat/np.sqrt(self.data_model.gamma) * np.mean(self.data_model._UTPhiPhiTU/
                                                    (self.lamb + Vhat * self.data_model.spec_Omega))

        else:#rescale
            #print("rescaled")
            V = np.mean(self.data_model.spec_Omega/(1 + Vhat * self.data_model.spec_Omega))

            if self.data_model.commute:
                q = np.mean((self.data_model.spec_Omega**2 * qhat +
                            mhat**2 * self.data_model.spec_Omega * self.data_model.spec_PhiPhit) /
                            (1 + Vhat*self.data_model.spec_Omega)**2)

                m = mhat/np.sqrt(self.data_model.gamma) * np.mean(self.data_model.spec_PhiPhit /
                                                        (1 + Vhat*self.data_model.spec_Omega))

            else:
                q = qhat * np.mean(self.data_model.spec_Omega**2 / (1 + Vhat*self.data_model.spec_Omega)**2)
                q += mhat**2 * np.mean(self.data_model._UTPhiPhiTU *
                                       self.data_model.spec_Omega /
                                       (1 + Vhat * self.data_model.spec_Omega)**2)

                m = mhat/np.sqrt(self.data_model.gamma) * np.mean(self.data_model._UTPhiPhiTU/
                                                    (1 + Vhat * self.data_model.spec_Omega))

        return V, q, m

    def _update_hatoverlaps(self, V, q, m):
        if self.lamb>1e-6:
            #print("rescaled")
            Vhat = self.alpha * 1/(1+V)
            qhat = self.alpha * (self.data_model.rho + q - 2*m)/(1+V)**2
            mhat = self.alpha/np.sqrt(self.data_model.gamma) * 1/(1+V)
        else:
            #print("rescaled")
            Vhat = self.alpha * 1/(self.lamb+V)
            qhat = self.alpha * (self.data_model.rho + q - 2*m)/(self.lamb+V)**2
            mhat = self.alpha/np.sqrt(self.data_model.gamma) * 1/(self.lamb+V)

        return Vhat, qhat, mhat

    def update_se(self, V, q, m):
        Vhat, qhat, mhat = self._update_hatoverlaps(V, q, m)
        return self._update_overlaps(Vhat, qhat, mhat)


    def get_test_error(self, q, m):
        return self.data_model.rho + q - 2*m

    def get_train_loss(self, V, q, m):
        return (self.data_model.rho + q - 2*m) / (1+V)**2
