import lightning as L
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
import numpy as np
import torch

from losses import nt_xent_loss


class ContrastiveLearningModel(L.LightningModule):
    def __init__(self, encoder_raw, projector_raw, encoder_dft, projector_dft, lr=1e-3, weight_decay=1e-4, temperature=0.1, test_sizes=[0.2]):
        super().__init__()
        self.encoder_raw = encoder_raw
        self.projector_raw = projector_raw
        self.encoder_dft = encoder_dft
        self.projector_dft = projector_dft
        self.lr = lr
        self.weight_decay = weight_decay
        self.temperature = temperature
        self.test_sizes = test_sizes
        self.test_embeddings = []
        self.test_labels = []
        self.test_results = None

    def training_step(self, batch, batch_idx):
        _ = batch_idx

        x_i, x_j, f_i, f_j = batch 

        x_i = x_i.unsqueeze(1)
        x_j = x_j.unsqueeze(1)
        f_i = f_i.unsqueeze(1)
        f_j = f_j.unsqueeze(1)

        h_x_i = self.encoder_raw(x_i)
        h_x_j = self.encoder_raw(x_j)
        h_f_i = self.encoder_dft(f_i)
        h_f_j = self.encoder_dft(f_j)

        z_x_i = self.projector_raw(h_x_i)
        z_x_j = self.projector_raw(h_x_j)
        z_f_i = self.projector_dft(h_f_i)
        z_f_j = self.projector_dft(h_f_j)

        z_i = torch.cat([z_x_i, z_f_i], dim=1)
        z_j = torch.cat([z_x_j, z_f_j], dim=1)

        loss = nt_xent_loss(z_i, z_j,temperature=self.temperature)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)   
        return loss

    def on_test_start(self):
        print("executing on_test_start")
        self.test_embeddings = []
        self.test_labels = []
        self.test_results = None
    
    def test_step(self, batch, batch_idx):
        _ = batch_idx
        x_i, f_j, y_i = batch
        
        x_i = x_i.unsqueeze(1)
        f_j = f_j.unsqueeze(1)  

        h_x_i = self.encoder_raw(x_i)
        h_f_j = self.encoder_dft(f_j)

        z_x_i = self.projector_raw(h_x_i)
        z_f_j = self.projector_dft(h_f_j)

        z_i = torch.cat([z_x_i, z_f_j], dim=1)

        self.test_embeddings.append(z_i)
        self.test_labels.append(y_i)
    
    def on_test_end(self):
        X = torch.cat(self.test_embeddings, dim=0).cpu().numpy()
        y = torch.cat(self.test_labels, dim=0).cpu().numpy()

        print("Collected embeddings shape:", X.shape) 
        print("Collected labels shape:", y.shape)
        
        results = {}

        for test_size in self.test_sizes:
            print(f"Testing with test size: {test_size}")
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, stratify=y, random_state=28)

            clf = LogisticRegression(solver='saga', tol=1e-4, max_iter=5000, class_weight='balanced')
            clf.fit(X_train, y_train)

            y_pred = clf.predict(X_test)
            y_prob = clf.predict_proba(X_test)[:, 1]    

            print(confusion_matrix(y_test, y_pred))
            results[test_size] = {"accuracy": accuracy_score(y_test, y_pred),"confusion_matrix": confusion_matrix(y_test, y_pred).tolist(), "roc_auc": roc_auc_score(y_test, y_prob)}

        self.test_results = results 

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        return optimizer