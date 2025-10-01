from cross_validation import get_train_validation_test_participants
from Data_Generation.data_loader import load_cached_data
from Data_Generation.data_augmentation import ConsecutiveSegmentPairs, SingleSegmentEmbeddings
from torch.utils.data import DataLoader

from Model.recurrent_encoder import RecurrentEncoder
from Model.projector import EEGProjector
from Model.lightning_modules import ContrastiveLearningModel

import lightning as L
from lightning.pytorch.utilities import rank_zero_only
from lightning.pytorch.callbacks import ModelCheckpoint

@rank_zero_only
def log(msg): 
    print(msg)

def train(cfg):

    train_participants, validation_participants, _ = get_train_validation_test_participants(cfg['participants'], cfg['number_of_test_participants'], cfg['k'], cfg['fold'], cfg['seed'])


    log(f"Train participants: {train_participants}")
    log (f"Validation participants: {validation_participants}")


    data, artifact_detection_matrx = load_cached_data(cfg['sampling_rate'])


    train_dataset = ConsecutiveSegmentPairs(data,artifact_detection_matrx, cfg['sampling_rate'], train_participants, supervised=cfg['hparams']['supervised'], normalize_with_sigmoid=cfg['hparams']['normalize_with_sigmoid'])
    train_loader = DataLoader(train_dataset, batch_size=cfg['hparams']['batch_size'], shuffle=True, num_workers=8, pin_memory=True, persistent_workers=True, prefetch_factor=4, drop_last=True)


    encoder_raw = RecurrentEncoder(first_hidden_dim=cfg['hparams']['first_hidden_dim'], gru_residual_dim=cfg['hparams']['gru_residual_dim'], final_dim=cfg['hparams']['final_dim'])
    projector_raw = EEGProjector(input_dim=cfg['hparams']['final_dim'], lstm_dims=tuple(cfg['hparams']['lstm_dims']), dense_layer=cfg['hparams']['dense_layer'], output_dim=cfg['hparams']['output_dim'])
    encoder_dft = RecurrentEncoder(first_hidden_dim=cfg['hparams']['first_hidden_dim'], gru_residual_dim=cfg['hparams']['gru_residual_dim'], final_dim=cfg['hparams']['final_dim'])
    projector_dft = EEGProjector(input_dim=cfg['hparams']['final_dim'], lstm_dims=tuple(cfg['hparams']['lstm_dims']), dense_layer=cfg['hparams']['dense_layer'], output_dim=cfg['hparams']['output_dim'])


    model = ContrastiveLearningModel(encoder_raw=encoder_raw, projector_raw=projector_raw, encoder_dft=encoder_dft, projector_dft = projector_dft, lr=cfg['hparams']['learning_rate'], weight_decay=cfg['hparams']['weight_decay'], temperature=cfg['hparams']['temperature'], test_sizes=cfg['test_splits'])
    
    checkpoint_cb = ModelCheckpoint(dirpath="results", filename=f"fold{cfg['fold']}-{{epoch}}", save_last=True, save_top_k=-1, every_n_epochs=1)

    trainer = L.Trainer(accelerator="gpu", num_nodes=cfg['num_nodes'], devices=cfg['devices'], max_epochs=cfg['hparams']['epochs'], strategy="ddp", callbacks=[checkpoint_cb])


    trainer.fit(model=model, train_dataloaders=train_loader)
    results = {}

    for validation_participant in validation_participants:
        #trainer = L.Trainer(accelerator="gpu", devices=1)
        log(f"Validation participant: {validation_participant}")
        validation_dataset = SingleSegmentEmbeddings(data, artifact_detection_matrx, cfg['sampling_rate'], [validation_participant], normalize_with_sigmoid=cfg['hparams']['normalize_with_sigmoid'])
        validation_loader = DataLoader(validation_dataset, batch_size=cfg['hparams']['batch_size'], shuffle=False, num_workers=8, pin_memory=True, persistent_workers=True, prefetch_factor=4, drop_last=False)

        trainer.test(model=model, dataloaders=validation_loader)
        results["participant_"+str(validation_participant)] = model.test_results

    return results


