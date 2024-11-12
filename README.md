# TA_Project

## Datasets

Per aggiungere un dataset ereditate da `UnlearningDataset`, caricate i dati in forma _(PIL_img, label)_ in `self.data`. Se il dataset che volete usare arriva già diviso in train e test, concatenate i due insiemi in `self.data` e settate `self.TRAIN` con gli indici delle immagini di train e `self.TEST` con quelli di test (come fatto per CIFAR10). Se invece il dataset non è già diviso, caricate tutto in `self.data` e dividete gli indici in `self.TRAIN` e `self.TEST` seconda la percentuale in `split` (come fatto per SVHN).
