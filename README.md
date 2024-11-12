# TA_Project

First of all, install the requirements with `pip install -r requirements.txt`.

## Datasets

Per aggiungere un dataset ereditate da `UnlearningDataset`, caricate i dati in forma _(PIL_img, label)_ in `self.data`. Se il dataset che volete usare arriva già diviso in train e test, concatenate i due insiemi in `self.data` e settate `self.TRAIN` con gli indici delle immagini di train e `self.TEST` con quelli di test (come fatto per CIFAR10). Se invece il dataset non è già diviso, caricate tutto in `self.data` e dividete gli indici in `self.TRAIN` e `self.TEST` seconda la percentuale in `split` (come fatto per SVHN).

## WandB

To use WandB create an account on [wandb.ai](https://wandb.ai), create a project called TrendsAndApps on the site and create a file called `.env` in the root of the repo and add the following line:

```
WANDB_SECRET="YOUR_WAND_API_KEY"
```

YOUR_WAND_API_KEY can be found in your account settings on wandb.ai under `User settings` > `API keys`.
