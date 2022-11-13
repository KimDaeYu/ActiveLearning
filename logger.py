import wandb

def logger_init(pj_name:str,tag:str=None):
    project_name = pj_name
    wandb.init(project=pj_name, entity="inha_mai")
    # wandb_run_name = 
    # wandb.run.name = wandb_run_name
    wandb.run.save()