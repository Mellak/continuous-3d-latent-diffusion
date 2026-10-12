from .ddpm import objective as _objective
def objective(model,x,schedule):return _objective(model,x,schedule,'eps')
