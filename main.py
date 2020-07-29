from train import train
from model_testing import test
from webapp import WebApp

model = train()
loss = test(model)
print("Loss: ", loss)

webapp = WebApp()
webapp.run()