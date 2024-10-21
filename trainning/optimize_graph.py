"""Freezes and optimize the model. Use after training."""
import argparse
import torch
from model import SpeechRecognition
from collections import OrderedDict

def trace(model):
    model.eval()
    x = torch.rand(1, 81, 300)
    hidden = model._init_hidden(1)
    traced = torch.jit.trace(model, (x, hidden))
    return traced

def main(model_checkpoint, save_path):
    print("loading model from", model_checkpoint)
    checkpoint = torch.load(model_checkpoint, map_location=torch.device('cpu'))
    h_params = SpeechRecognition.hyper_parameters
    model = SpeechRecognition(**h_params)

    model_state_dict = checkpoint['state_dict']
    new_state_dict = OrderedDict()
    for k, v in model_state_dict.items():
        name = k.replace("model.", "") # remove `model.`
        new_state_dict[name] = v

    model.load_state_dict(new_state_dict)

    print("tracing model...")
    traced_model = trace(model)
    print("saving to", save_path)
    traced_model.save(save_path)
    print("Done!")

# Remplacez ces chemins par vos chemins réels
model_checkpoint = 'model/model.ckpt.ckpt'
save_path = 'model/optimized/optimized_model'

main(model_checkpoint, save_path)
