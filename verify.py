import sys 
import torch 

# return the model dick for a path
def load_model_dict(path):
    return torch.load(path, map_location='cpu', weights_only=False) 

# return true if the same, false if not 
def compare_model_state(state_a, state_b):
    assert state_a.keys() == state_b.keys()

    for key in state_a:
        a_val = state_a[key]
        b_val = state_b[key]

        if not torch.equal(a_val, b_val):
            return False 

    return True 

# for this file, paste in the two thins 
if __name__ == "__main__":
    path_a, path_b = sys.argv[1], sys.argv[2]
    model_a, model_b = load_model_dict(path_a), load_model_dict(path_b)
    match = compare_model_state(model_a["model"], model_b["model"])
    print("the models match :) yay!" if match else "oh ohhh no match :(")
  