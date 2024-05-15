import torch
import torch.nn.functional as F

metrics = {
    "coefficient-variation": lambda c_e: torch.std(c_e) / torch.mean(c_e),
    "Lmax-over-Lmin": lambda c_e: (torch.max(c_e) + 1) / (torch.min(c_e) + 1),
    "Lmax-over-Lmean": lambda c_e: torch.max(c_e) / torch.mean(c_e),
}


'''
reset_balance_profile: This function initializes or resets a balance profile.
 It takes a dictionary balance_dict, the number of layers num_layers, 
 and a balance strategy balance_strategy. 
 For each key in the metrics dictionary, it sets the corresponding value 
 in balance_dict to a list of None values with length equal to num_layers. 
 
 If a balance strategy is provided, 
 it also adds a key to the balance_dict for the loss associated with that strategy,
 again initialized to a list of None values.

'''
def reset_balance_profile(balance_dict, num_layers, balance_strategy):
    for key in metrics:
        balance_dict[key] = [None for _ in range(num_layers)]
    if balance_strategy:
        balance_dict[f"{balance_strategy}_loss"] = [None for _ in range(num_layers)]


def update_balance_profile(
    balance_dict,
    gate_top_k_idx,
    _gate_score_top_k,
    gate_context,
    layer_idx,
    num_expert,
    balance_strategy,
):
    # Fill in this function to conduct balance related jobs
    pass
