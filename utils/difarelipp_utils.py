import os

def get_data_path(dataset, set_, main_path="/data/mint/DPM_Dataset"):
    if dataset == "ffhq":
        data_path = f"{main_path}/ffhq_256_with_anno/ffhq_256/"
    elif dataset == "ffhq_png":
        data_path = f"{main_path}/ffhq_256_with_anno/ffhq_256_no_aliasing_png/"
    elif dataset == "mp_test":
        data_path = f"{main_path}/MultiPIE/MultiPIE_testset/mp_aligned/"
    elif dataset == "mp_test2":
        data_path = f"{main_path}/MultiPIE/MultiPIE_testset2/mp_aligned/"
    elif dataset == "mp_valid":
        data_path = f"{main_path}/MultiPIE/MultiPIE_validset/mp_aligned/"
    elif dataset == "mp_valid2":
        data_path = f"{main_path}/MultiPIE/MultiPIE_validset2/mp_aligned/"
    else: 
        raise ValueError(f"[#] Unknown dataset: {dataset}.")

    if set_ == "train":
        data_path = os.path.join(data_path, "train/")
    elif set_ == "valid":
        data_path = os.path.join(data_path, "valid/")
    else:
        raise ValueError(f"[#] Unknown set: {set_}.")

    return data_path