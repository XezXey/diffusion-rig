import argparse
import os
import sys
import json
import tqdm

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch as th
from glob import glob

from utils.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)

from torchvision.utils import save_image
import torchvision
import numpy as np

from decalib.deca import DECA
from decalib.utils.config import cfg as deca_cfg
from decalib.datasets import datasets as deca_dataset
from utils.sh_utils import rotate_sh, interp_sh
from utils.difarelipp_utils import get_data_path

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import pickle
from utils.logging import createLogger
logger = createLogger()

def create_inter_data(dataset, modes, meanshape_path="", 
                      mani_light_dict = {"mani_light": "rotate_sh", "rotate_sh_axis": 2, "num_frames": 60, "use_self_light": True}):

    # Build DECA
    deca_cfg.model.use_tex = True
    deca_cfg.model.tex_path = "data/FLAME_texture.npz"
    deca_cfg.model.tex_type = "FLAME"
    deca_cfg.rasterizer_type = "pytorch3d"
    deca = DECA(config=deca_cfg)

    meanshape = None
    if os.path.exists(meanshape_path):
        # print("use meanshape: ", meanshape_path)
        with open(meanshape_path, "rb") as f:
            meanshape = pickle.load(f)
    else:
        pass
        # print("not use meanshape")

    img2 = dataset[-1]["image"].unsqueeze(0).to("cuda")
    with th.no_grad():
        code2 = deca.encode(img2)
    image2 = dataset[-1]["original_image"].unsqueeze(0).to("cuda")
    
    use_self_light = mani_light_dict["use_self_light"]
    mani_light = mani_light_dict["mani_light"]
    rotate_sh_axis = mani_light_dict["rotate_sh_axis"]
    num_frames = mani_light_dict["num_frames"]

    for i in range(len(dataset) - 1):

        img1 = dataset[i]["image"].unsqueeze(0).to("cuda")

        with th.no_grad():
            code1 = deca.encode(img1)

        # To align the face when the pose is changing
        ffhq_center = None
        ffhq_center = deca.decode(code1, return_ffhq_center=True)

        tform = dataset[i]["tform"].unsqueeze(0)
        tform = th.inverse(tform).transpose(1, 2).to("cuda")
        original_image = dataset[i]["original_image"].unsqueeze(0).to("cuda")

        code1["tform"] = tform
        if meanshape is not None:
            code1["shape"] = meanshape

        for mode in modes:

            code = {}
            for k in code1:
                code[k] = code1[k].clone()

            origin_rendered = None

            if mode == "light":
                source_light = code['light'].clone().detach().cpu().numpy()  # [1, 9, 3]
                if use_self_light:
                    target_light = code['light'].clone().detach().cpu().numpy()    # [1, 9, 3]
                else:
                    target_light = code2["light"].clone().detach().cpu().numpy()   # [1, 9, 3]
                if mani_light == "rotate_sh":
                    target_light = rotate_sh({'light': target_light.reshape(1, 27)}, 
                                            src_idx=0, n_step=num_frames, axis=rotate_sh_axis)['light']
                    target_light = target_light.reshape(num_frames, 9, 3)
                    target_light = th.tensor(target_light, dtype=th.float32).to("cuda")  # [num_frames, 9, 3]
                elif mani_light == "interp_sh":
                    target_light = interp_sh({
                        'source_light': source_light.reshape(1, 9, 3),
                        'target_light': target_light.reshape(1, 9, 3)}, 
                        n_step=num_frames)['light']
                    target_light = target_light.reshape(num_frames, 9, 3)
                else:
                    raise NotImplementedError(f"[#] Only 'rotate_sh' and 'interp_sh' modes are implemented, got {mani_light}.")
                
                target_light = th.tensor(target_light, dtype=th.float32).to("cuda")  # [num_frames, 9, 3]
                code["light"] = target_light  # [num_frames, 9, 3]
            else: 
                raise NotImplementedError(f"[#] Only 'light' mode is implemented, got {mode}.")

            # Match all code dim with the light
            # B = code['light'].shape[0]
            # for k in code.keys():
            #     if k != 'light':
            #         code[k] = code[k].repeat_interleave(dim=0, repeats=B)
            
            
            opdict, _ = deca.decode(
                code,
                render_orig=True,
                original_image=original_image,
                tform=code["tform"],
                align_ffhq=True,
                ffhq_center=ffhq_center,
            )
            
            # Match opdict of normal_images and albedo_images to the rendered
            B = opdict["rendered_images"].shape[0]
            opdict["normal_images"] = opdict["normal_images"].repeat_interleave(dim=0, repeats=B)
            opdict["albedo_images"] = opdict["albedo_images"].repeat_interleave(dim=0, repeats=B)

            origin_rendered = opdict["rendered_images"].detach()

            batch = {}
            batch["image"] = original_image * 2 - 1
            batch["image2"] = image2 * 2 - 1
            batch["rendered"] = opdict["rendered_images"].detach()
            batch["normal"] = opdict["normal_images"].detach()
            batch["albedo"] = opdict["albedo_images"].detach()
            
            batch["mode"] = mode
            batch["origin_rendered"] = origin_rendered
            yield batch


def main():
    
    def run(imagepath_list, vis_dir):
        dataset = deca_dataset.TestData(imagepath_list, iscrop=True, size=args.image_size)

        modes = args.modes.split(",")

        mani_light_dict = {
            "mani_light": args.mani_light,
            "rotate_sh_axis": args.rotate_sh_axis,
            "num_frames": args.num_frames,
            "use_self_light": args.use_self_light,
        }
        data = create_inter_data(dataset, modes, args.meanshape, mani_light_dict=mani_light_dict)

        sample_fn = (
            diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop
        )

        noise = th.randn(1, 3, args.image_size, args.image_size).to("cuda")

        for batch in data:
            image = batch["image"]
            image2 = batch["image2"]
            rendered, normal, albedo = batch["rendered"], batch["normal"], batch["albedo"]

            physic_cond = th.cat([rendered, normal, albedo], dim=1)

            image = image
            physic_cond = physic_cond
            
            
            sub_batch_size = 10
            num_sub_batches = (physic_cond.shape[0] + sub_batch_size - 1) // sub_batch_size
            
            with th.no_grad():
                if batch["mode"] == "latent":
                    detail_cond = model.encode_cond(image2)
                else:
                    detail_cond = model.encode_cond(image)

            all_output = []
            t = tqdm.tqdm(range((num_sub_batches)), desc="[#] Sampling sub batch", leave=False)
            for i in t:
                sub_batch_physic_cond = physic_cond[i * sub_batch_size:(i + 1) * sub_batch_size]
                sub_batch_detail_cond = detail_cond.repeat_interleave(dim=0, repeats=sub_batch_physic_cond.shape[0])
                
                sub_B = sub_batch_physic_cond.shape[0]
                
                # t.set_description(f"[#] Sampling sub-batch {i}/{num_sub_batches}.")

                sample = sample_fn(
                    model,
                    (sub_B, 3, args.image_size, args.image_size),
                    # (1, 3, args.image_size, args.image_size),
                    noise=noise.repeat_interleave(dim=0, repeats=sub_B),
                    clip_denoised=args.clip_denoised,
                    model_kwargs={"physic_cond": sub_batch_physic_cond, "detail_cond": sub_batch_detail_cond},
                )
                sample = (sample + 1) / 2.0
                sample = sample.contiguous()
                
                all_output.append(sample)

                # save_image(
                #     sample, os.path.join(vis_dir, "{}_".format(idx) + batch["mode"]) + ".png"
                # )
        all_output = th.cat(all_output, dim=0)
        
        # Save output frames
        # albedo = [0, 1]
        # normal = [-1, 1]
        # rendered = [~0, ~1]   # Need to be clipped
        # all_output = [0, 1]
        to_save = [rendered, all_output, albedo, normal]
        for idx, name in enumerate(['ren', 'res', 'albedo', 'normal']):
            # Save per frame
            to_save_img = to_save[idx]
            for i in range(to_save[idx].shape[0]):
                save_image(to_save_img[i:i+1], os.path.join(vis_dir, f"{name}_frame{i:03d}.png"), normalize=True, range=(0, 1))
            
            # Save as a video
            to_save_vid = to_save_img.permute(0, 2, 3, 1)  # [B, H, W, C]
            to_save_vid = th.clip(to_save_vid * 255, 0, 255).to(th.uint8).cpu().numpy()  # [B, H, W, C]
            torchvision.io.write_video(os.path.join(vis_dir, f"{name}.mp4"), to_save_vid, fps=24, video_codec='libx264', options={"crf": "18"})
            # Save a roundtrip version
            to_save_vid_roundtrip = np.concatenate((to_save_vid, np.flip(to_save_vid, axis=0)), axis=0)  # [2B, H, W, C]
            torchvision.io.write_video(os.path.join(vis_dir, f"{name}_rt.mp4"), to_save_vid_roundtrip, fps=24, video_codec='libx264', options={"crf": "18"})
        
        all_out = th.cat((
            all_output.permute(0, 2, 3, 1),
            rendered.permute(0, 2, 3, 1),
            albedo.permute(0, 2, 3, 1),
            normal.permute(0, 2, 3, 1)
        ), dim=2)
        all_out = th.clip(all_out * 255, 0, 255).cpu().numpy()
        torchvision.io.write_video(os.path.join(vis_dir, "out.mp4"), all_out, fps=24, video_codec='libx264', options={"crf": "18"})
        all_out_rt = np.concatenate((all_out, np.flip(all_out, axis=0)), axis=0)
        torchvision.io.write_video(os.path.join(vis_dir, "out_rt.mp4"), all_out_rt, fps=24, video_codec='libx264', options={"crf": "18"})

    args = create_argparser().parse_args()

    print("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )

    ckpt = th.load(args.model_path)

    model.load_state_dict(ckpt)
    model.to("cuda")
    model.eval()
    

    with open(args.sample_pair_json, 'r') as f:
        sample_pairs = json.load(f)['pair']
        sample_pairs_k = [k for k in sample_pairs.keys()]
        sample_pairs_v = [v for v in sample_pairs.values()]
        
        
    if len(args.idx) > 2:
        # Filter idx to be within 0 < idx < len(sample_pairs)
        to_run_idx = [i for i in args.idx if 0 <= i < len(sample_pairs)]
    elif args.idx == [-1]:
        s = 0
        e = len(sample_pairs)
        to_run_idx = list(range(s, e))
    elif len(args.idx) == 2:
        s, e = args.idx
        s = max(0, s)
        e = min(e, len(sample_pairs))
        to_run_idx = list(range(s, e))
    else:
        raise ValueError("Invalid index range provided. Please provide a valid range or -1 for all indices.")

    data_path = get_data_path(args.dataset, args.set_)
    logger.info("#" * 80)
    logger.warning("DiffusionRig's light manipulation...")
    logger.info(f"[#] Manipulated light: {args.mani_light}")
    logger.info(f"[#] Rotate SH axis (affect if mani_light = rotate_sh): {args.rotate_sh_axis}")
    logger.info(f"[#] Use self light: {args.use_self_light}")
    logger.info(f"[#] Use mean shape: {args.meanshape}")
    logger.warning("DiffusionRig's running on...")
    logger.info(f"[#] Save path: {args.save_path}")
    logger.info(f"[#] Running idx: {to_run_idx}")
    logger.info(f"[#] Sample json: {args.sample_pair_json}")
    logger.warning("DiffsionRig's dataset...")
    logger.info(f"[#] Datapath: {data_path}")
    logger.info(f"[#] Dataset: {args.dataset}")
    logger.info(f"[#] Set: {args.set_}")
    logger.info(f"[#] Image ext: {args.img_ext}")
    logger.info("#" * 80)
   
    
    to_run_idx = tqdm.tqdm(to_run_idx, desc="Processing indices", total=len(to_run_idx), unit="index")
    for idx in to_run_idx:
        pair = sample_pairs_v[idx]
        pair_id = sample_pairs_k[idx]
        # fn = f'{pair_id}_src={pair["src"]}_dst={pair["dst"]}'
        
        to_run_idx.set_description(f"[#] Processing index {idx} with src image {pair['src']} and dst image {pair['dst']}...")
        
        source_image = pair["src"].replace('.jpg', args.img_ext)
        target_image = pair["dst"].replace('.jpg', args.img_ext)
        
        imagepath_list = [f'{data_path}/{source_image}', f'{data_path}/{target_image}']
        if not os.path.exists(imagepath_list[0]) and os.path.exists(imagepath_list[1]):
            logger.warning(f"[!] Source image {imagepath_list[0]} does not exist, skipping index {idx}.")
            continue

        vis_dir = f'{args.save_path}/src={pair["src"]}_dst={pair["dst"]}/n_step={args.num_frames}/'
        os.makedirs(vis_dir, exist_ok=True)
        run(imagepath_list, vis_dir)
    



def create_argparser():
    defaults = dict(
        clip_denoised=True,
        num_samples=10000,
        use_ddim=True,
        model_path="",
        modes="light",
        meanshape="",
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    # Dataset path
    parser.add_argument("--dataset", type=str, default="ffhq", help="path to the dataset")
    parser.add_argument("--set_", type=str, default="valid", help="dataset split to use")
    parser.add_argument("--img_ext", type=str, default=".jpg", help="image extension to use")
    # DiFaReli++'s running cfg for comparison
    parser.add_argument("--sample_pair_json", type=str, required=True, help="sample pair json file for DiFaReli++ comparison")
    parser.add_argument("--idx", nargs='+', type=int, default=[-1], help="index of the source spherical harmonics coefficients to rotate")
    parser.add_argument("--save_path", type=str, required=True, help="result save path")
    # Light manipulation
    parser.add_argument("--num_frames", type=int, default=60, help="number of frames for light manipulation")
    parser.add_argument('--use_self_light', action='store_true', default=False, help='Use self light for light mode')
    parser.add_argument("--mani_light", type=str, required=True, help="manipulated light path for DiFaReli++ comparison")
    parser.add_argument("--rotate_sh_axis", type=int, default=2, help="axis to rotate spherical harmonics coefficients, 0 for x, 1 for y, 2 for z")
    
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
