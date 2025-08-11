import argparse
import os
import sys

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

from decalib.deca import DECA
from decalib.utils.config import cfg as deca_cfg
from decalib.datasets import datasets as deca_dataset
from utils.sh_utils import rotateSH, interp_sh

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
        print("use meanshape: ", meanshape_path)
        with open(meanshape_path, "rb") as f:
            meanshape = pickle.load(f)
    else:
        print("not use meanshape")

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
                if use_self_light:
                    target_light = code['light']    # [1, 9, 3]
                else:
                    target_light = code2["light"]   # [1, 9, 3]
                if mani_light == "rotate_sh":
                    target_light = rotateSH({'light': target_light.reshape(1, 27)}, 
                                            src_idx=0, n_step=num_frames, axis=rotate_sh_axis)['light']
                    target_light = target_light.reshape(num_frames, 9, 3)
                elif mani_light == "interp_sh":
                    target_light = interp_sh({'light': target_light.reshape(1, 9, 3)}, 
                                             src_idx=0, n_step=num_frames)['light']
                    target_light = target_light.reshape(num_frames, 9, 3)
                else:
                    raise NotImplementedError(f"[#] Only 'rotate_sh' and 'interp_sh' modes are implemented, got {mani_light}.")
                
                code["light"] = target_light  # [num_frames, 9, 3]
            else: 
                raise NotImplementedError(f"[#] Only 'light' mode is implemented, got {mode}.")

            opdict, _ = deca.decode(
                code,
                render_orig=True,
                original_image=original_image,
                tform=code["tform"],
                align_ffhq=True,
                ffhq_center=ffhq_center,
            )

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
    args = create_argparser().parse_args()

    print("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )

    ckpt = th.load(args.model_path)

    model.load_state_dict(ckpt)
    model.to("cuda")
    model.eval()

    imagepath_list = []

    if not os.path.exists(args.source) or not os.path.exists(args.target):
        print("source file or target file doesn't exists.")
        return

    imagepath_list = []
    if os.path.isdir(args.source):
        imagepath_list += (
            glob(args.source + "/*.jpg")
            + glob(args.source + "/*.png")
            + glob(args.source + "/*.bmp")
        )
    else:
        imagepath_list += [args.source]
    imagepath_list += [args.target]
    dataset = deca_dataset.TestData(imagepath_list, iscrop=True, size=args.image_size)

    modes = args.modes.split(",")

    mani_light_dict = {
        "mani_light": args.mani_light,
        "rotate_sh_axis": args.rotate_sh_axis,
        "num_frames": args.num_frames,
        "use_self_light": args.use_self_light,
    }
    data = create_inter_data(dataset, modes, args.meanshape, args.use_self_light, args.mani_light)

    sample_fn = (
        diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop
    )

    os.system("mkdir -p " + args.output_dir)

    noise = th.randn(1, 3, args.image_size, args.image_size).to("cuda")

    vis_dir = args.output_dir

    idx = 0
    for batch in data:
        image = batch["image"]
        image2 = batch["image2"]
        rendered, normal, albedo = batch["rendered"], batch["normal"], batch["albedo"]

        physic_cond = th.cat([rendered, normal, albedo], dim=1)

        image = image
        physic_cond = physic_cond

        with th.no_grad():
            if batch["mode"] == "latent":
                detail_cond = model.encode_cond(image2)
            else:
                detail_cond = model.encode_cond(image)

        sample = sample_fn(
            model,
            (1, 3, args.image_size, args.image_size),
            noise=noise,
            clip_denoised=args.clip_denoised,
            model_kwargs={"physic_cond": physic_cond, "detail_cond": detail_cond},
        )
        sample = (sample + 1) / 2.0
        sample = sample.contiguous()

        save_image(
            sample, os.path.join(vis_dir, "{}_".format(idx) + batch["mode"]) + ".png"
        )
        idx += 1


def create_argparser():
    defaults = dict(
        clip_denoised=True,
        num_samples=10000,
        use_ddim=True,
        model_path="",
        source="",
        target="",
        output_dir="",
        modes="pose,exp,light",
        meanshape="",
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    # DiFaReli++'s running cfg for comparison
    parser.add_argument("--sample_pair_json", type=str, required=True, help="sample pair json file for DiFaReli++ comparison")
    parser.add_argument("--idx", nargs='+', type=int, default=[-1], help="index of the source spherical harmonics coefficients to rotate")
    parser.add_argument("--video_path", type=str, required=True, help="reference and shading") 
    parser.add_argument("--save_path", type=str, default="result.mp4", help="result save path")
    # Light manipulation
    parser.add_argument("--num_frames", type=int, default=60, help="number of frames for light manipulation")
    parser.add_argument('--use_self_light', action='store_true', default=False, help='Use self light for light mode')
    parser.add_argument("--mani_light", type=str, required=True, help="manipulated light path for DiFaReli++ comparison")
    parser.add_argument("--rotate_sh_axis", type=int, default=2, help="axis to rotate spherical harmonics coefficients, 0 for x, 1 for y, 2 for z")
    
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
