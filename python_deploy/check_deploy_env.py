import importlib.util


REQUIRED = {
    "dill": "dill",
    "numpy": "numpy",
    "torch": "torch",
    "torchvision": "torchvision",
    "cv2": "opencv-python",
    "hydra": "hydra-core",
    "omegaconf": "omegaconf",
    "termcolor": "termcolor",
    "tqdm": "tqdm",
    "swanlab": "swanlab",
    "einops": "einops",
    "diffusers": "diffusers",
    "zarr": "zarr",
    "numcodecs": "numcodecs",
    "numba": "numba",
    "PIL": "Pillow",
    "imageio": "imageio",
    "pyrealsense2": "pyrealsense2",
    "rtde_control": "ur-rtde",
    "rtde_receive": "ur-rtde",
}

OPTIONAL = {
    "robotiq_gripper": "robotiq_gripper, only needed for Robotiq gripper configs",
    "Jetson.GPIO": "Jetson.GPIO, only needed on Jetson GPIO gripper configs",
}


def check_modules(modules):
    missing = []
    for module_name, package_name in modules.items():
        try:
            spec = importlib.util.find_spec(module_name)
        except ModuleNotFoundError:
            spec = None
        if spec is None:
            missing.append((module_name, package_name))
    return missing


def main():
    missing_required = check_modules(REQUIRED)
    missing_optional = check_modules(OPTIONAL)

    if missing_required:
        print("[missing required]")
        for module_name, package_name in missing_required:
            print(f"  module={module_name:16s} install={package_name}")
    else:
        print("[ok] required Python modules are importable")

    if missing_optional:
        print("[missing optional]")
        for module_name, package_name in missing_optional:
            print(f"  module={module_name:16s} note={package_name}")

    if missing_required:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
