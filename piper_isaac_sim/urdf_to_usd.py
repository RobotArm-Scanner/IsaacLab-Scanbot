
import argparse
import os
from typing import Tuple, Type, Any

from isaaclab.app import AppLauncher


def _get_converter_classes() -> Tuple[Type[Any], Type[Any]]:
    """Lazy import to ensure Kit extensions are available after AppLauncher starts."""
    try:
        from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg
    except ImportError:
        # fallback: omni.isaac.asset.importer.urdf 또는 다른 모듈 이름
        from omni.isaac.asset.importer.urdf import ImportConfig as UrdfConverterCfg
        from omni.isaac.asset.importer.urdf import UrdfConverter
    return UrdfConverter, UrdfConverterCfg

def convert_urdf_to_usd(
    urdf_path: str,
    usd_output_dir: str,
    usd_filename: str,
    *,
    merge_fixed_joints: bool = False,
    fixed_base: bool = True,
    root_link_name: str = None
) -> str:
    """
    URDF → USD 변환, 옵션 제어.
    반환값: 생성된 USD 파일 경로
    """
    # ensure output directory exists
    os.makedirs(usd_output_dir, exist_ok=True)
    usd_path = os.path.join(usd_output_dir, usd_filename)

    UrdfConverter, UrdfConverterCfg = _get_converter_classes()

    # 설정 객체 생성
    cfg = UrdfConverterCfg(
        asset_path=urdf_path,
        usd_dir=usd_output_dir,
        usd_file_name=usd_filename
    )

    # 옵션 설정
    cfg.merge_fixed_joints = merge_fixed_joints
    cfg.fix_base = fixed_base
    if root_link_name is not None:
        cfg.root_link_name = root_link_name

    # 만약 cfg에 일부 메서드 기반 설정이 있다면 호출
    # 예: cfg.set_merge_fixed_joints(False) 등
    try:
        cfg.set_merge_fixed_joints(merge_fixed_joints)
    except AttributeError:
        pass

    try:
        cfg.set_fix_base(fixed_base)
    except AttributeError:
        pass

    try:
        if root_link_name is not None:
            cfg.set_root_link_name(root_link_name)
    except AttributeError:
        pass

    # 변환 실행
    converter = UrdfConverter(cfg)
    generated_usd = converter.usd_path
    print("[convert] generated USD:", generated_usd)
    return generated_usd

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert URDF to USD using Isaac Sim URDF importer.")
    parser.add_argument(
        "urdf_path",
        nargs="?",
        default="/workspace/isaaclab/piper_isaac_sim/usd/piper_no_gripper_description_transmission.urdf",
        help="Path to the URDF file.",
    )
    parser.add_argument(
        "--usd-output-dir",
        default="/workspace/isaaclab/piper_isaac_sim/",
        help="Directory to store the generated USD file.",
    )
    parser.add_argument(
        "--usd-filename",
        default="piper_no_gripper_description_transmission.usd",
        help="Filename for the generated USD.",
    )
    parser.add_argument(
        "--merge-fixed-joints",
        action="store_true",
        help="Merge fixed joints during conversion.",
    )
    parser.add_argument(
        "--floating-base",
        action="store_true",
        help="If set, do not fix the robot base.",
    )
    parser.add_argument(
        "--root-link",
        default="base_link",
        help="Root link name for the articulation.",
    )

    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()

    # Isaac Sim 내부 Python 을 초기화해야 Kit 명령을 사용할 수 있음
    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    try:
        convert_urdf_to_usd(
            urdf_path=args.urdf_path,
            usd_output_dir=args.usd_output_dir,
            usd_filename=args.usd_filename,
            merge_fixed_joints=args.merge_fixed_joints,
            fixed_base=not args.floating_base,
            root_link_name=args.root_link,
        )
    finally:
        simulation_app.close()
    
