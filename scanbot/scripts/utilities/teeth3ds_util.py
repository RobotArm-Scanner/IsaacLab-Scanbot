"""Utilities for 3DS teeth assets."""

from __future__ import annotations

import json
import os
import shutil
from collections import defaultdict

from pxr import Gf, Usd, UsdGeom, UsdPhysics, UsdShade


def _parse_obj(path: str) -> tuple[list[Gf.Vec3f], list[list[int]]]:
    vertices: list[Gf.Vec3f] = []
    faces: list[list[int]] = []
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            if raw.startswith("v "):
                _, x, y, z, *_ = raw.split()
                vertices.append(Gf.Vec3f(float(x), float(y), float(z)))
            elif raw.startswith("f "):
                parts = raw.split()[1:]
                face: list[int] = []
                for part in parts:
                    idx = part.split("/", 1)[0]
                    face.append(int(idx) - 1)
                faces.append(face)
    return vertices, faces


def _read_labels(path: str) -> list[int]:
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    return payload["labels"]


def _build_faces_by_label(
    faces: list[list[int]],
    labels: list[int],
) -> dict[int, list[list[int]]]:
    faces_by_label: dict[int, list[list[int]]] = defaultdict(list)
    for face in faces:
        face_labels = [labels[idx] for idx in face]
        first_label = face_labels[0]
        if all(lbl == first_label for lbl in face_labels):
            label = first_label
        else:
            counts: dict[int, int] = {}
            for lbl in face_labels:
                counts[lbl] = counts.get(lbl, 0) + 1
            label = max(counts.items(), key=lambda item: (item[1], -item[0]))[0]
        faces_by_label[label].append(face)
    return faces_by_label


def _make_mesh(stage: Usd.Stage, prim_path: str, vertices, faces) -> UsdGeom.Mesh:
    mesh = UsdGeom.Mesh.Define(stage, prim_path)
    mesh.CreateSubdivisionSchemeAttr().Set("none")
    mesh.CreateDoubleSidedAttr().Set(True)

    index_map: dict[int, int] = {}
    points: list[Gf.Vec3f] = []
    face_vertex_counts: list[int] = []
    face_vertex_indices: list[int] = []

    for face in faces:
        face_vertex_counts.append(len(face))
        for idx in face:
            local_idx = index_map.get(idx)
            if local_idx is None:
                local_idx = len(points)
                index_map[idx] = local_idx
                points.append(vertices[idx])
            face_vertex_indices.append(local_idx)

    mesh.CreatePointsAttr(points)
    mesh.CreateFaceVertexCountsAttr(face_vertex_counts)
    mesh.CreateFaceVertexIndicesAttr(face_vertex_indices)
    UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
    return mesh


def build_segmented_usd(
    obj_path: str,
    json_path: str,
    looks_usd: str,
    out_usd_path: str,
    root_name: str,
    looks_root: str = "/Shaders",
) -> str:
    vertices, faces = _parse_obj(obj_path)
    labels = _read_labels(json_path)
    if len(vertices) != len(labels):
        raise ValueError("vertex/label mismatch")

    faces_by_label = _build_faces_by_label(faces, labels)

    stage = Usd.Stage.CreateNew(out_usd_path)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    root_path = f"/{root_name}"
    teeth_array_path = f"{root_path}/TeethArray"
    gum_path = f"{root_path}/Gum"
    shaders_path = f"{root_path}/Shaders"

    root = UsdGeom.Xform.Define(stage, root_path)
    stage.SetDefaultPrim(root.GetPrim())
    UsdGeom.Xform.Define(stage, teeth_array_path)
    stage.DefinePrim(shaders_path)

    shaders_prim = stage.GetPrimAtPath(shaders_path)
    shaders_prim.GetReferences().AddReference(looks_usd, looks_root)
    gum_mat = UsdShade.Material(stage.GetPrimAtPath(f"{shaders_path}/OmniPBR"))
    teeth_mat = UsdShade.Material(stage.GetPrimAtPath(f"{shaders_path}/OmniPBR_01"))

    for label in sorted(faces_by_label.keys()):
        faces_for_label = faces_by_label[label]
        if label == 0:
            mesh = _make_mesh(stage, gum_path, vertices, faces_for_label)
            mesh.GetPrim().SetMetadata("displayName", "Gum")
            UsdShade.MaterialBindingAPI(mesh.GetPrim()).Bind(gum_mat)
            continue
        prim_name = f"Tooth_{label}"
        prim_path = f"{teeth_array_path}/{prim_name}"
        mesh = _make_mesh(stage, prim_path, vertices, faces_for_label)
        mesh.GetPrim().SetMetadata("displayName", prim_name)
        UsdShade.MaterialBindingAPI(mesh.GetPrim()).Bind(teeth_mat)

    stage.GetRootLayer().Save()
    return out_usd_path


def _root_name(dataset_id: str) -> str:
    if dataset_id.endswith("_lower"):
        return f"Teeth_{dataset_id}"
    return f"Teeth_{dataset_id}_lower"


def ensure_t3ds_shaders_usd(resources_root: str) -> str:
    out_dir = os.path.join(resources_root, "teeth", "t3ds")
    os.makedirs(out_dir, exist_ok=True)
    out_usd = os.path.join(out_dir, "shaders.usd")
    if os.path.isfile(out_usd):
        return out_usd

    src_usd = os.path.join(resources_root, "teeth", "t2", "9000.usd")
    stage = Usd.Stage.CreateNew(out_usd)
    shaders = stage.DefinePrim("/Shaders")
    stage.SetDefaultPrim(shaders)
    shaders.GetReferences().AddReference(src_usd, "/World/Looks")
    stage.GetRootLayer().Save()
    return out_usd


def _update_shaders_reference(stage: Usd.Stage, root_name: str, shaders_usd: str) -> bool:
    shaders_path = f"/{root_name}/Shaders"
    shaders_prim = stage.GetPrimAtPath(shaders_path)
    if not shaders_prim:
        return False
    refs = shaders_prim.GetReferences()
    refs.ClearReferences()
    refs.AddReference(shaders_usd, "/Shaders")
    stage.GetRootLayer().Save()
    return True


def _has_segmentation_layout(stage: Usd.Stage, root_name: str) -> bool:
    teeth_array_path = f"/{root_name}/TeethArray"
    gum_path = f"/{root_name}/Gum"
    return stage.GetPrimAtPath(teeth_array_path).IsValid() and stage.GetPrimAtPath(gum_path).IsValid()


def ensure_t3ds_usd(resources_root: str, dataset_id: str = "A9TECAGP") -> str:
    out_dir = os.path.join(resources_root, "teeth", "t3ds")
    os.makedirs(out_dir, exist_ok=True)
    out_usd = os.path.join(out_dir, f"{dataset_id}_lower_segmented.usd")
    root_name = _root_name(dataset_id)
    shaders_usd = ensure_t3ds_shaders_usd(resources_root)
    if os.path.isfile(out_usd):
        stage = Usd.Stage.Open(out_usd)
        if stage:
            default_prim = stage.GetDefaultPrim()
            if default_prim and default_prim.GetName() == root_name:
                if _has_segmentation_layout(stage, root_name):
                    if _update_shaders_reference(stage, root_name, shaders_usd):
                        return out_usd

    src_usd = os.path.join(
        resources_root,
        "teeth3ds",
        dataset_id,
        f"{dataset_id}_lower_segmented.usd",
    )
    if os.path.isfile(src_usd):
        src_stage = Usd.Stage.Open(src_usd)
        if src_stage:
            src_default = src_stage.GetDefaultPrim()
            if src_default and src_default.GetName() == root_name and _has_segmentation_layout(src_stage, root_name):
                shutil.copyfile(src_usd, out_usd)
                stage = Usd.Stage.Open(out_usd)
                if stage:
                    _update_shaders_reference(stage, root_name, shaders_usd)
                return out_usd

    obj_path = os.path.join(resources_root, "teeth3ds", dataset_id, f"{dataset_id}_lower.obj")
    json_path = os.path.join(resources_root, "teeth3ds", dataset_id, f"{dataset_id}_lower.json")
    return build_segmented_usd(
        obj_path,
        json_path,
        shaders_usd,
        out_usd,
        root_name=root_name,
    )
