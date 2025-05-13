# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from .detr_vae import build as build_vae
from .detr_vae import build_cnnmlp as build_cnnmlp
from .detr_vae import build_tactile as build_tactile
from .detr_vae import build_normals as build_normals

def build_ACT_model(args):
    return build_vae(args)

def build_TactileACT_model(args):
    return build_tactile(args)

def build_ACTNormals_model(args):
    return build_normals(args)

def build_CNNMLP_model(args):
    return build_cnnmlp(args)