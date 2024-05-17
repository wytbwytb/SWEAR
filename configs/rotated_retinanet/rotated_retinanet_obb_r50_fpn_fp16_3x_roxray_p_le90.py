_base_ = ['./rotated_retinanet_obb_r50_fpn_3x_roxray_p_le90.py']

fp16 = dict(loss_scale='dynamic')
