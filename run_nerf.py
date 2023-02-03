import os
import time
import svox
import math
import logging
import argparse
import numpy as np
import cv2 as cv
import trimesh
import skimage
import torch
import torch.nn.functional as F
from icecream import ic
from tqdm import tqdm
from pyhocon import ConfigFactory
from models.fields import NeRF
from models.my_dataset import Dataset
from models.my_nerf import MyNeRF, CheatNeRF
from models.my_renderer import MyNerfRenderer
import open3d as o3d
import cv2
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

class Runner:
    def __init__(self, conf_path, mode='render', case='CASE_NAME', is_continue=False):
        self.device = torch.device('cpu')
        # self.device = torch.device('cuda:0')

        # Configuration
        self.conf_path = conf_path
        f = open(self.conf_path)
        conf_text = f.read()
        conf_text = conf_text.replace('CASE_NAME', case)
        f.close()

        self.conf = ConfigFactory.parse_string(conf_text)
        self.conf['dataset.data_dir'] = self.conf['dataset.data_dir'].replace('CASE_NAME', case)
        self.base_exp_dir = self.conf['general.base_exp_dir']
        os.makedirs(self.base_exp_dir, exist_ok=True)
        self.dataset = Dataset(self.conf['dataset'], self.device)
        self.iter_step = 0
        self.use_white_bkgd = self.conf.get_bool('train.use_white_bkgd')

        self.is_continue = is_continue
        self.mode = mode
        self.model_list = []
        self.writer = None

        # Networks
        self.coarse_nerf = NeRF(**self.conf['model.coarse_nerf']).to(self.device)
        self.fine_nerf = NeRF(**self.conf['model.fine_nerf']).to(self.device)
        self.my_nerf = MyNeRF()
        self.renderer = MyNerfRenderer(self.my_nerf,
                                     **self.conf['model.nerf_renderer'])
        self.load_checkpoint(r'D:\李昭阳\课程\数据结构\期末大作业\nerf_model.pth', absolute=True)


    def load_checkpoint(self, checkpoint_name, absolute=False):
        if absolute:
            checkpoint = torch.load(checkpoint_name, map_location=self.device)
        else:
            checkpoint = torch.load(os.path.join(self.base_exp_dir, 'checkpoints', checkpoint_name), map_location=self.device)
        self.coarse_nerf.load_state_dict(checkpoint['coarse_nerf'])
        self.fine_nerf.load_state_dict(checkpoint['fine_nerf'])
        logging.info('End')

    def use_nerf(self):
        self.my_nerf = CheatNeRF(self.fine_nerf)
        self.renderer = MyNerfRenderer(self.my_nerf,
                                     **self.conf['model.nerf_renderer'])

    def save(self, NeedRefine=False):
        RS = 128
        thre = -50
        tRS = 32
        pRS = 16

        pts_xyz = torch.zeros((RS, RS, RS, 3))
        for i in tqdm(range(RS)):
            for j in range(RS):
                pts_xyz[:, i, j, 0] = torch.linspace(-0.125, 0.125, RS)
                pts_xyz[i, :, j, 1] = torch.linspace(0.75, 1.0, RS)
                pts_xyz[i, j, :, 2] = torch.linspace(-0.125, 0.125, RS)
        pts_xyz = pts_xyz.reshape((RS*RS*RS, 3))
        batch_size = 1024
        sigma = torch.zeros((RS*RS*RS, 1))
        color = torch.zeros((RS*RS*RS, 3))
        for batch in tqdm(range(0, pts_xyz.shape[0], batch_size)):
            batch_pts_xyz = pts_xyz[batch:batch+batch_size]
            net_sigma, net_color = self.fine_nerf(batch_pts_xyz, torch.zeros_like(batch_pts_xyz))
            sigma[batch:batch+batch_size] = net_sigma
            color[batch:batch+batch_size] = net_color

        if NeedRefine:
            octree = self.build_tree(thre, pts_xyz, sigma, tRS, pRS)
            self.my_nerf.octree = octree
            self.my_nerf.octree.save("octree_" + str(tRS*pRS) + ".pth")
        else:
            self.my_nerf.save(pts_xyz, sigma, color)

    def build_tree(self, thre, m_pts_xyz, m_sigma, tRS, pRS):
        # tRS为初始八叉树分辨率， pRS为单个体素细化后分辨率
        octree = svox.N3Tree(data_dim=4, init_refine=int(math.log2(tRS)) - 1, center=[0, 0.875, 0], radius=0.125)

        # 将体内的点进一步划分
        octree[m_pts_xyz[torch.where(m_sigma[:, 0] > thre)], 0].refine(int(math.log2(pRS)))

        # 对每个叶节点进行取样，即把细分后的节点取出
        pts_xyz = octree.sample(1)
        pts_xyz = pts_xyz.reshape(pts_xyz.shape[0], 3)

        batch_size = 1024
        sigma = torch.zeros((pts_xyz.shape[0], 1))
        color = torch.zeros((pts_xyz.shape[0], 3))
        for batch in tqdm(range(0, pts_xyz.shape[0], batch_size)):
            batch_pts_xyz = pts_xyz[batch:batch + batch_size]
            net_sigma, net_color = self.fine_nerf(batch_pts_xyz, torch.zeros_like(batch_pts_xyz))
            sigma[batch:batch + batch_size] = net_sigma
            color[batch:batch + batch_size] = net_color
        # 将同一节点的颜色和密度拼起来，便于查找
        octree[pts_xyz] = torch.cat((color, sigma), dim=1)

        # octree[m_pts_xyz] = m_sigma
        #
        # divide_times = 1
        # octree[m_pts_xyz[torch.where(m_sigma[:, 0] > thre)], 0].refine(1)
        #
        # while divide_times <= int(math.log2(pRS)):
        #     divide_times = divide_times + 1
        #
        #     pts_xyz = octree.sample(1)
        #     pts_xyz = pts_xyz.reshape(pts_xyz.shape[0], 3)
        #
        #     batch_size = 1024
        #     sigma = torch.zeros((pts_xyz.shape[0], 1))
        #     color = torch.zeros((pts_xyz.shape[0], 3))
        #     for batch in tqdm(range(0, pts_xyz.shape[0], batch_size)):
        #         batch_pts_xyz = pts_xyz[batch:batch + batch_size]
        #         net_sigma, net_color = self.fine_nerf(batch_pts_xyz, torch.zeros_like(batch_pts_xyz))
        #         sigma[batch:batch + batch_size] = net_sigma
        #         color[batch:batch + batch_size] = net_color
        #
        #     octree[pts_xyz[torch.where(sigma[:, 0] > thre)], 0].refine(1)
        #
        # octree[pts_xyz] = torch.cat((color, sigma), dim=1)

        return octree


    def render_video(self, IsPro=False):
        images = []
        resolution_level = 1
        n_frames = 90
        RS = 128
        tRS = 32
        pRS = 16
        for idx in tqdm(range(n_frames)):
            rays_o, rays_d = self.dataset.gen_rays_at(idx, resolution_level=resolution_level)
            H, W, _ = rays_o.shape
            rays_o = rays_o.reshape(-1, 3).split(1024)
            rays_d = rays_d.reshape(-1, 3).split(1024)

            out_rgb_fine = []

            for rays_o_batch, rays_d_batch in tqdm(zip(rays_o, rays_d)):
                near, far = self.dataset.near_far_from_sphere(rays_o_batch, rays_d_batch)
                background_rgb = torch.ones([1, 3], device=self.device) if self.use_white_bkgd else None

                render_out = self.renderer.render(rays_o_batch,
                                                rays_d_batch,
                                                near,
                                                far,
                                                background_rgb=background_rgb)

                def feasible(key): return (key in render_out) and (render_out[key] is not None)

                if feasible('fine_color'):
                    out_rgb_fine.append(render_out['fine_color'].detach().cpu().numpy())

                del render_out
            
            img_fine = (np.concatenate(out_rgb_fine, axis=0).reshape([H, W, 3]) * 256).clip(0, 255)
            img_fine = cv.resize(cv.flip(img_fine, 0), (512, 512))
            images.append(img_fine)
            if IsPro:
                os.makedirs(os.path.join(self.base_exp_dir, 'render_pro' + str(tRS*pRS)), exist_ok=True)
                cv.imwrite(os.path.join(self.base_exp_dir,  'render_pro' + str(tRS*pRS), '{}.jpg'.format(idx)), img_fine)
            else:
                os.makedirs(os.path.join(self.base_exp_dir, 'render' + str(RS)), exist_ok=True)
                cv.imwrite(os.path.join(self.base_exp_dir,  'render' + str(RS), '{}.jpg'.format(idx)), img_fine)

        # fourcc = cv.VideoWriter_fourcc(*'mp4v')
        # h, w, _ = images[0].shape
        # writer = cv.VideoWriter(os.path.join(self.base_exp_dir,  'render', 'show.mp4'),
        #                         fourcc, 30, (w, h))
        # for image in tqdm(images):
        #     writer.write(image)
        # writer.release()

    def graphbuild(self):
        RS = 128
        thre = -50
        tRS = 32
        pRS = 16

        if self.my_nerf.octree is not None:
            tree_tall = torch.where(self.my_nerf.octree.depths == self.my_nerf.octree.max_depth)
            size = 2**(int(self.my_nerf.octree[tree_tall[0, 0]].depths)+1)
            the_obj = torch.zeros((size, size, size))
            the_obj[:, :, :] = -3000

            pts_xyz = self.my_nerf.octree[tree_tall].corners
            X_index = ((pts_xyz[:, 0] + 0.125) * 4 * size).clamp(0, size - 1).long()
            Y_index = ((pts_xyz[:, 1] - 0.75) * 4 * size).clamp(0, size - 1).long()
            Z_index = ((pts_xyz[:, 2] + 0.125) * 4 * size).clamp(0, size - 1).long()

            the_obj[X_index, Y_index, Z_index] = self.my_nerf.octree[pts_xyz][:, 3]
            sigma = the_obj.detach().numpy() * (-1)

            verts, faces, _, _ = skimage.measure.marching_cubes(sigma, thre)
            mesh = trimesh.Trimesh(verts, faces)
            mesh.export("object_pro_" + str(tRS*pRS) + ".obj")
        else:
            object = torch.load("data_" + str(RS) + ".pth")
            sigma = object["volume_sigma"].reshape(RS, RS, RS)
            sigma = sigma.detach().numpy() * (-1)

            verts, faces, _, _ = skimage.measure.marching_cubes(sigma, thre)
            mesh = trimesh.Trimesh(verts, faces)
            mesh.export("object_" + str(RS) + ".obj")

    def show_image_mesh(self, thre):
        RS = 128
        object = torch.load("data_" + str(RS) + ".pth")
        sigma = object["volume_sigma"].reshape(RS, RS, RS)
        sigma = sigma.detach().numpy() * (-1)

        sigma_f = sigma[:, :, sigma.shape[2] // 2:sigma.shape[2]]
        sigma_b = sigma[:, :, 0:sigma.shape[2] // 2]

        verts_f, faces_f, _, _ = skimage.measure.marching_cubes(sigma_f, thre)
        verts_b, faces_b, _, _ = skimage.measure.marching_cubes(sigma_b, thre)
        verts_b[:, 2] -= (sigma_b.shape[0] // 2) - 1.2
        faces_b[:] += faces_f[faces_f.shape[0] - 1, 0]

        verts = np.concatenate((verts_f, verts_b))
        faces = np.concatenate((faces_f, faces_b))

        index_f = faces_f.reshape(faces_f.shape[0] * 3).shape[0]
        index_b = faces_b.reshape(faces_b.shape[0] * 3).shape[0]
        ind = faces.reshape(faces.shape[0] * 3)

        uv_c = np.zeros((ind.shape[0], 2))
        uv_c[:, :] = verts[ind][:, 0:2]

        uv_c[0:index_f, 0] = index_f - uv_c[0:index_f, 0]
        uv_c[:, 1] = sigma.shape[0] - uv_c[:, 1]
        uv_c = uv_c / sigma.shape[0]

        uv_c[0:index_f, 0] -= 0.31
        uv_c[0:index_f, 0] /= 0.9
        uv_c[0:index_f, 1] -= 0.25
        uv_c[0:index_f, 1] /= 0.8

        uv_c[index_f:uv_c.shape[0], 0] -= 0.8
        uv_c[index_f:uv_c.shape[0], 1] -= 0.25

        mesh = o3d.geometry.TriangleMesh(o3d.open3d.utility.Vector3dVector(verts),
                                         o3d.open3d.utility.Vector3iVector(faces))

        tex_img = cv2.imread(r"./try2.png")
        tex_img = cv2.cvtColor(tex_img, cv2.COLOR_BGR2RGB)
        # tex_img = cv2.flip(tex_img, 0)  # 对uv坐标做了1-v的预处理，如果没有做，就不需要垂直反转图像

        mesh.triangle_uvs = o3d.open3d.utility.Vector2dVector(uv_c)
        mesh.triangle_material_ids = o3d.utility.IntVector([0] * len(faces))
        mesh.textures = [o3d.geometry.Image(tex_img)]
        mesh.compute_vertex_normals()
        o3d.io.write_triangle_mesh(r'image_mesh_try.obj', mesh)
        # #################################################################### 双面纹理效果很差，但模型效果好
        # verts, faces, _, _ = skimage.measure.marching_cubes(sigma, thre)
        # uv_c = np.zeros((faces.reshape(faces.shape[0] * 3).shape[0], 2))
        #
        # uv_c[:, :] = verts[faces.reshape(faces.shape[0] * 3)][:, 0:2]
        # uv_c[0:uv_c.shape[0] // 2, 0] = sigma.shape[0] - uv_c[0:uv_c.shape[0] // 2, 0]
        # uv_c[:, 1] = sigma.shape[0] - uv_c[:, 1]
        # uv_c = uv_c / sigma.shape[0]
        #
        # uv_c[0:uv_c.shape[0] // 2, 0] -= 0.3
        # uv_c[0:uv_c.shape[0] // 2, 0] /= 0.9
        # uv_c[0:uv_c.shape[0] // 2, 1] -= 0.25
        # uv_c[0:uv_c.shape[0] // 2, 1] /= 0.8
        #
        # uv_c[uv_c.shape[0] // 2:uv_c.shape[0], 0] -= 0.1
        # uv_c[uv_c.shape[0] // 2:uv_c.shape[0], 1] -= 0.25
        #
        # mesh = o3d.geometry.TriangleMesh(o3d.open3d.utility.Vector3dVector(verts),
        #                                  o3d.open3d.utility.Vector3iVector(faces))
        #
        # tex_img = cv2.imread(r"./try2.png")
        # tex_img = cv2.cvtColor(tex_img, cv2.COLOR_BGR2RGB)
        #
        # mesh.triangle_uvs = o3d.open3d.utility.Vector2dVector(uv_c)
        # mesh.triangle_material_ids = o3d.utility.IntVector([0] * len(faces))
        # mesh.textures = [o3d.geometry.Image(tex_img)]
        # mesh.compute_vertex_normals()
        # o3d.io.write_triangle_mesh(r'image_mesh_try.obj', mesh)

        # #################################################################### 生成正面纹理
        # front = sigma[:, :, sigma.shape[2] // 2:sigma.shape[2]]
        # verts_f, faces_f, _, _ = skimage.measure.marching_cubes(front, thre)
        #
        # mesh_f = o3d.geometry.TriangleMesh(o3d.open3d.utility.Vector3dVector(verts_f),
        #                                  o3d.open3d.utility.Vector3iVector(faces_f))
        #
        # tex_img_f = cv2.imread(r"./纹理mesh/test_f.png")
        # tex_img_f = cv2.cvtColor(tex_img_f, cv2.COLOR_BGR2RGB)
        #
        # uv_c_b = np.zeros((faces_b.reshape(faces_b.shape[0] * 3).shape[0], 2))
        # uv_c_b[:, :] = verts_b[faces_b.reshape(faces_b.shape[0] * 3), 0:2]
        # uv_c_b[:, 1] = sigma.shape[0] - uv_c_b[:, 1]
        # uv_c_b = uv_c_b / sigma.shape[0]
        #
        # uv_c_f[0:uv_c_f.shape[0] // 2, 0] -= 0.3
        # uv_c_f[0:uv_c_f.shape[0] // 2, 0] /= 0.9
        # uv_c_f[0:uv_c_f.shape[0] // 2, 1] -= 0.25
        # uv_c_f[0:uv_c_f.shape[0] // 2, 1] /= 0.8
        #
        # mesh_f.triangle_uvs = o3d.open3d.utility.Vector2dVector(uv_c_f)
        # mesh_f.triangle_material_ids = o3d.utility.IntVector([0] * len(faces_f))
        # mesh_f.textures = [o3d.geometry.Image(tex_img_f)]
        # mesh_f.compute_vertex_normals()
        # o3d.io.write_triangle_mesh(r'image_mesh_f.obj', mesh_f)

        # #################################################################### 生成背面纹理
        # back = sigma[:, :, 0:sigma.shape[2] // 2]
        # verts_b, faces_b, _, _ = skimage.measure.marching_cubes(back, thre)
        # mesh_b = o3d.geometry.TriangleMesh(o3d.open3d.utility.Vector3dVector(verts_b),
        #                                  o3d.open3d.utility.Vector3iVector(faces_b))
        #
        # tex_img_b = cv2.imread(r"./纹理mesh/test.png")
        # tex_img_b = cv2.cvtColor(tex_img_b, cv2.COLOR_BGR2RGB)
        #
        # uv_c_b = np.zeros((faces_b.reshape(faces_b.shape[0] * 3).shape[0], 2))
        # uv_c_b[:, :] = verts_b[faces_b.reshape(faces_b.shape[0] * 3), 0:2]
        # uv_c_b[:, 1] = sigma.shape[0] - uv_c_b[:, 1]
        # uv_c_b = uv_c_b / sigma.shape[0]
        #
        # uv_c_b[uv_c_b.shape[0] // 2:uv_c_b.shape[0], 0] -= 0.1
        # uv_c_b[uv_c_b.shape[0] // 2:uv_c_b.shape[0], 1] -= 0.25
        #
        # uv_c_f = np.zeros((faces_f.reshape(faces_f.shape[0] * 3).shape[0], 2))
        # uv_c_f[:, :] = verts_f[faces_f.reshape(faces_f.shape[0] * 3), 0:2]
        # uv_c_f[:, 1] = sigma.shape[0] - uv_c_f[:, 1]
        # uv_c_f = uv_c_f / sigma.shape[0]
        #
        # mesh_b.triangle_uvs = o3d.open3d.utility.Vector2dVector(uv_c_b)
        # mesh_b.triangle_material_ids = o3d.utility.IntVector([0] * len(faces_b))
        # mesh_b.textures = [o3d.geometry.Image(tex_img_b)]
        # mesh_b.compute_vertex_normals()
        # o3d.io.write_triangle_mesh(r'image_mesh_b.obj', mesh_b)

if __name__ == '__main__':
    torch.set_grad_enabled(False)
    FORMAT = "[%(filename)s:%(lineno)s - %(funcName)20s() ] %(message)s"
    logging.basicConfig(level=logging.DEBUG, format=FORMAT)

    parser = argparse.ArgumentParser()
    parser.add_argument('--conf', type=str, default='./confs/base.conf')
    parser.add_argument('--mode', type=str, default='render')
    parser.add_argument('--mcube_threshold', type=float, default=0.0)
    parser.add_argument('--is_continue', default=False, action="store_true")
    parser.add_argument('--case', type=str, default='')
    parser.add_argument('--dataroot', type=str, default='')

    args = parser.parse_args()
    runner = Runner(args.conf, args.mode, args.case, args.is_continue)

    if args.mode == 'render':
        runner.save(True)
        runner.render_video(True)
        runner.graphbuild()
        # runner.show_image_mesh(-50)
    elif args.mode == 'test':
        runner.use_nerf()
        runner.render_video()
