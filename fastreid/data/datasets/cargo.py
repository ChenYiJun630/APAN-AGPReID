# encoding: utf-8

import os
import os.path as osp
import glob

from fastreid.data.datasets import DATASET_REGISTRY
from fastreid.data.datasets.bases import ImageDataset

import pdb

__all__ = ['CARGO', ]


@DATASET_REGISTRY.register()#装饰器将该类注册到数据集注册表中，便于系统发现和使用
class CARGO(ImageDataset):
    dataset_dir = "/workspace/Datasets/CARGO"
    dataset_name = 'cargo'

    def __init__(self, root='datasets', **kwargs):
        self.root = root
        self.data_dir = osp.join(root, self.dataset_dir)

      
        self.train_dir = osp.join(self.data_dir, 'train')
        self.query_dir = osp.join(self.data_dir, 'query')
        self.gallery_dir = osp.join(self.data_dir, 'gallery')

        ##调用 process_dir 方法处理三个数据子集，分别获取训练、查询和图库数据
        train = self.process_dir(self.train_dir, is_train=True)
        query = self.process_dir(self.query_dir, is_train=False)
        gallery = self.process_dir(self.gallery_dir, is_train=False)

        super().__init__(train, query, gallery, **kwargs)

    def process_dir(self, dir_path, is_train=True):
        #遍历13个摄像头目录(Cam1-Cam13)，收集所有jpg图像文件路径
        img_paths = []
        for cam_index in range(13):
            img_paths = img_paths + glob.glob(osp.join(dir_path, f'Cam{cam_index + 1}', '*.jpg'))

        data = []
        for img_path in img_paths:
            #从文件名中解析行人ID(pid)、摄像头ID(camid)和视角类型(viewid
            pid = int(img_path.split('/')[-1].split('_')[2])
            camid = int(img_path.split('/')[-1].split('_')[0][3:])
            viewid = 'Aerial' if camid <= 5 else 'Ground'
            camid -= 1  # index starts from 0

            if is_train:
                #训练模式下，为pid和camid添加数据集名称前缀
                pid = self.dataset_name + "_" + str(pid)
                camid = self.dataset_name + "_" + str(camid)
            data.append((img_path, pid, camid, viewid))
            #将图像路径、行人ID、摄像头ID和视角类型组成元组添加到数据列表
        return data


@DATASET_REGISTRY.register()
class CARGO_AA(ImageDataset):
    dataset_dir = "CARGO"
    dataset_name = 'cargo_aa'

    def __init__(self, root='datasets', **kwargs):
        # self.root = '/workspace/AGReID/Datasets/'
        # self.data_dir = osp.join(root, self.dataset_dir)

        self.data_dir = '/workspace/ReID/AGPReID/dataset/CARGO'

        self.train_dir = osp.join(self.data_dir, 'train')
        self.query_dir = osp.join(self.data_dir, 'query')
        self.gallery_dir = osp.join(self.data_dir, 'gallery')

        train = self.process_dir(self.train_dir, is_train=True)
        query = self.process_dir(self.query_dir, is_train=False)
        gallery = self.process_dir(self.gallery_dir, is_train=False)

        super().__init__(train, query, gallery, **kwargs)

    def process_dir(self, dir_path, is_train=True):
        img_paths = []
        for cam_index in range(13):
            img_paths = img_paths + glob.glob(osp.join(dir_path, f'Cam{cam_index + 1}', '*.jpg'))

        data = []
        for img_path in img_paths:
            pid = int(img_path.split('/')[-1].split('_')[2])#提取行人ID，即_后三个字段
            camid = int(img_path.split('/')[-1].split('_')[0][3:])
            viewid = 'Aerial' if camid <= 5 else 'Ground'         
            camid -= 1  # index starts from 0


            if viewid == 'Ground':#如果为groud视角的则跳过，不加入训练或测试
                continue
            if is_train:
                pid = self.dataset_name + "_" + str(pid)
                camid = self.dataset_name + "_" + str(camid)
            data.append((img_path, pid, camid, viewid))#返回四元组
        return data


@DATASET_REGISTRY.register()
class CARGO_GG(ImageDataset):
    dataset_dir = "CARGO"
    dataset_name = 'cargo_gg'

    def __init__(self, root='datasets', **kwargs):
        # self.root = '/workspace/AGReID/Datasets/'
        # self.data_dir = osp.join(root, self.dataset_dir)

        self.data_dir = '/workspace/ReID/AGPReID/dataset/CARGO'

        self.train_dir = osp.join(self.data_dir, 'train')
        self.query_dir = osp.join(self.data_dir, 'query')
        self.gallery_dir = osp.join(self.data_dir, 'gallery')

        train = self.process_dir(self.train_dir, is_train=True)
        query = self.process_dir(self.query_dir, is_train=False)
        gallery = self.process_dir(self.gallery_dir, is_train=False)

        super().__init__(train, query, gallery, **kwargs)

    def process_dir(self, dir_path, is_train=True):
        img_paths = []
        for cam_index in range(13):
            img_paths = img_paths + glob.glob(osp.join(dir_path, f'Cam{cam_index + 1}', '*.jpg'))

        data = []
        for img_path in img_paths:
            pid = int(img_path.split('/')[-1].split('_')[2])
            camid = int(img_path.split('/')[-1].split('_')[0][3:])
            viewid = 'Aerial' if camid <= 5 else 'Ground'
            if viewid == 'Aerial':#如果为空中视角的都跳过
                continue
            camid -= 1  # index starts from 0

            if is_train:
                pid = self.dataset_name + "_" + str(pid)
                camid = self.dataset_name + "_" + str(camid)
            data.append((img_path, pid, camid, viewid))
        return data


@DATASET_REGISTRY.register()
class CARGO_AG(ImageDataset):
    dataset_dir = "CARGO"
    dataset_name = 'cargo_ag'

    def __init__(self, root='datasets', **kwargs):
        # self.root = '/workspace/AGReID/Datasets/'
        # self.data_dir = osp.join(root, self.dataset_dir)

        self.data_dir = '/workspace/ReID/AGPReID/dataset/CARGO'

        self.train_dir = osp.join(self.data_dir, 'train')
        self.query_dir = osp.join(self.data_dir, 'query')
        self.gallery_dir = osp.join(self.data_dir, 'gallery')

        train = self.process_dir(self.train_dir, is_train=True)
        #查询集指定为空中视角的
        query = self.process_dir(self.query_dir, is_train=False,view='Aerial')#查询的是空中视角的
        gallery = self.process_dir(self.gallery_dir, is_train=False,view='Ground')#图库是地面视角的

        super().__init__(train, query, gallery, **kwargs)

    def process_dir(self, dir_path, is_train=True, view='all'):
        img_paths = []
        for cam_index in range(13):
            img_paths = img_paths + glob.glob(osp.join(dir_path, f'Cam{cam_index + 1}', '*.jpg'))

        data = []
        for img_path in img_paths:
            pid = int(img_path.split('/')[-1].split('_')[2])
            camid = int(img_path.split('/')[-1].split('_')[0][3:])
            viewid = 'Aerial' if camid <= 5 else 'Ground'
            camid = 1 if camid <= 5 else 2
            camid -= 1  # index starts from 0
            
            if view!='all' and viewid !=view:
                continue
            #当 view 不为 'all' 且图像视角与指定视角不匹配时，跳过该图像这使得可以根据需要过滤特定视角的图像

            if is_train:
                pid = self.dataset_name + "_" + str(pid)
                camid = self.dataset_name + "_" + str(camid)
            data.append((img_path, pid, camid, viewid))
        return data

@DATASET_REGISTRY.register()
class CARGO_GA(ImageDataset):
    dataset_dir = "CARGO"
    dataset_name = 'cargo_ag'

    def __init__(self, root='datasets', **kwargs):
        # self.root = '/workspace/AGReID/Datasets/'
        # self.data_dir = osp.join(root, self.dataset_dir)

        self.data_dir = '/workspace/ReID/AGPReID/dataset/CARGO'
        
        self.train_dir = osp.join(self.data_dir, 'train')
        self.query_dir = osp.join(self.data_dir, 'query')
        self.gallery_dir = osp.join(self.data_dir, 'gallery')

        train = self.process_dir(self.train_dir, is_train=True)
        query = self.process_dir(self.query_dir, is_train=False,view='Ground')
        gallery = self.process_dir(self.gallery_dir, is_train=False,view='Aerial')

        super().__init__(train, query, gallery, **kwargs)

    def process_dir(self, dir_path, is_train=True, view='all'):
        img_paths = []
        for cam_index in range(13):
            img_paths = img_paths + glob.glob(osp.join(dir_path, f'Cam{cam_index + 1}', '*.jpg'))

        data = []
        for img_path in img_paths:
            pid = int(img_path.split('/')[-1].split('_')[2])
            camid = int(img_path.split('/')[-1].split('_')[0][3:])
            viewid = 'Aerial' if camid <= 5 else 'Ground'
            camid = 1 if camid <= 5 else 2
            camid -= 1  # index starts from 0
            
            if view!='all' and viewid !=view:
                continue

            if is_train:
                pid = self.dataset_name + "_" + str(pid)
                camid = self.dataset_name + "_" + str(camid)
            data.append((img_path, pid, camid, viewid))
        return data