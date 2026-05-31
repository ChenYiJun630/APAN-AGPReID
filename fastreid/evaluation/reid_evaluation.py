# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""
import copy
import logging
import time
import itertools
from collections import OrderedDict

import numpy as np
import torch
import torch.nn.functional as F
from sklearn import metrics

from fastreid.utils import comm
from fastreid.utils.compute_dist import build_dist
from .evaluator import DatasetEvaluator
from .query_expansion import aqe
from .rank_cylib import compile_helper

logger = logging.getLogger(__name__)


class ReidEvaluator(DatasetEvaluator):
    def __init__(self, cfg, num_query, output_dir=None):
        self.cfg = cfg
        self._num_query = num_query
        self._output_dir = output_dir

        self._cpu_device = torch.device('cpu')

        self._predictions = []
        self._compile_dependencies()

    def reset(self):#清除之前预测的结果，进入下一轮评估
        self._predictions = []

    def process(self, inputs, outputs):
        prediction = {
            'feats': outputs.to(self._cpu_device, torch.float32),
            'pids': inputs['targets'].to(self._cpu_device),#人物ID
            'camids': inputs['camids'].to(self._cpu_device)#相机ID

        }
        self._predictions.append(prediction)#将每个batch的预测结果添加到列表中

    def evaluate(self):
        if comm.get_world_size() > 1:#在多进程训练时，收集所有进程的预测结果，仅主进程执行后续评估
            comm.synchronize()
            predictions = comm.gather(self._predictions, dst=0)
            predictions = list(itertools.chain(*predictions))

            if not comm.is_main_process():
                return {}

        else:
            predictions = self._predictions

        #合并预测结果
        #作用：将所有 batch 的特征、标签、相机 ID 合并为一个整体张量
        features = []
        pids = []
        camids = []
        for prediction in predictions:
            features.append(prediction['feats'])
            pids.append(prediction['pids'])
            camids.append(prediction['camids'])

        features = torch.cat(features, dim=0)
        pids = torch.cat(pids, dim=0).numpy()
        camids = torch.cat(camids, dim=0).numpy()
        # query feature, person ids and camera ids

        #分割query和gallery特征
        #前 num_query 个为查询集，其余为图库集。
        query_features = features[:self._num_query]
        query_pids = pids[:self._num_query]
        query_camids = camids[:self._num_query]

        # gallery features, person ids and camera ids
        gallery_features = features[self._num_query:]
        gallery_pids = pids[self._num_query:]
        gallery_camids = camids[self._num_query:]

        self._results = OrderedDict()


        # 查询扩展
        #使用平均查询扩展（AQE）提升检索性能。
        if self.cfg.TEST.AQE.ENABLED:#
            logger.info("Test with AQE setting")
            qe_time = self.cfg.TEST.AQE.QE_TIME#1
            qe_k = self.cfg.TEST.AQE.QE_K#5
            alpha = self.cfg.TEST.AQE.ALPHA#3.0
            query_features, gallery_features = aqe(query_features, gallery_features, qe_time, qe_k, alpha)

        #计算距离
        dist = build_dist(query_features, gallery_features, self.cfg.TEST.METRIC)



        if self.cfg.TEST.RERANK.ENABLED:
            logger.info("Test with rerank setting")
            k1 = self.cfg.TEST.RERANK.K1#20#KNN 中的最近邻个数，用于构建图结构
            k2 = self.cfg.TEST.RERANK.K2#6#KNN 中的次近邻个数，用于传播相似性
            lambda_value = self.cfg.TEST.RERANK.LAMBDA#0.3#权重系数，控制初始距离和重排序距离的融合比例

            if self.cfg.TEST.METRIC == "cosine":
                query_features = F.normalize(query_features, dim=1)
                gallery_features = F.normalize(gallery_features, dim=1)


            rerank_dist = build_dist(query_features, gallery_features, metric="jaccard", k1=k1, k2=k2)
            #融合原始距离和重排序距离
            dist = rerank_dist * (1 - lambda_value) + dist * lambda_value

        #指标计算
        from .rank import evaluate_rank
        cmc, all_AP, all_INP = evaluate_rank(dist, query_pids, gallery_pids, query_camids, gallery_camids)

        mAP = np.mean(all_AP)
        mINP = np.mean(all_INP)
        for r in [1, 5, 10]:
            self._results['Rank-{}'.format(r)] = cmc[r - 1] * 100
        self._results['mAP'] = mAP * 100
        self._results['mINP'] = mINP * 100
        self._results["metric"] = (mAP + cmc[0]) / 2 * 100

        #可选的ROC曲线，如果启用 ROC 评估，计算 TPR@FPR 指标。
        if self.cfg.TEST.ROC.ENABLED:
            from .roc import evaluate_roc
            scores, labels = evaluate_roc(dist, query_pids, gallery_pids, query_camids, gallery_camids)
            fprs, tprs, thres = metrics.roc_curve(labels, scores)

            for fpr in [1e-4, 1e-3, 1e-2]:
                ind = np.argmin(np.abs(fprs - fpr))
                self._results["TPR@FPR={:.0e}".format(fpr)] = tprs[ind]

        return copy.deepcopy(self._results)

    def _compile_dependencies(self):
        # Since we only evaluate results in rank(0), so we just need to compile
        # cython evaluation tool on rank(0)
        if comm.is_main_process():
            try:
                from .rank_cylib.rank_cy import evaluate_cy
            except ImportError:
                start_time = time.time()
                logger.info("> compiling reid evaluation cython tool")

                compile_helper()

                logger.info(
                    ">>> done with reid evaluation cython tool. Compilation time: {:.3f} "
                    "seconds".format(time.time() - start_time))
        comm.synchronize()
