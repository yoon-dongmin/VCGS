import torch
import torch.nn as nn
from torch.nn import init
from torch.optim import lr_scheduler
import torch.nn.functional as F

from models import losses
from pointnet2_ops import pointnet2_modules as pointnet2


def get_scheduler(optimizer, opt):
    # 학습률 스케줄러를 설정하는 함수
    if opt.lr_policy == 'lambda':
        # lambda 정책을 사용하여 학습률을 점진적으로 감소시키는 스케줄러 설정
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + 1 + 1 - opt.niter) / float(opt.niter_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        # step 정책을 사용하여 일정한 에포크마다 학습률을 감소시키는 스케줄러 설정
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        # plateau 정책을 사용하여 성능 향상이 없을 때 학습률을 감소시키는 스케줄러 설정
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    else:
        # 지원되지 않는 정책일 경우 에러 발생
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler


def init_weights(net, init_type, init_gain):
    # 네트워크 가중치를 초기화하는 함수
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            # 초기화 타입에 따라 가중치를 초기화
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
        elif classname.find('batchnorm.BatchNorm') != -1:
            # 배치 정규화 계층의 가중치 초기화
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    net.apply(init_func)


def init_net(net, init_type, init_gain, gpu_ids):
    # 네트워크를 초기화하고 GPU에 할당하는 함수
    if len(gpu_ids) > 0:
        assert (torch.cuda.is_available())
        net.cuda(gpu_ids[0])
        net = net.cuda()
        net = torch.nn.DataParallel(net, gpu_ids)
    if init_type != 'none':
        # 가중치 초기화 함수 호출
        init_weights(net, init_type, init_gain)
    return net


def define_classifier(opt, gpu_ids, init_type, init_gain, device):
    # 분류기 네트워크를 정의하는 함수
    if opt.arch == "sampler":
        net = GraspSamplerVAE(opt.model_scale, opt.pointnet_radius, opt.pointnet_nclusters, opt.latent_size, device)
    elif opt.arch == "evaluator":
        net = GraspEvaluator(opt.model_scale, opt.pointnet_radius, opt.pointnet_nclusters, device)
    else:
        raise NotImplementedError("Loss not found")
    # 네트워크 초기화 함수 호출
    return init_net(net, init_type, init_gain, gpu_ids)


def define_loss(opt):
    # 손실 함수를 정의하는 함수
    if opt.arch == "sampler":
        # VAE의 KL divergence와 재구성 손실 정의
        kl_loss = losses.kl_divergence
        reconstruction_loss = losses.control_point_l1_loss
        return kl_loss, reconstruction_loss
    elif opt.arch == "evaluator":
        # 분류 손실 정의
        ce_loss = losses.classification_loss
        return ce_loss
    else:
        # 지원되지 않는 아키텍처일 경우 에러 발생
        raise NotImplementedError("Loss not found")



class GraspSampler(nn.Module):
    def __init__(self, latent_size, device):
        super(GraspSampler, self).__init__()
        self.latent_size = latent_size  # 잠재 공간의 크기
        self.device = device  # 사용할 장치 (CPU 또는 GPU)

    def create_decoder(self, model_scale, pointnet_radius, pointnet_nclusters, num_input_features):
        # 디코더의 입력 특징 수는 3 + k + u입니다
        # 3: 포인트 클라우드의 x, y, z 위치
        # k: 잠재 공간의 크기 (주로 2)
        # u: 제약 네트워크를 학습할 때 사용하는 추가 특징 (1)
        self.decoder = base_network(pointnet_radius, pointnet_nclusters, model_scale, num_input_features, self.device)
        # 쿼터니언 출력을 위한 선형 계층 (회전)
        self.q = nn.Linear(model_scale * 1024, 4)
        # 변환 출력을 위한 선형 계층 (이동)
        self.t = nn.Linear(model_scale * 1024, 3)

    def decode(self, xyz, z, features=None):
        # 포인트 클라우드와 잠재 벡터를 결합합니다.
        xyz_features = self.concatenate_z_with_pc(xyz, z)
        # 추가 특징이 주어지면 결합합니다.
        if features is not None:
            xyz_features = torch.cat((xyz_features, features), -1)
        else:
            # 쿼리 포인트 특징을 설정하여 결합합니다.
            query_point_encoding = self.setup_query_point_feature(xyz.shape[0], xyz.shape[1])
            xyz_features = torch.cat((xyz_features, query_point_encoding), -1)

        # 특징 텐서를 전치하고 연속적인 메모리 배열로 변환합니다.
        xyz_features = xyz_features.transpose(-1, 1).contiguous()
        # 디코더 네트워크를 통해 특징을 추출합니다.
        for i, module in enumerate(self.decoder[0]):
            xyz, xyz_features = module(xyz, xyz_features)
        # 마지막 MLP 계층을 통해 특징을 변환하여 출력합니다.
        x = self.decoder[1](xyz_features.squeeze(-1))
        # 쿼터니언과 변환 값을 결합하여 최종 예측을 반환합니다.
        predicted_qt = torch.cat((F.normalize(self.q(x), p=2, dim=-1), self.t(x)), -1) #4차원 quat와 3차원 translation 값 생성
        return predicted_qt

    def concatenate_z_with_pc(self, pc, z):
        # 잠재 벡터를 포인트 클라우드의 각 포인트에 결합합니다.
        z.unsqueeze_(1)
        z = z.expand(-1, pc.shape[1], -1)
        return torch.cat((pc, z), -1)

    def setup_query_point_feature(self, batch_size, num_points):
        # 쿼리 포인트 특징을 설정합니다 (초기값은 0, 마지막 포인트는 1)
        query_point_feature = torch.zeros((batch_size, num_points, 1)).to(self.device)
        query_point_feature[:, -1] = 1
        return query_point_feature



class GraspSamplerVAE(GraspSampler):
    """Generative VAE grasp-sampler를 학습하기 위한 네트워크"""

    def __init__(self, model_scale, pointnet_radius=0.02, pointnet_nclusters=128, latent_size=2, device="cpu"):
        super(GraspSamplerVAE, self).__init__(latent_size, device)
        self.create_encoder(model_scale, pointnet_radius, pointnet_nclusters, 19 + 1)
        self.create_decoder(model_scale, pointnet_radius, pointnet_nclusters, latent_size + 3 + 1)
        self.create_bottleneck(model_scale * 1024, latent_size)

    def create_encoder(self, model_scale, pointnet_radius, pointnet_nclusters, num_input_features):
        # 인코더의 입력 특징 수는 20입니다:
        # 포인트 클라우드의 x, y, z 위치, [3]
        # 펼쳐진 4x4=16 그립 포즈 행렬, [16]
        # 그립을 생성할 포인트를 나타내는 1/0 이진 인코딩 [1]
        self.encoder = base_network(pointnet_radius, pointnet_nclusters, model_scale, num_input_features, self.device)
        # base network PointNet++를 통해 : 1024 차원

    def create_bottleneck(self, input_size, latent_size):
        # 잠재 공간의 평균과 로그 분산을 계산하는 선형 계층을 생성합니다.
        mu = nn.Linear(input_size, latent_size) # model_scale * 1024, 2
        logvar = nn.Linear(input_size, latent_size) # model_scale * 1024, 2
        self.latent_space = nn.ModuleList([mu, logvar])

    def encode(self, pc_xyz, grasps, position_constraint_feature):
        """
        pc_xyz:                      torch.Size([B, 1024, 3])
        grasps:                      torch.Size([B, 16])
        position_constraint_feature: torch.Size([B, 1024, 1])
        """
        # 그립 포즈 행렬을 확장하여 포인트 클라우드와 동일한 크기로 맞춥니다
        grasp_features = grasps.unsqueeze(1).expand(-1, pc_xyz.shape[1], -1) # B,1024,16
        # 포인트 클라우드와 그립 포즈 특징을 결합합니다
        features = torch.cat((pc_xyz, grasp_features), -1) # B,1024,19
        # 위치 제약 특징을 추가로 결합합니다.
        features = torch.cat((features, position_constraint_feature), -1) # B,1024,20
        # 특징 텐서를 전치하고 연속적인 메모리 배열로 변환합니다
        features = features.transpose(-1, 1).contiguous()
        # 인코더 네트워크를 통해 특징을 추출합니다
        for i, module in enumerate(self.encoder[0]): # Three PointSAModules
            pc_xyz, features = module(pc_xyz, features)  
        # 최종 MLP 계층을 통해 특징을 변환하여 출력합니다.
        return self.encoder[1](features.squeeze(-1))

    def bottleneck(self, z):
        # 잠재 공간의 평균과 로그 분산을 반환합니다.
        return self.latent_space[0](z), self.latent_space[1](z)

    def reparameterize(self, mu, logvar):
        # 재매개변수화를 통해 잠재 벡터를 생성 => gradient decent 가능하게 함
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, pc, grasp=None, features=None, train=True):
        if train:
            return self.forward_train(pc, grasp, features)
        else:
            return self.forward_test(pc, grasp, features)

    def forward_train(self, pc, grasp, features):
        # 인코더를 통해 입력을 잠재 벡터로 변환합니다.
        z = self.encode(pc, grasp, features) # z: 1024
        mu, logvar = self.bottleneck(z)
        z = self.reparameterize(mu, logvar) # z 샘플링
        # 디코더를 통해 grasp 생성
        qt = self.decode(pc, z, features)
        return qt, mu, logvar

    def forward_test(self, pc, grasp, features):
        # 인코더를 통해 입력을 잠재 벡터로 변환합니다.
        z = self.encode(pc, grasp, features)
        mu, _ = self.bottleneck(z)
        # 디코더를 통해 그립을 생성합니다.
        qt = self.decode(pc, mu, features)
        return qt

    def sample_latent(self, batch_size):
        # 잠재 공간에서 샘플링하여 잠재 벡터를 생성합니다.
        return torch.randn(batch_size, self.latent_size).to(self.device)

    def generate_grasps(self, pc, z=None, features=None):
        if z is None:
            # 잠재 공간에서 샘플링하여 잠재 벡터를 생성합니다.
            z = self.sample_latent(pc.shape[0])
        # 디코더를 통해 그립을 생성합니다.
        qt = self.decode(pc, z, features)
        return qt, z.squeeze()


class GraspEvaluator(nn.Module):
    def __init__(self, model_scale=1, pointnet_radius=0.02, pointnet_nclusters=128, device="cpu"):
        super(GraspEvaluator, self).__init__()
        self.device = device
        # 평가자를 생성하는 메서드를 호출하여 네트워크를 초기화합니다.
        self.create_evaluator(pointnet_radius, model_scale, pointnet_nclusters)

    def create_evaluator(self, pointnet_radius, model_scale, pointnet_nclusters):
        # 평가자의 입력 특성 수는 4입니다: x, y, z 위치와 물체와 그리퍼를 구분하기 위한 이진 특성
        self.evaluator = base_network(pointnet_radius, pointnet_nclusters, model_scale, 4, self.device)
        # 1024 * model_scale 차원의 출력을 1차원으로 변환하는 완전 연결 계층
        self.predictions_logits = nn.Linear(1024 * model_scale, 1)

    def evaluate(self, xyz, xyz_features):
        # 세 개의 PointSAModules를 순차적으로 적용하여 특징을 추출합니다.
        for i, module in enumerate(self.evaluator[0]): 
            xyz, xyz_features = module(xyz, xyz_features)
        # 최종적으로 MLP 계층을 통해 특징을 변환하여 출력합니다.
        return self.evaluator[1](xyz_features.squeeze(-1))

    def forward(self, pc, gripper_pc):
        # 물체와 그리퍼 포인트 클라우드를 병합하고 특징을 생성합니다
        pc, pc_features = self.merge_pc_and_gripper_pc(pc, gripper_pc)
        pc = pc.to(self.device)
        pc_features = pc_features.to(self.device)
        # 평가를 수행하여 logits를 얻습니다
        x = self.evaluate(pc, pc_features.contiguous()).to(self.device)
        # logits를 반환합니다
        return self.predictions_logits(x)

    def merge_pc_and_gripper_pc(self, pc, gripper_pc):
        """
        물체 포인트 클라우드와 그리퍼 포인트 클라우드를 병합하고 각 포인트가 물체 또는 그리퍼에 속하는지
        나타내는 이진 보조 특성을 추가합니다.
        """
        pc_shape = pc.shape
        gripper_shape = gripper_pc.shape
        # 두 포인트 클라우드의 배치 크기가 동일한지 확인합니다.
        assert (len(pc_shape) == 3)
        assert (len(gripper_shape) == 3)
        assert (pc_shape[0] == gripper_shape[0])
        batch_size = pc_shape[0]
        # 물체와 그리퍼 포인트 클라우드를 병합합니다.
        # 왜 10_xyz? 확인 필요
        l0_xyz = torch.cat((pc, gripper_pc), 1).to(self.device)
        # 물체와 그리퍼를 구분하기 위한 이진 레이블을 생성합니다.
        labels = [torch.ones(pc.shape[1], 1, dtype=torch.float32), torch.zeros(gripper_pc.shape[1], 1, dtype=torch.float32)]
        labels = torch.cat(labels, 0) # 두 텐서를 첫 번째 차원(포인트 차원)에서 결합
        labels.unsqueeze_(0) # batch 차원 추가
        # 배치 크기에 맞게 레이블을 반복합니다.
        labels = labels.repeat(batch_size, 1, 1)
        # 포인트 클라우드와 레이블을 결합하여 최종 특징을 생성합니다.
        l0_points = torch.cat([l0_xyz, labels.to(self.device)], -1).transpose(-1, 1) #(batch_size, 4, N + M)
        return l0_xyz, l0_points


def base_network(pointnet_radius, pointnet_nclusters, scale, in_features, device="cpu"):
    """
    PointNet2 Backbone network를 정의하는 함수
    """
    # PointNet2 SAModule의 링크: https://github.com/erikwijmans/Pointnet2_PyTorch/blob/b5ceb6d9ca0467ea34beb81023f96ee82228f626/pointnet2_ops_lib/pointnet2_ops/pointnet2_modules.py#L118
    
    # 첫 번째 SAModule을 정의합니다. npoint는 포인트 클라우드의 수, radius는 반경, nsample은 샘플링할 포인트의 수, mlp는 다층 퍼셉트론(MLP) 구조입니다.
    sa1_module = pointnet2.PointnetSAModule(
        npoint=pointnet_nclusters, 
        radius=pointnet_radius, 
        nsample=64,
        mlp=[in_features, 64 * scale, 64 * scale, 128 * scale]
    ).to(device)
    
    # 두 번째 SAModule을 정의합니다. npoint는 포인트 클라우드의 수, radius는 반경, nsample은 샘플링할 포인트의 수, mlp는 다층 퍼셉트론(MLP) 구조입니다.
    sa2_module = pointnet2.PointnetSAModule(
        npoint=32, 
        radius=0.04, 
        nsample=128,
        mlp=[128 * scale, 128 * scale, 128 * scale, 256 * scale]
    ).to(device)
    
    # 세 번째 SAModule을 정의합니다. 이 모듈은 최종적으로 512차원으로 변환합니다.
    sa3_module = pointnet2.PointnetSAModule(
        mlp=[256 * scale, 256 * scale, 256 * scale, 512 * scale]
    ).to(device)
    
    # 세 개의 SAModule을 하나의 모듈 리스트로 묶습니다.
    sa_modules = nn.ModuleList([sa1_module, sa2_module, sa3_module])
    
    # 완전 연결 계층을 정의합니다. BatchNorm1d와 ReLU 활성화 함수를 포함합니다.
    fc_layer = nn.Sequential(
        nn.Linear(512 * scale, 1024 * scale),
        nn.BatchNorm1d(1024 * scale), 
        nn.ReLU(True),
        nn.Linear(1024 * scale, 1024 * scale),
        nn.BatchNorm1d(1024 * scale), 
        nn.ReLU(True)
    )

    # 두 개의 모듈 리스트(sa_modules)와 완전 연결 계층(fc_layer)을 포함하는 nn.ModuleList를 반환합니다.
    return nn.ModuleList([sa_modules, fc_layer])

