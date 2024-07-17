import torch
from . import networks
from os.path import join
import utils.utils as utils


class GraspNetModel:
    """ Class for training Model weights
    :args opt: structure containing configuration params
    e.g., --dataset_mode -> sampling / evaluation)
    """

    def __init__(self, opt):
        self.opt = opt
        self.gpu_ids = opt.gpu_ids
        self.is_train = opt.is_train
        if self.gpu_ids and self.gpu_ids[0] >= torch.cuda.device_count():
            self.gpu_ids[0] = torch.cuda.device_count() - 1
        self.device = torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')
        self.save_dir = join(opt.checkpoints_dir, opt.name)
        self.optimizer = None
        self.loss = None
        self.pcs = None
        self.grasps = None
        self.features = None

        # load/define networks
        # GraspSamplerVAE class
        self.net = networks.define_classifier(opt, self.gpu_ids, opt.init_type, opt.init_gain, self.device)

        self.criterion = networks.define_loss(opt)
        # (kl_loss, reconstruction_loss) -> sampler
        # classification_loss -> evaluator

        # TODO different loss initialization
        if self.opt.arch == "sampler":
            self.kl_loss = None
            self.reconstruction_loss = None
        elif self.opt.arch == "evaluator":
            self.classification_loss = None
        else:
            raise NameError("There is no architecture name. Check __init__ in GraspNetModel")

        if self.is_train:
            self.optimizer = torch.optim.Adam(self.net.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.scheduler = networks.get_scheduler(self.optimizer, opt)
        if not self.is_train or opt.continue_train:
            self.load_network(opt.which_epoch, self.is_train)

    def set_input(self, data):
        input_pcs = torch.from_numpy(data['pc']).contiguous()
        input_grasps = torch.from_numpy(data['grasp_rt']).float()

        #TODO different targets according the network(sampler:target_cps, evaluator:labels)
        if self.opt.arch == "sampler":
            features = torch.from_numpy(data['features']).float()
            self.features = features.to(self.device).requires_grad_(self.is_train)
            targets = torch.from_numpy(data['target_cps']).float()
        elif self.opt.arch == "evaluator":
            #TODO labels are not known yet.
            targets = torch.from_numpy(data['labels']).float()
        else:
            raise NameError("There is no architecture name. Check set_input in GraspNetModel")

        self.pcs = input_pcs.to(self.device).requires_grad_(self.is_train)
        self.grasps = input_grasps.to(self.device).requires_grad_(self.is_train)
        self.targets = targets.to(self.device)

    def generate_grasps(self, pcs, z=None):
        with torch.no_grad():
            return self.net.module.generate_grasps(pcs, z=z)

    def evaluate_grasps(self, pcs, gripper_pcs):
        # TODO adjust the output with GraspEvaluator network
        success = self.net.module(pcs, gripper_pcs)
        return torch.sigmoid(success)

    def forward(self):
        if self.opt.arch == "sampler":
            return self.net(self.pcs, self.grasps, self.features, train=self.is_train)
        elif self.opt.arch == "evaluator":
            # forward
            return self.net(self.pcs, self.grasps)#, train=self.is_train)

    def backward(self, out):
        # TODO different backward with different loss
        if self.opt.arch == 'sampler':
            predicted_cp, mu, logvar = out # There is no confidence
            predicted_cp = utils.transform_control_points(predicted_cp, predicted_cp.shape[0], device=self.device)
            # 아래 코드 사용하는지 확인
            self.predicted_cp_dens = utils.transform_control_points(out[0], out[0].shape[0], device=self.device, dense=True)
            self.reconstruction_loss = self.criterion[1](predicted_cp, self.targets, device=self.device)
            self.kl_loss = self.opt.kl_loss_weight * self.criterion[0](mu, logvar, device=self.device)
            self.loss = self.kl_loss + self.reconstruction_loss
        elif self.opt.arch == 'evaluator':
            # There is no confidence
            grasp_classification = out
            self.classification_loss = self.criterion(grasp_classification.squeeze(), self.targets,device=self.device)
            self.loss = self.classification_loss
        else:
            raise NameError("There is no architecture name. Please check backward in GraspNetModel")
        self.loss.backward()


    def validate(self):
        with torch.no_grad():
            if self.opt.arch == "sampler":
                out = self.generate_grasps(self.pcs) # working only decoder of sampler
                self.predicted_grasp_qt, z = out
                self.predicted_cp = utils.transform_control_points(self.predicted_grasp_qt, self.predicted_grasp_qt.shape[0], device=self.device)
                self.predicted_cp_dens = utils.transform_control_points(self.predicted_grasp_qt, self.predicted_grasp_qt.shape[0], device=self.device, dense=True)
                self.z = z
                reconstruction_loss = self.criterion[1](self.predicted_cp, self.targets, device=self.device)
                return reconstruction_loss
            elif self.opt.arch == "evaluator":
                out = self.evaluate_grasps(self.pcs, self.grasps) # grasp_classification / sigmoid output
                # predicted = torch.round(torch.sigmoid(out)).squeeze()
                # correct = (predicted == self.targets).sum().item()
                classification_loss = self.criterion(out.squeeze(), self.targets, device=self.device)
                return classification_loss # correct, len(self.targets)
            else:
                raise NameError("There is no architecture name. Please check validate in GraspNetModel")


    def optimize_parameters(self):
        self.optimizer.zero_grad()
        out = self.forward()
        self.backward(out)
        self.optimizer.step()

    def load_network(self, which_epoch, train=True):
        """load model from disk"""
        print("Loading network")
        save_filename = '%s_net.pth' % which_epoch
        load_path = join(self.save_dir, save_filename)
        net = self.net
        if isinstance(net, torch.nn.DataParallel):
            net = net.module
        print('loading the model from %s' % load_path)
        checkpoint = torch.load(load_path, map_location=self.device)
        if hasattr(checkpoint['model_state_dict'], '_metadata'):
            del checkpoint['model_state_dict']._metadata
        if 'confidence.weight' in checkpoint['model_state_dict'].keys():
            del checkpoint['model_state_dict']['confidence.weight']
        if 'confidence.bias' in checkpoint['model_state_dict'].keys():
            del checkpoint['model_state_dict']['confidence.bias']
        net.load_state_dict(checkpoint['model_state_dict'])
        # train = False
        if train:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            self.opt.epoch_count = checkpoint["epoch"]
        else:
            net.eval()

    def save_network(self, net_name, epoch_num):
        """save model to disk"""
        save_filename = '%s_net.pth' % (net_name)
        save_path = join(self.save_dir, save_filename)
        torch.save({'epoch': epoch_num + 1,
                    'model_state_dict': self.net.module.cpu().state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict(),
                    }, save_path)
        if len(self.gpu_ids) > 0 and torch.cuda.is_available():
            self.net.cuda(self.gpu_ids[0])

    def update_learning_rate(self):
        """update learning rate (called once every epoch)"""
        self.scheduler.step()
        lr = self.optimizer.param_groups[0]['lr']
        print('learning rate = %.7f' % lr)

    def test(self):
        """tests model
        returns: number correct and total number(1)
        """
        with torch.no_grad():
            out = self.forward()
            prediction = out
            if self.opt.arch == "sampler":
                self.predicted_cp = utils.transform_control_points(prediction, prediction.shape[0], device=self.device)
                self.predicted_cp_dens = utils.transform_control_points(prediction, prediction.shape[0], device=self.device, dense=True)
                reconstruction_loss = self.criterion[1](self.predicted_cp, self.targets, device=self.device)
                return reconstruction_loss, 1
            elif self.opt.arch == "evaluator":
                predicted = torch.round(torch.sigmoid(prediction)).squeeze()
                correct = (predicted == self.targets).sum().item()
                return correct, len(self.targets)
            else:
                raise NameError("There is no architecture name. Please check test in GraspNetModel")


    def get_random_grasp_and_point_cloud(self):
        num_point_clouds = self.pcs.shape[0]
        random_point_cloud_index = torch.randint(high=num_point_clouds, size=(1,))
        object_point_cloud = self.pcs[random_point_cloud_index]
        inlier_grasps = utils.get_inlier_grasp_indices_with_control_points(self.predicted_cp_dens[random_point_cloud_index], device=self.device)
        grasp_point_cloud = inlier_grasps.flatten(0, 1).unsqueeze(0)
        grasp_colors = self.get_color_tensor([255, 0, 0], grasp_point_cloud.shape[1]).unsqueeze(0)
        point_cloud = torch.cat((object_point_cloud, grasp_point_cloud), 1)

        # blue object
        object_colors = self.get_color_tensor([0, 0, 255], object_point_cloud.shape[1]).unsqueeze(0)

        indices = self.features[random_point_cloud_index] == 1
        # green grasp
        object_colors[0, indices[0, :, 0], :] = torch.tensor([0, 255, 0])

        point_cloud_color = torch.cat((object_colors, grasp_colors), 1)
        return point_cloud, point_cloud_color

    def get_color_tensor(self, color, num_rows):
        color_tensor = torch.tensor(color)
        return color_tensor.repeat(num_rows, 1)

