import os
import time
import random
try:
    from tensorboardX import SummaryWriter
except ImportError as error:
    print('tensorboard X not installed, visualizing wont be available')
    SummaryWriter = None


class Writer:
    def __init__(self, opt):
        self.name = opt.name
        self.opt = opt
        self.save_dir = os.path.join(opt.checkpoints_dir, opt.name)
        self.log_name = os.path.join(self.save_dir, 'loss_log.txt')
        self.testacc_log = os.path.join(self.save_dir, 'testacc_log.txt')
        self.start_logs()
        self.nexamples = 0
        self.ncorrect = 0
        self.accuracy = 0

        if opt.is_train and not opt.no_vis and SummaryWriter is not None:
            self.display = SummaryWriter(logdir=os.path.join(self.opt.checkpoints_dir, self.opt.name) + "/tensorboard")  # comment=opt.name)
        else:
            self.display = None

    def start_logs(self):
        """ creates test / train log files """
        if self.opt.is_train:
            with open(self.log_name, "a") as log_file:
                now = time.strftime("%c")
                log_file.write(
                    '================ Training Loss (%s) ================\n' %
                    now)
        else:
            with open(self.testacc_log, "a") as log_file:
                now = time.strftime("%c")
                log_file.write(
                    '================ Testing Acc (%s) ================\n' %
                    now)

    def print_current_losses(self, epoch, i, losses, t, t_data, loss_types="total_loss"):
        """ prints train loss to terminal / file """
        if type(losses) == list:
            message = '(epoch: %d, iters: %d, time: %.3f, data: %.3f)' % (epoch, i, t, t_data)
            for (loss_type, loss_value) in zip(loss_types, losses):
                message += ' %s: %.3f' % (loss_type, loss_value.item())
        else:
            message = '(epoch: %d, iters: %d, time: %.3f, data: %.3f) loss: %.3f ' % (epoch, i, t, t_data, losses.item())

        print(message)
        with open(self.log_name, "a") as log_file:
            log_file.write('%s\n' % message)

    def plot_loss(self, losses, epoch, i, n, loss_types):
        iters = i + (epoch - 1) * n
        if self.display:
            if type(losses) == list:
                for (loss_type, loss_value) in zip(loss_types, losses):
                    self.display.add_scalar('data/train_loss/' + loss_type,  loss_value, iters)
            else:
                self.display.add_scalar('data/train_loss', losses, iters)

    def plot_valid_loss(self, loss_mean, epoch, arch):
        if self.display:
            if arch == "sampler":
                self.display.add_scalar('data/valid_loss/reconstruction_loss',  loss_mean, epoch)
            else: # evaluator
                self.display.add_scalar('data/valid_loss/classification_loss',  loss_mean, epoch)

    def plot_model_wts(self, model, epoch):
        if self.opt.is_train and self.display:
            for name, param in model.net.named_parameters():
                self.display.add_histogram(name,
                                           param.clone().cpu().data.numpy(),
                                           epoch)

    def print_acc(self, epoch):
        """ prints test accuracy to terminal / file """
        message = 'epoch: {}, TEST REC LOSS: [{:.5}]\n'.format(epoch, self.accuracy)
        print(message)
        with open(self.testacc_log, "a") as log_file:
            log_file.write('%s\n' % message)

    def plot_acc(self, epoch):
        if self.display:
            self.display.add_scalar('data/test_loss/grasp_reconstruction', self.accuracy, epoch)

    def plot_grasps(self, point_clouds, epoch, train_or_test="test"):
        if self.display:
            if len(point_clouds) > 0:
                try:
                    random_point_cloud = point_clouds[random.randint(0, len(point_clouds)-1)]
                    self.display.add_mesh(train_or_test+'/grasps', vertices=random_point_cloud[0], colors=random_point_cloud[1], global_step=epoch)
                except IndexError:
                    pass

    def update_counter(self, ncorrect, nexamples):
        self.nexamples += nexamples
        self.ncorrect += ncorrect

    def calculate_accuracy(self):
        self.accuracy = float(self.ncorrect) / self.nexamples

    def close(self):
        if self.display is not None:
            self.display.close()