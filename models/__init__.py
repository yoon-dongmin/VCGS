from .grasp_net import GraspNetModel
def create_model(opt):
    model = GraspNetModel(opt)
    return model