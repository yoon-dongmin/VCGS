from options.test_options import TestOptions
from models import create_model
from data import VCGS_DataLoader
from utils.writer import Writer

def run_valid(opt, model, epoch, writer, dataset):
    print('Running Validation')
    losses = []
    point_clouds = []
    for data in dataset:
        model.set_input(data)
        if opt.arch == "sampler":
            reconstruction_loss = model.validate()
            point_clouds.append(model.get_random_grasp_and_point_cloud())
            losses.append(reconstruction_loss)
        elif opt.arch == "evaluator":
            classification_loss = model.validate()
            losses.append(classification_loss)

    if opt.arch == "sampler":
        writer.plot_grasps(point_clouds, epoch)
    return sum(losses)/len(losses)

def run_test(epoch=-1, name="", writer=None, dataset_test=None):
    print('Running Test')
    # opt = TestOptions().parse()

    if opt is None:
        return

    # opt.serial_batches = True  # no shuffle
    # opt.name = name
    #
    # data_loader = DataLoader(opt)
    # _, test_dataset, _ = data_loader.split_dataset(opt.dataset_split_ratio)
    # dataset_test = data_loader.create_dataloader(test_dataset, shuffle_batches=False)
    # dataset_test_size = len(test_dataset)
    #
    # print('#test images = %d' % dataset_test_size)

    model = create_model(opt)

    # test
    point_clouds = []
    for data in dataset_test:
        model.set_input(data)
        ncorrect, nexamples = model.test()
        point_clouds.append(model.get_random_grasp_and_point_cloud())
        writer.update_counter(ncorrect, nexamples)

    writer.calculate_accuracy()
    writer.print_acc(epoch)
    writer.plot_acc(epoch)
    writer.plot_grasps(point_clouds, epoch)


if __name__ == '__main__':
    # TODO create dataset to put in dataset_test argument
    opt = TestOptions().parse()
    opt.serial_batches = True  # no shuffle
    opt.name = input() # sampler_lr_0002_bs_64_scale_1_npoints_128_radius_02_latent_size_2
    data_loader = VCGS_DataLoader(opt)
    _, test_dataset, _ = data_loader.split_dataset(opt.dataset_split_ratio)
    dataset_test = data_loader.create_dataloader(test_dataset, shuffle_batches=False)
    dataset_test_size = len(test_dataset)

    print('#test images = %d' % dataset_test_size)
    writer = Writer(opt)
    run_test(name=opt.name, writer=writer, dataset_test=dataset_test)
