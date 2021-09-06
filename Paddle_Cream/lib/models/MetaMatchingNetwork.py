'''
本文件为原lib/models/MetaMatchingNetwork.py的转写
'''

from copy import deepcopy
import paddle
from paddle.autograd import backward
from ..utils.util import cross_entropy_loss_with_soft_target
from paddle.nn.functional import softmax

# Meta Matching Network
class MetaMatchingNetwork():
    def __init__(self, cfg):
        self.cfg = cfg

    # only update student network weights
    def update_student_weights_only(self, 
        random_cand, 
        optimizer, 
        model):
        past_params = optimizer._parameter_list
        optimizer._parameter_list = [param for param in model.rand_parameters(random_cand)]
        optimizer.step()
        optimizer._parameter_list = past_params

    # only update meta networks weights
    def update_meta_weights_only(self,  
        teacher_cand, 
        model, 
        optimizer):

        assert [param for param in model.rand_parameters(teacher_cand, True)][0].grad.max() > 0.

        past_params = optimizer._parameter_list
        optimizer._parameter_list = [param for param in model.rand_parameters(teacher_cand, self.cfg.SUPERNET.PICK_METHOD == 'meta')]
        optimizer.step()
        optimizer._parameter_list = past_params

    def simulate_sgd_update(self, w, g, optimizer):
        if type(g) == type(None):
            return 0.
        return -g * optimizer.get_lr() + w

    # split training images into several slices
    def get_minibatch_input(self, input):
        slice = self.cfg.SUPERNET.SLICE
        x = deepcopy(input[:slice].clone().detach())
        return x

    def calculate_1st_gradient(self, kd_loss, model, random_cand, optimizer):
        optimizer.clear_grad()
        backward(kd_loss, retain_graph = True)
        grad = []
        for param in model.rand_parameters(random_cand):
            grad.append(param.grad)
        return grad

    def calculate_2nd_gradient(self, validation_loss, model, optimizer, teacher_cand, students_weight):
        optimizer.clear_grad()

        backward(validation_loss, retain_graph = True)
        backward(students_weight[0], retain_graph = True)

        grad_teacher = []
        for param in model.rand_parameters(teacher_cand, 
            self.cfg.SUPERNET.PICK_METHOD == 'meta'):
            grad_teacher.append(param.grad)
        return grad_teacher

    # forward training data
    def forward_training(self, x, model, random_cand, teacher_cand, meta_value):
        output = model(x, random_cand)
        with paddle.no_grad():
            teacher_output = model(x, teacher_cand)
            soft_label = softmax(teacher_output, axis = 1)
        kd_loss = meta_value * cross_entropy_loss_with_soft_target(output, soft_label)
        return kd_loss

    # forward validation data
    def forward_validation(self, input, target, random_cand, model, loss_fn):
        slice = self.cfg.SUPERNET.SLICE
        x = input[slice:slice * 2].clone()
        output_2 = model(x, random_cand)
        validation_loss = loss_fn(output_2, target[slice:slice * 2])
        return validation_loss

    def isUpdate(self, current_epoch, batch_idx, prioritized_board):
        isUpdate = True
        isUpdate &= (current_epoch > self.cfg.SUPERNET.META_STA_EPOCH)
        isUpdate &= (batch_idx > 0)
        isUpdate &= (batch_idx % self.cfg.SUPERNET.UPDATE_ITER == 0)
        isUpdate &= (prioritized_board.board_size() > 0)
        return isUpdate

    # update meta matching networks
    def run_update(self, input, target, random_cand, model, optimizer,
                   prioritized_board, loss_fn, current_epoch, batch_idx):
        if self.isUpdate(current_epoch, batch_idx, prioritized_board):
            x = self.get_minibatch_input(input)

            meta_value, teacher_cand = prioritized_board.select_teacher(model, random_cand)

            kd_loss = self.forward_training(x, model, random_cand, teacher_cand, meta_value)

            # calculate 1st gradient
            grad_1st = self.calculate_1st_gradient(kd_loss, model, random_cand, optimizer)

            # update student weights
            self.update_student_weights_only(random_cand, optimizer, model)
            # optimizer.step()
            
            if self.cfg.SUPERNET.PICK_METHOD == "meta":
                # simulate updated student weights
                students_weight = [
                    self.simulate_sgd_update(
                        p, grad_item, optimizer) for p, grad_item in zip(
                        model.rand_parameters(random_cand), grad_1st)]

                

                validation_loss = self.forward_validation(input, target, random_cand, model, loss_fn)

                # # calculate 2nd gradient
                grad_teacher = self.calculate_2nd_gradient(validation_loss, model, optimizer, teacher_cand, students_weight)

                # # update meta matching networks
                self.update_meta_weights_only(random_cand, teacher_cand, model, optimizer)

                # # delete internal variants
                del grad_teacher, grad_1st, x, validation_loss, kd_loss, students_weight
