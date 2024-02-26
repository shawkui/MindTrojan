import mindspore.nn as nn
import mindspore.ops as ops
import mindspore as ms
from mindarmour.adv_robustness.attacks.iterative_gradient_method import *

from mindspore.nn.loss import SoftmaxCrossEntropyWithLogits

class Custom_PGD(ProjectedGradientDescent):
    def __init__(self, network, network_ref, eps=0.3, eps_iter = 0.07, loss_fn=None, bound = 0.3, steps = 10, is_targeted = False, norm_level='inf',):
        super(Custom_PGD, self).__init__(network,
                                        eps=eps,
                                        eps_iter=eps_iter,
                                        bounds=bound,
                                        is_targeted=is_targeted,
                                        nb_iter=steps,
                                        loss_fn=loss_fn,
                                        norm_level=norm_level)

        self._network = network
        self._network_ref = network_ref


    def generate(self, inputs, labels):
        """
        Iteratively generate adversarial examples based on BIM method. The
        perturbation is normalized by projected method with parameter norm_level .

        Args:
            inputs (Union[numpy.ndarray, tuple]): Benign input samples used as references to
                create adversarial examples.
            labels (Union[numpy.ndarray, tuple]): Original/target labels. \
                For each input if it has more than one label, it is wrapped in a tuple.

        Returns:
            numpy.ndarray, generated adversarial examples.
        """
        
        '''
        * Change grad to grad.sign()
        * Change projection and CLIP
        
        '''
        
        inputs_image, inputs, labels = check_inputs_labels(inputs, labels)
        arr_x = inputs_image
        adv_x = copy.deepcopy(inputs_image) + self._eps

        if self._bounds is not None:
            clip_min, clip_max = self._bounds

        for _ in range(self._nb_iter):
            inputs_tensor = to_tensor_tuple(inputs)
            labels_tensor = to_tensor_tuple(labels)
            out_grad = self._loss_grad(*inputs_tensor, *labels_tensor)
            gradient = out_grad.asnumpy()
            sum_perturbs = adv_x - arr_x + self._eps_iter * np.sign(gradient)

            if self._norm_level == 'l2':
                sum_perturbs = sum_perturbs / np.sqrt(np.sum(sum_perturbs**2, axis=(1, 2, 3), keepdims=True))
            elif self._norm_level == 'l1':
                sum_perturbs = sum_perturbs / np.sum(np.abs(sum_perturbs), axis=(1, 2, 3), keepdims=True)
            elif self._norm_level == 'inf':
                sum_perturbs = np.clip(sum_perturbs, -self._eps, self._eps)

            adv_x = arr_x + sum_perturbs

            if self._bounds is not None:
                adv_x = np.clip(adv_x, clip_min, clip_max)

            if isinstance(inputs, tuple):
                inputs = (adv_x,) + inputs[1:]
            else:
                inputs = adv_x
        return adv_x



class GradWrapWithLoss(nn.Cell):
    def __init__(self, network):
        super(GradWrapWithLoss, self).__init__()
        self._grad_all = ops.composite.GradOperation(get_all=True, sens_param=False)
        self._network = network

    def construct(self, inputs, labels):
        gout = self._grad_all(self._network)(inputs, labels)
        return gout[0]
    

class WithLossCell_Share(nn.Cell):
    def __init__(self, network, network_ref, loss_fn):
        super(WithLossCell_Share, self).__init__()
        self._network = network
        self._loss_fn = loss_fn
        self._network_ref = network_ref

    def construct(self, data, label):
        # Get prediction
        per_logits = self._network(data)
        per_logits_ref = self._network_ref(data)
        
        pert_label = per_logits.argmax(1)
        pert_label_ref = per_logits_ref.argmax(1)
            
        success_attack = pert_label != label
        success_attack_ref = pert_label_ref != label
        common_attack = ops.logical_and(success_attack, success_attack_ref)
        shared_attack = ops.logical_and(common_attack, pert_label == pert_label_ref)

        sample_wise_ls = SoftmaxCrossEntropyWithLogits(sparse=True, reduction="none")
        
        loss_adv = ms.Tensor(0.0, ms.float32)
        if ops.logical_not(success_attack).sum()!=0:
            loss_adv += sample_wise_ls(per_logits, label)[ops.logical_not(success_attack)].sum()
        if ops.logical_not(success_attack_ref).sum()!=0:
            loss_adv += sample_wise_ls(per_logits_ref, label)[ops.logical_not(success_attack_ref)].sum()
        loss_adv = loss_adv/2/data.shape[0]

        # JS divergence
        p_model = ops.softmax(per_logits, axis = 1).clamp(min=1e-8)
        p_ref = ops.softmax(per_logits_ref, axis = 1).clamp(min=1e-8)
        mix_p = 0.5*(p_model+p_ref)
        loss_js = 0.5*(p_model*p_model.log() + p_ref*p_ref.log()) - 0.5*(p_model*mix_p.log() + p_ref*mix_p.log())
        loss_cross = loss_js[ops.logical_not(shared_attack)].sum()/data.shape[0]

        shared_loss = 0.01*loss_adv - loss_cross
        print(f'SAE info: Total Loss: {shared_loss}, Loss_adv: {loss_adv}, Loss_cross: {loss_cross}')
        return shared_loss


class Shared_PGD(ProjectedGradientDescent):
    def __init__(self, network, network_ref, eps=0.3, eps_iter = 0.07, loss_fn=None, bound = 0.3, steps = 10, is_targeted = False, norm_level='inf',):
        super(Shared_PGD, self).__init__(network,
                                        eps=eps,
                                        eps_iter=eps_iter,
                                        bounds=bound,
                                        is_targeted=is_targeted,
                                        nb_iter=steps,
                                        loss_fn=loss_fn,
                                        norm_level=norm_level)

        self._network = network
        self._network_ref = network_ref
        with_loss_cell = WithLossCell_Share(self._network, self._network_ref, loss_fn)
        self._loss_grad = GradWrapWithLoss(with_loss_cell)


    def generate(self, inputs, labels):
        """
        Iteratively generate adversarial examples based on BIM method. The
        perturbation is normalized by projected method with parameter norm_level .

        Args:
            inputs (Union[numpy.ndarray, tuple]): Benign input samples used as references to
                create adversarial examples.
            labels (Union[numpy.ndarray, tuple]): Original/target labels. \
                For each input if it has more than one label, it is wrapped in a tuple.

        Returns:
            numpy.ndarray, generated adversarial examples.
        """
        inputs_image, inputs, labels = check_inputs_labels(inputs, labels)
        arr_x = inputs_image
        adv_x = copy.deepcopy(inputs_image) + self._eps

        if self._bounds is not None:
            clip_min, clip_max = self._bounds

        for _ in range(self._nb_iter):
            inputs_tensor = to_tensor_tuple(inputs)
            labels_tensor = to_tensor_tuple(labels)
            out_grad = self._loss_grad(*inputs_tensor, *labels_tensor)
            gradient = out_grad.asnumpy()
            sum_perturbs = adv_x - arr_x + self._eps_iter * np.sign(gradient)

            if self._norm_level == 'l2':
                sum_perturbs = sum_perturbs / np.sqrt(np.sum(sum_perturbs**2, axis=(1, 2, 3), keepdims=True))
            elif self._norm_level == 'l1':
                sum_perturbs = sum_perturbs / np.sum(np.abs(sum_perturbs), axis=(1, 2, 3), keepdims=True)
            elif self._norm_level == 'inf':
                sum_perturbs = np.clip(sum_perturbs, -self._eps, self._eps)

            adv_x = arr_x + sum_perturbs

            if self._bounds is not None:
                adv_x = np.clip(adv_x, clip_min, clip_max)

            if isinstance(inputs, tuple):
                inputs = (adv_x,) + inputs[1:]
            else:
                inputs = adv_x
        return adv_x
    
