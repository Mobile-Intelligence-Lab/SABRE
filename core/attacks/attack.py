import torch


class Attack(object):
    r"""
    Base class for EoT-PGD attacks.

    Adapted from torchattacks.
    """
    def __init__(self, name, model):
        self.attack = name
        self.model = model
        self.model_name = str(model).split("(")[0]

        self.training = model.training
        self.device = next(model.parameters()).device
        
        self._transform_label = self._get_label
        self._targeted = -1
        self._attack_mode = 'default'
        self._return_type = 'float'
        self._kth_min = 1

        self._target_map_function = None

    def forward(self, *input):
        raise NotImplementedError
            
    def set_mode_default(self):
        if self._attack_mode == 'only_default':
            self._attack_mode = "only_default"
        else:
            self._attack_mode = "default"
            
        self._targeted = -1
        self._transform_label = self._get_label
        
    def set_mode_targeted(self, target_map_function=None):
        if self._attack_mode == 'only_default':
            raise ValueError("Changing attack mode is not supported in this attack method.")
            
        self._attack_mode = "targeted"
        self._targeted = 1
        if target_map_function is None:
            self._target_map_function = lambda images, labels: labels
        else:
            self._target_map_function = target_map_function
        self._transform_label = self._get_target_label

    def set_mode_least_likely(self, kth_min=1):
        if self._attack_mode == 'only_default':
            raise ValueError("Changing attack mode is not supported in this attack method.")
            
        self._attack_mode = "least_likely"
        self._targeted = 1
        self._transform_label = self._get_least_likely_label
        self._kth_min = kth_min

    def set_return_type(self, type):
        if type == 'float':
            self._return_type = 'float'
        elif type == 'int':
            self._return_type = 'int'
        else:
            raise ValueError(type + " is not a valid type. [Options: float, int]")

    def save(self, data_loader, save_path=None, verbose=True):
        if (self._attack_mode == 'targeted') and (self._target_map_function is None):
            raise ValueError("save is not supported for target_map_function=None")
        
        if save_path is None:
            raise ValueError("Missing argument: save_path")

        image_list = []
        label_list = []

        correct = 0
        total = 0
        l2_distance = []
        
        total_batch = len(data_loader)

        training_mode = self.model.training
        for step, (images, labels) in enumerate(data_loader):
            adv_images = self.__call__(images, labels)

            batch_size = len(images)
            
            if save_path is not None:
                image_list.append(adv_images.cpu())
                label_list.append(labels.cpu())

            if self._return_type == 'int':
                adv_images = adv_images.float()/255

            if verbose:
                with torch.no_grad():
                    if training_mode:
                        self.model.eval()
                    outputs = self.model(adv_images)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    right_idx = (predicted == labels.to(self.device))
                    correct += right_idx.sum()
                    
                    delta = (adv_images - images.to(self.device)).view(batch_size, -1)
                    l2_distance.append(torch.norm(delta[~right_idx], p=2, dim=1))                    
                    acc = 100 * float(correct) / total
                    print('- Save Progress: %2.2f %% / Accuracy: %2.2f %% / L2: %1.5f' \
                          % ((step+1)/total_batch*100, acc, torch.cat(l2_distance).mean()), end='\r')

        if save_path is not None:
            x = torch.cat(image_list, 0)
            y = torch.cat(label_list, 0)            
            torch.save((x, y), save_path)
            print('\n- Save Complete!')

        if training_mode:
            self.model.train()
        
    def _get_label(self, inputs, labels):
        return labels
    
    def _get_target_label(self, inputs, labels):
        return self._target_map_function(inputs, labels)
    
    def _get_least_likely_label(self, inputs, labels):
        outputs = self.model(inputs)
        if self._kth_min < 0:
            pos = outputs.shape[1] + self._kth_min + 1
        else:
            pos = self._kth_min
        _, labels = torch.kthvalue(outputs.data, pos)
        labels = labels.detach_()
        return labels
    
    def _to_uint(self, images):
        r"""
        Return images as int.
        """
        return (images*255).type(torch.uint8)

    def _switch_model(self):
        if self.training:
            self.model.train()
        else:
            self.model.eval()

    def __str__(self):
        info = self.__dict__.copy()
        
        del_keys = ['model', 'attack']
        
        for key in info.keys():
            if key[0] == "_":
                del_keys.append(key)
                
        for key in del_keys:
            del info[key]
        
        info['attack_mode'] = self._attack_mode
        if info['attack_mode'] == 'only_default':
            info['attack_mode'] = 'default'
            
        info['return_type'] = self._return_type
        
        return self.attack + "(" + ', '.join('{}={}'.format(key, val) for key, val in info.items()) + ")"

    def __call__(self, *input, **kwargs):
        training_mode = self.model.training
        if training_mode:
            self.model.eval()
        inputs = self.forward(*input, **kwargs)
        if training_mode:
            self.model.train()
        
        if self._return_type == 'int':
            inputs = self._to_uint(inputs)

        return inputs
