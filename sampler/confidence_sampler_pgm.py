import math
from tqdm import tqdm
import numpy as np
import random
import torch


class ConfidenceSampler(object):
    def __init__(self, sm_temp=1, w=3, randomize="linear", r_temp=4.5, sched_mode="arccos", step=12):
        super().__init__()
        self.sm_temp = sm_temp
        self.w = w
        self.randomize = randomize
        self.r_temp = r_temp
        self.sched_mode = sched_mode
        self.step = step

    def __str__(self):
        s = f"Scheduler: {self.sched_mode}, Steps: {self.step}, " \
            f"SoftMax temp: {self.sm_temp}, CFG w: {self.w}, " \
            f"Randomize temp: {self.r_temp}"
        return s

    @staticmethod
    def build_sequence(step, input_size, mode="arccos", leave=False, verbose=True):
        """ Create a sampling scheduler
           :param
            step  -> int:  number of prediction during inference
            mode  -> str:  the rate of value to unmask
            leave -> bool: tqdm arg on either to keep the bar or not
           :return
            scheduler -> torch.LongTensor(): the list of token to predict at each step
        """
        r = torch.linspace(0, 1, step)
        if mode == "root":  # root scheduler
            val_to_mask = r ** .5
        elif mode == "linear":  # linear scheduler
            val_to_mask = r
        elif mode == "square":  # square scheduler
            val_to_mask = r ** 2
        elif mode == "cosine":  # cosine scheduler
            val_to_mask = 1 - torch.cos(r * math.pi * .5)
        elif mode == "arccos":  # arc cosine scheduler
            val_to_mask = 1 - (torch.arccos(r) / (math.pi * .5))
        else:
            return

        # fill the scheduler by the ratio of tokens to predict at each step
        sche = val_to_mask * input_size**2
        
        return sche.int()
        # return tqdm(sche.int(), leave=leave) if verbose else sche.int()
    
    @torch.no_grad()
    def __call__(self, trainer, init_code=None, nb_sample=50, labels=None, verbose=True):
        """ Generate sample with the trainer model
           :param
            init_code   -> torch.LongTensor: nb_sample x 16 x 16, the starting initialization code
            nb_sample   -> int:              the number of image to generated
            labels      -> torch.LongTensor: the list of classes to generate
            sm_temp     -> float:            the temperature before softmax
            w           -> float:            scale for the classifier free guidance
            randomize   -> str:              linear|warm_up|random|no, either or not to add randomness
            r_temp      -> float:            temperature for the randomness
            sched_mode  -> str:              root|linear|square|cosine|arccos, the shape of the scheduler
            step:       -> int:              number of step for the decoding
           :return
            x          -> torch.FloatTensor: nb_sample x 3 x 256 x 256, the generated images
            code       -> torch.LongTensor:  nb_sample x step x 16 x 16, the code corresponding to the generated images
        """
        model = trainer.ema if trainer.args.use_ema else trainer.vit
        if trainer.args.is_multi_gpus:
            model = model.module
        model_length = model.config.model.length
        l_codes = []  # Save the intermediate codes predicted
        l_mask = []   # Save the intermediate masks
        if labels is None:  # Default classes generated
            # goldfish, chicken, tiger cat, hourglass, ship, dog, race car, airliner, teddy bear, random
            labels = [1, 7, 282, 604, 724, 179, 751, 404, 850] + [random.randint(0, 999) for _ in range(nb_sample - 9)]
            labels = torch.LongTensor(labels[:nb_sample]).to(trainer.args.device)

        drop = torch.ones(nb_sample, dtype=torch.bool).to(trainer.args.device)
        if init_code is not None:  
            code = init_code
        else:  
            code = torch.full(size=(nb_sample, 1), 
                   fill_value=trainer.args.bos_value).to(trainer.args.device)
              
            clean_positions = torch.zeros(size=(nb_sample, 1), 
                                dtype=torch.int64).to(trainer.args.device)
            noisy_positions = torch.arange(start=1, 
                                        end=model_length, 
                                        dtype=torch.int64)[None
                                            ].repeat(nb_sample, 1).to(trainer.args.device)
            concrete_lengths = torch.ones(size=(nb_sample,),           
                                dtype=torch.int64).to(trainer.args.device)
            
        scheduler = self.build_sequence(self.step, trainer.input_size, mode=self.sched_mode, verbose=verbose)

        scheduler[-1] += model_length - 1  - scheduler[-2] 
        for index, num_unmasked in enumerate(tqdm(scheduler)):
            num_unmasked = scheduler[index] - scheduler[index - 1] if index > 0 else max(scheduler[index], 1)
            
            if self.w != 0: # Model Prediction CFG
                logit = model(torch.cat([code, code], dim=0),
                              y=torch.cat([labels, labels], dim=0),
                              drop_label=torch.cat([~drop, drop], dim=0),
                              clean_positions=torch.cat([clean_positions, clean_positions], dim=0),
                              noisy_positions=torch.cat([noisy_positions, noisy_positions], dim=0),
                              concrete_lengths=torch.cat([concrete_lengths, concrete_lengths], dim=0),
                              use_inference_mode=True)
                logit_c, logit_u = torch.chunk(logit, 2, dim=0)
                _w = self.w * (index / (len(scheduler) - 1))

                logit = (1 + _w) * logit_c - _w * logit_u
            else: # No CFG
                logit = model(code,
                            y=labels,
                            drop_label=~drop,
                            clean_positions=clean_positions, 
                            noisy_positions=noisy_positions, 
                            concrete_lengths=concrete_lengths, 
                            use_inference_mode=True)
            
            prob = torch.softmax(logit * self.sm_temp, -1)

            pred_code = torch.multinomial(prob.flatten(0,1), num_samples=1)
            pred_code = pred_code.reshape(prob.shape[:-1])
            conf = torch.gather(prob, -1, pred_code.unsqueeze(-1)).squeeze()

            if self.randomize == "linear":  # add gumbel noise decreasing over the sampling process
                ratio = (index / (len(scheduler) - 1))
                rand = self.r_temp * np.random.gumbel(size=conf.shape) * (1 - ratio)
                conf = torch.log(conf) + torch.from_numpy(rand).to(trainer.args.device)
            elif self.randomize == "warm_up":  # chose random sample for the 2 first steps
                conf = torch.rand_like(conf) if index < 2 else conf
            elif self.randomize == "random":  # chose random prediction at each step
                conf = torch.rand_like(conf)

            if conf.dim() == 1:
                conf = conf.unsqueeze(0)  
            num_unmasked = min(num_unmasked, conf.size(-1))  
            tresh_conf, index_mask = torch.topk(conf, k=num_unmasked, dim=-1)

            tresh_conf = tresh_conf[:, [-1]]
            f_mask = (conf >= tresh_conf)

            top_output = noisy_positions.masked_select(f_mask).view(clean_positions.size(0), -1)
 
            clean_positions = torch.cat([clean_positions, top_output], dim=1)
            noisy_positions = noisy_positions.masked_select(~f_mask).view(clean_positions.size(0), -1)
            
            concrete_lengths += num_unmasked
            code = torch.cat([code, pred_code.masked_select(f_mask).view(clean_positions.size(0), -1)], dim=1) 

        out = torch.empty_like(code).scatter_(dim=-1, 
                                       index=clean_positions, 
                                       src=code)
        out = torch.clamp(out[:, 1:].view(nb_sample, trainer.input_size, trainer.input_size), 0, trainer.args.codebook_size - 1)
        x = trainer.ae.decode_code(out)
        x = torch.clamp(x, -1, 1)
        trainer.vit.train()
        return x, None, None