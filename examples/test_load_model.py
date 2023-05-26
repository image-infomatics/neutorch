#!/usr/bin/env python
# -*- coding: utf-8 -*-

fname = '/mnt/ceph/users/neuro/wasp_em/jwu/22_affs_whole_brain/model_135000.chkpt'

model = Model(1, 3)
				   
if 'preload' in self.cfg.train:
	fname = self.cfg.train.preload
else:
	fname = os.path.join(self.cfg.train.output_dir, 
		f'model_{self.cfg.train.iter_start}.chkpt')

if os.path.exists(fname):
	model = load_chkpt(model, fname)

# note that we have to wrap the nn.DataParallel(model) before 
# loading the model since the dictionary is changed after the wrapping
if self.num_gpus > 1:
	print(f'use {self.num_gpus} gpus!')
	model = torch.nn.parallel.DistributedDataParallel(
		model, device_ids=[self.local_rank],
		output_device=self.local_rank)

model.to('cuda')
