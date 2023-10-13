## Stage I: Contrastive Learning
1) Here we use a contrastive learning framework to train the basic speaker encoder. The EER (Vox_O) in our paper is 7.02%. We get this result in 60 training epochs. In fact, the results may get better if train for more epochs.
2) This module is modified from the Stage I from [Self-supervised speaker recognition with loss-gated learning].(https://github.com/TaoRuijie/Loss-Gated-Learning). In my experience, the channels of ECAPA-TDNN is change from 512 to 1024, which can improve the EER from 7.5% to 7.0% in Vox\_O.
