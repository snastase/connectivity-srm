# Connectivity-based shared response model

[![OpenNeuro](https://img.shields.io/badge/Data-OpenNeuro-teal)](https://openneuro.org/datasets/ds002345)
[![DataLad](https://img.shields.io/badge/Data-DataLad-orange)](http://datasets.datalad.org/?dir=/labs/hasson/narratives)

This repo accompanies a manuscript by Samuel A. Nastase, Yun-Fei Liu, Hanna Hillman, Kenneth A. Norman, and Uri Hasson published in *NeuroImage* ([Nastase et al., 2020](https://doi.org/10.1016/j.neuroimage.2020.116865)). Following the logic of connectivity hyperalignment ([Guntupalli et al., 2018](https://doi.org/10.1371/journal.pcbi.1006120)), we leverage intersubject functional correlations (ISFCs; [Simony et al., 2016](https://doi.org/10.1038/ncomms12141)) to estimate a single connectivity-based shared response model (SRM; [Chen et al., 2015](http://papers.nips.cc/paper/5855-a-reduced-dimension-fmri-shared-response-model)) across disjoint datasets. We evaluate this model on 10 publicly-available story-listening fMRI datasets from the [Narratives](https://snastase.github.io/datasets/ds002345) collection ([Nastase et al., 2019](https://openneuro.org/datasets/ds002345)).

![Alt text](./cSRM_schematic.png?raw=true&s=100 "cSRM schematic")

#### References
* Chen, P. H. C., Chen, J., Yeshurun, Y., Hasson, U., Haxby, J., & Ramadge, P. J. (2015). A reduced-dimension fMRI shared response model. In C. Cortes, N. D. Lawrence, D. D. Lee, M. Sugiyama, & R. Garnett, R. (Eds.), *Advances in Neural Information Processing Systems 28*. Paper presented at Neural Information Processing Systems 2015, Montreal (pp. 460-468). Red Hook, NY: Curran Associates, Inc. http://papers.nips.cc/paper/5855-a-reduced-dimension-fmri-shared-response-model

* Guntupalli, J. S., Feilong, M., & Haxby, J. V. (2018). A computational model of shared fine-scale structure in the human connectome. *PLOS Computational Biology*, *14*(4), e1006120. https://doi.org/10.1371/journal.pcbi.1006120

* Nastase, S. A., Liu, Y.-F., Hillman, H., Norman, K. A., & Hasson, U. (2019). Leveraging shared connectivity to aggregate heterogeneous datasets into a common response space. *NeuroImage*, 116865. https://doi.org/10.1016/j.neuroimage.2020.116865

* Nastase, S. A., Liu, Y.-F., Hillman, H., Zadbood, A., Hasenfratz, L., Keshavarzian, N., Chen, J., Honey, C. J., Yeshurun, Y., Regev, M., Nguyen, M., Chang, C. H. C., Baldassano, C. B., Lositsky, O., Simony, E., Chow, M. A., Leong, Y. C., Brooks, P. P., Micciche, E., Choe, G., Goldstein, A., Halchenko, Y. O., Norman, K. A., & Hasson, U. Narratives: fMRI data for evaluating models of naturalistic language comprehension. *OpenNeuro*, ds002345. https://doi.org/10.18112/openneuro.ds002345.v1.0.1

* Simony, E., Honey, C. J., Chen, J., Lositsky, O., Yeshurun, Y., Wiesel, A., & Hasson, U. (2016). Dynamic reconfiguration of the default mode network during narrative comprehension. *Nature Communications*, *7*, 12141. https://doi.org/10.1038/ncomms12141
