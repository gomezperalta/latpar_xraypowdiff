# latpar_xraypowdiff

This repository serves as electronic resource of the paper "Convolutional Neural Networks to assist the assessment of the lattice parameter from X-ray powder diffraction", which is published in Journal of Physical Chemistry A (DOI: https://dx.doi.org/10.1021/acs.jpca.3c03860)

This repository is organized in three directories:

<ul>
 <li> Codes: There you will find the executed codes to create the binary representation of the compound in the unit cell(quantitative input data), and the code used to train the networks.</li>
 <li> Results: This directory contains all the files generated after the training of each CNN. You can find the model (h5-file), the compounds in the training (dftraval.csv) and test(dftest.csv) sets, the assessed values by the CNN for the training (*_predtraval.npy) and test sets (*_predtest.npy), the actual values (files ytraval.npy and ytest.npy), and the output file (log-file). </li>
</ul>

It is important to mention that the input data (simulated xrds and quantitative input vector) are not here due to their size. However, you can check a larger version of that dataset in https://bit.ly/4iivld8 . This larger dataset is consequence of a spin-off of this work. For more details, visit https://github.com/gomezperalta/insightsXRDCNN.

Additionally, the simulated diffraction patterns were created and processed with the code available in https://github.com/gomezperalta/band-gap_pxrd)
