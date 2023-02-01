# latpar_xraypowdiff

This repository serves as electronic resource of the paper "Convolutional Neural Networks to assist the assessment of the lattice parameter from X-ray powder diffraction", which is currently under review.

This repository is organized in three directories:

<ul>
 <li> Codes: There you will find the executed codes to create the binary representation of the compound in the unit cell(quantitative input data), the output data, as well as the code used to train the networks and to calculate the lattice parameter from the CNN assessments via particle swarm optimization</li>
 <li> Collections: there you will find the csv-files with information about the organic compounds used to train the CNNs, as well as the output data (as npy-file)</li>
 <li> Results: This directory contains all the files generated after the training of each CNN. You can find the model (h5-file), the compounds in the training (dftraval.csv) and test(dftest.csv) sets, the assessed values by the CNN for the training (*_predtraval.npy) and test sets (*_predtest.npy), the actual values (files ytraval.npy and ytest.npy), and the output file (log-file) which recorded all what happened during the training of the CNN. The log-file is zipped in some directories since it is larger than 25 MB </li>
</ul>
