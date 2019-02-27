# CVAE\_pilot\_MD 



The program is designed to run autonomous MD simulations, monitored by CVAE, which would also access the MD simulations conditions and decide when to stop or restart an OpenMM simulation.



For now, there are three case studies were set up:

1. [Fs-peptide](https://github.com/hengma1001/CVAE_pilot_MD/tree/master/Fs-pep): 

   * Implicit water model
   * Toy case
   * Passed respawn of OpenMM simulation 
   * ready for implementation of retraining of CVAE

2. [P27](https://github.com/hengma1001/CVAE_pilot_MD/tree/master/P27):
   * Still working on it 

3. [ExAB](https://github.com/hengma1001/CVAE_pilot_MD/tree/master/ExAB):
   * The contact map is too big to be handled by CVAE (TensorFlow error)


