# BrainGAN

In the medical industry, misdiagnosis of disease is acknowledged as the most common and harmful medical errors as it can cost a human life. Radiologists require a lot of time to manually annotate and segment the images. Over the several years, deep learning has been playing a vital role in the field of computer vision. One of its key uses in the medical industry is to minimize misdiagnosis and the amount of time taken to annotate and segment the images. In this paper, a new deep learning approach for brain tumor classification on MRI Images is introduced. A deep neural network is pretrained as a discriminator in a generative adversarial network (GAN) on MR Images by using multi-scale gradient GAN (MSGGAN) with auxiliary classification to extract the features and to classify the images. In the discriminator, one of the fully connected blocks acts as an auxiliary classifier and the other fully connected block acts as an adversarial. The fully connected layers of the auxiliary block are fine-tuned to classify the type of tumor. The proposed approach is tested on two publicly available MRI datasets as a whole, consists of four types of brain tumors (glioma, meningioma, pituitary, and no tumor). Our proposed method achieved 98.57% accuracy which is better as compared to state of art methods. Also, our method appears to be a useful technique when the availability of medical images is limited.

Here we published the codes, used in the paper.

# Citation

```
@INPROCEEDINGS{9496036,
  author={Yerukalareddy, Dinesh Reddy and Pavlovskiy, Evgeniy},
  booktitle={2021 IEEE Ural-Siberian Conference on Computational Technologies in Cognitive Science, Genomics and Biomedicine (CSGB)}, 
  title={Brain Tumor Classification based on MR Images using GAN as a Pre-Trained Model}, 
  year={2021},
  volume={},
  number={},
  pages={380-384},
  doi={10.1109/CSGB53040.2021.9496036}
  }
```
