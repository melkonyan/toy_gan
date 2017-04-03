A very simple GAN that tries to learn and reproduce a Gaussian distribution. Code is based on [this](http://blog.aylien.com/introduction-generative-adversarial-networks-code-tensorflow/) tutorial.

# Examples
 * One Gaussian distribution, plain GAN(cross-entropy loss), no mini-batches.
   * Gen loss: 0.745805
   * Dis loss: 2.23916
   
![alt tag](https://github.com/melkonyan/toy_gan/blob/master/images/batch.png)

 * One Gaussian distribution, plain GAN(cross-entropy loss), mini-batches.
   * Gen loss: 0.75175
   * Dis loss: 2.25065

![alt tag](https://github.com/melkonyan/toy_gan/blob/master/images/mini_batch.png)
 
