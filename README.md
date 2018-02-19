# NotHotDog
Remember silicon valley's seefood app? I trained a neural network binary classifier using tensorflow to recognize hot dog. Because.., why not?

It only used around 800 examples for training. So obviously, accuracy is in the dust (Around 56%. Which I think is quite impressive for only 800 examples though). But you can improve it by adding more images. Btw, you don't have to preprocess the data. I've written a script to do that for you. 

Step1: Add more images.
Step2: Run preprocessor script (image_data.py)
Step3: Run the neural network

That's it. Thank you

P.S: It uses only 1 hidden layer. If you use more hidden layers, make sure you have enough data otherwise it would overfit. 
