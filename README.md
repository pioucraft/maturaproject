# maturaproject

You can run the web UI without training the model first, as the trained model is already included in the repository.

## how to train the mode :
- If you already have the nvcc command available, you can directly run the dev.sh script.
- If you don't have the nvcc command available, you can enter the nix shell with the command `nix-shell dev.nix` and then run the dev.sh script inside the shell.

## how to serve the web interface :
- Simply run the prod.sh script and the web interface will be available at http://localhost:8000

## sources :
- [But what is a convolution ? - 3B1B](https://www.youtube.com/watch?v=KuXjwB4LzSA)
- [CNNs - StatQuest](https://www.youtube.com/watch?v=HGwBXDKFk9I)
- [MIT 6.S191 - Convolutional Neural Networks](https://www.youtube.com/watch?v=oGpzWAlP5p0)
- [Dropout - PyTorch](https://docs.pytorch.org/docs/stable/generated/torch.nn.Dropout.html)
- [Kaiming Initialization - Geekforgeeks](https://www.geeksforgeeks.org/deep-learning/kaiming-initialization-in-deep-learning/)

