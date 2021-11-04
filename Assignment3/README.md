# PHYS449

## Dependencies

- json
- numpy

## Running `main.py`

To run `main.py`, use

```sh
python3 main.py parameters.json [-1.0,-1.0] [1.0,1.0] 3
```

This model attempts to learn the value of dfdt at (x,y) and adds it scaled by some time step to (x,y) multiple times to plot a solution trajectory. Initially the model itself was going to be a simple linear layer however, it was not able to pick up on the nonlinearities of the 2nd field. So I added an extra layer with moe depth and added a tanh activation function. This version was able to learn both fields. Another thing to note is that Adam is much better at optimizing this loss function than SGD. SGD was not able to optimize the model enough for the 2nd field.
